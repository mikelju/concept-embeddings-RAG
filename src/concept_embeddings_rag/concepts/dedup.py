"""Near-duplicate concepts: detecting them, merging them, and recording the merge.

A dictionary that carries the same concept three times does not gain three
dimensions. Worse, it splits the activation of a unit across near-identical
atoms, which is precisely the co-activation Phase 4 is meant to travel over.

Two properties make this auditable rather than a matter of trust. The criterion
is a declared cosine threshold, and atoms are unit-norm, so cosine is a dot
product and the threshold means the same thing for every pair. And every merge is
written to a versioned log - members, their pairwise similarities, and the index
of the atom that replaced them - so the decisions can be checked afterwards.

The rule that matters most is what happens to `X`. It is **recomputed** by coding
the corpus against the merged dictionary. Summing the columns of the old `X` is
not admissible: the result would be the output of no coding at all, and the
weight semantics of HU-2 would stop holding the moment it were used.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from concept_embeddings_rag import config
from concept_embeddings_rag.concepts.coding import ConceptMatrix, code_corpus
from concept_embeddings_rag.concepts.dictionary import ConceptArtifactError, ConceptDictionary
from concept_embeddings_rag.embeddings.backend import l2_normalize


@dataclass(frozen=True)
class MergeGroup:
    """One group of atoms that collapsed into a single concept."""

    members: list[int]
    similarities: list[tuple[int, int, float]]
    resulting_index: int


@dataclass(frozen=True)
class MergeLog:
    """The record of a deduplication, kept as an artifact rather than printed."""

    source_key: str
    merged_key: str
    threshold: float
    k_before: int
    k_after: int
    merged_fraction: float
    max_merged_fraction: float
    exceeded_merge_budget: bool
    finding: str | None
    groups: list[MergeGroup]


def _gram(atoms: np.ndarray) -> np.ndarray:
    """Pairwise cosine between unit-norm atoms, which is their dot product."""
    dense = np.ascontiguousarray(atoms, dtype=np.float64)
    norms = np.linalg.norm(dense, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ConceptArtifactError(
            "atoms are not L2-normalized, so a cosine threshold does not mean the same "
            f"thing for every pair (norms range {norms.min():.4f} to {norms.max():.4f})"
        )
    return dense @ dense.T


def find_duplicate_groups(atoms: np.ndarray, *, threshold: float) -> list[list[int]]:
    """Group atoms whose pairwise cosine reaches `threshold`.

    Grouping is transitive by union-find: if a is a duplicate of b and b of c,
    all three name the same concept even when a and c fall just under the
    threshold. Only groups of more than one atom come back; the partition that
    includes the singletons is built by `deduplicate_dictionary`.
    """
    gram = _gram(atoms)
    parent = list(range(gram.shape[0]))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in np.argwhere(np.triu(gram, k=1) >= threshold):
        a, b = find(int(left)), find(int(right))
        if a != b:
            parent[max(a, b)] = min(a, b)

    grouped: dict[int, list[int]] = {}
    for index in range(gram.shape[0]):
        grouped.setdefault(find(index), []).append(index)
    return [sorted(members) for _, members in sorted(grouped.items()) if len(members) > 1]


def deduplicate_dictionary(
    dictionary: ConceptDictionary,
    *,
    threshold: float = config.MERGE_COSINE_THRESHOLD,
    max_merged_fraction: float = config.MAX_MERGED_FRACTION,
) -> tuple[ConceptDictionary, MergeLog]:
    """Merge near-duplicate atoms, returning the new dictionary and its merge log.

    The replacement atom is the L2-normalized mean of its group: it points where
    the group pointed, and it stays unit-norm so every later cosine keeps meaning
    the same thing.

    Removing more than `max_merged_fraction` of the dictionary is recorded as a
    finding about K rather than absorbed. It is not an error - a dictionary that
    collapses is exactly the evidence that K is oversized for this corpus, and
    aborting the sweep would throw that evidence away.
    """
    gram = _gram(dictionary.atoms)
    duplicates = find_duplicate_groups(dictionary.atoms, threshold=threshold)
    in_a_group = {index: members for members in duplicates for index in members}

    partition: list[list[int]] = []
    for index in range(dictionary.k):
        members = in_a_group.get(index)
        if members is None:
            partition.append([index])
        elif members[0] == index:
            partition.append(list(members))

    source_atoms = np.asarray(dictionary.atoms, dtype=np.float64)
    stacked = np.stack([source_atoms[members].mean(axis=0) for members in partition])
    merged_atoms = l2_normalize(stacked.astype(np.float32))

    merged = ConceptDictionary(
        atoms=merged_atoms,
        k=len(partition),
        seed=dictionary.seed,
        sparsity_param=dictionary.sparsity_param,
        max_iter=dictionary.max_iter,
        unit_set_hash=dictionary.unit_set_hash,
        model=dictionary.model,
        revision=dictionary.revision,
        code_version=dictionary.code_version,
        merged_from=[list(members) for members in partition],
        merge_threshold=threshold,
    )

    groups: list[MergeGroup] = []
    for resulting_index, members in enumerate(partition):
        if len(members) == 1:
            continue
        groups.append(
            MergeGroup(
                members=list(members),
                similarities=[
                    (left, right, float(gram[left, right]))
                    for position, left in enumerate(members)
                    for right in members[position + 1 :]
                ],
                resulting_index=resulting_index,
            )
        )

    removed = dictionary.k - merged.k
    fraction = removed / dictionary.k if dictionary.k else 0.0
    exceeded = fraction > max_merged_fraction
    finding = None
    if exceeded:
        finding = (
            f"deduplication at cosine {threshold} removed {removed} of {dictionary.k} atoms "
            f"({fraction:.1%}), above the declared budget of {max_merged_fraction:.1%}: this "
            f"dictionary is oversized for the corpus at k={dictionary.k}"
        )

    log = MergeLog(
        source_key=dictionary.key,
        merged_key=merged.key,
        threshold=threshold,
        k_before=dictionary.k,
        k_after=merged.k,
        merged_fraction=fraction,
        max_merged_fraction=max_merged_fraction,
        exceeded_merge_budget=exceeded,
        finding=finding,
        groups=groups,
    )
    return merged, log


def recode_after_merge(
    merged: ConceptDictionary,
    vectors: np.ndarray,
    unit_ids: Sequence[str],
    *,
    alpha: float,
    batch_size: int = config.CODING_BATCH_SIZE,
) -> ConceptMatrix:
    """Recompute `X` against the merged dictionary, by coding - never by summing.

    This function exists to be found by anyone tempted to add the columns of the
    old `X` together, which is the cheap route and the wrong one: a column sum is
    the output of no coding, so nothing in HU-2's weight semantics would still
    describe it.
    """
    return code_corpus(merged, vectors, unit_ids, alpha=alpha, batch_size=batch_size)


def _path_for(directory: Path, merged_key: str) -> Path:
    return Path(directory) / f"merge-log-{merged_key}.json"


def save_merge_log(log: MergeLog, directory: Path | str) -> Path:
    """Persist the merge log beside the dictionary it describes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path_for(directory, log.merged_key)
    path.write_text(
        json.dumps(
            {
                "source_key": log.source_key,
                "merged_key": log.merged_key,
                "threshold": log.threshold,
                "k_before": log.k_before,
                "k_after": log.k_after,
                "merged_fraction": log.merged_fraction,
                "max_merged_fraction": log.max_merged_fraction,
                "exceeded_merge_budget": log.exceeded_merge_budget,
                "finding": log.finding,
                "groups": [
                    {
                        "members": group.members,
                        "similarities": [list(pair) for pair in group.similarities],
                        "resulting_index": group.resulting_index,
                    }
                    for group in log.groups
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def load_merge_log(merged_key: str, directory: Path | str) -> MergeLog:
    """Read back the merge log for a deduplicated dictionary."""
    path = _path_for(Path(directory), merged_key)
    if not path.exists():
        raise ConceptArtifactError(f"no merge log for dictionary {merged_key} in {directory}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return MergeLog(
        source_key=str(payload["source_key"]),
        merged_key=str(payload["merged_key"]),
        threshold=float(payload["threshold"]),
        k_before=int(payload["k_before"]),
        k_after=int(payload["k_after"]),
        merged_fraction=float(payload["merged_fraction"]),
        max_merged_fraction=float(payload["max_merged_fraction"]),
        exceeded_merge_budget=bool(payload["exceeded_merge_budget"]),
        finding=payload["finding"],
        groups=[
            MergeGroup(
                members=[int(member) for member in group["members"]],
                similarities=[
                    (int(pair[0]), int(pair[1]), float(pair[2])) for pair in group["similarities"]
                ],
                resulting_index=int(group["resulting_index"]),
            )
            for group in payload["groups"]
        ],
    )
