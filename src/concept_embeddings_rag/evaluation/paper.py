"""Phase 17, S6 (R5): every number of the paper is written here, from an artifact.

`cer p17-tables` reads the committed Phase 17 JSONs and the earlier phases' artifacts (read-only)
and writes `docs/paper/tables/*.tex` (booktabs tabulars), `tables/*.md` (the same numbers, same
rounding), `tables/numbers.tex` (one macro per number the prose cites, each commented with its
artifact path, SHA-256 prefix and JSON key), `tables/captions.tex` and `figures/*.tex`
(pgfplots with written coordinates). The hand-written prose cites macros only; `r5_violations`
is the guard that proves it.

Conventions: counts with a comma as thousands separator; shares as percentages with two
decimals; differences in percentage points with an explicit sign and two decimals; p-values with
two significant digits in scientific notation, `< 10^-300` when the exact test underflowed to 0.
Everything is deterministic: the same inputs give the same bytes.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from concept_embeddings_rag import config

# --------------------------------------------------------------------------------------------
# Names. LaTeX macro names hold letters only, so every digit is spelled out.

SET_TOKENS = {
    config.PHASE_17_HOTPOTQA: "Hotpot",
    config.PHASE_17_MUSIQUE: "Musique",
    config.PHASE_17_MULTIHOP_RAG: "Multihop",
}
SYSTEM_TOKENS = {
    "dense": "PTenA",
    "hybrid-bm25": "PTenB",
    "hybrid-bm25-entity-hop": "PTenC",
    "hybrid-bm25-seeded-hop": "PFourteen",
    config.PHASE_17_UNION: "Union",
}
JUDGE_TOKENS = {
    config.PHASE_17_LIGHT: "Light",
    config.PHASE_17_STRONG: "Strong",
    config.PHASE_17_DECISION: "Decision",
}
COMPARISON_TOKENS = {
    "j_p10a_vs_p10a": "PTenA",
    "j_p10b_vs_p10b": "PTenB",
    "j_p10c_vs_p10c": "PTenC",
    "j_p14_vs_p14": "PFourteen",
    "j_p14_vs_j_p10b": "Hop",
}
CROSS_TOKENS = {
    "strong_vs_light_p10b": "StrongVsLight",
    "decision_vs_strong_p10b": "DecisionVsStrong",
}
HOP_COMPARISON = "j_p14_vs_j_p10b"
K_TOKENS = {2: "Two", 5: "Five", 10: "Ten", 20: "Twenty", 100: "Hundred"}
BUDGET_TOKENS = {512: "FiveTwelve", 1024: "Kilo", 2048: "Standard", 4096: "FourK"}
DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
LADDER_SETS = {
    "hotpotqa_test_11": "HotpotTest",
    "musique_validation": "Musique",
    "multihop_rag": "Multihop",
}
LADDER_HEADERS = {
    "hotpotqa_test_11": "HotpotQA test-11",
    "musique_validation": "MuSiQue validation",
    "multihop_rag": "MultiHop-RAG",
}
SET_TICKS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "multihop-rag": "MultiHop-RAG"}
BENCHMARKS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "multihop": "MultiHop-RAG"}

FILES: dict[str, str] = {
    "outcome": "data/phase17/outcome.json",
    "metrics": "data/phase17/metrics.json",
    "pool": "data/phase17/pool.json",
    "integrity": "data/phase17/integrity.json",
    "scoring-light": "data/phase17/scoring-light.json",
    "scoring-strong": "data/phase17/scoring-strong.json",
    "scoring-decision": "data/phase17/scoring-decision.json",
    "published": "data/phase17/published-figures.json",
    "p7-gliner": "data/phase7/gliner/extraction.json",
    "p7-test": "data/phase7/test.json",
    "claude": "data/extraction/extraction-0107de3ae9b4e4a3.json",
    "p9-outcome": "data/phase9/outcome.json",
    "p9-extraction": "data/phase9/gliner/extraction.json",
    "p9-embedding": "data/phase9/embedding.json",
    "p10-outcome": "data/phase10/outcome.json",
    "p14-outcome": "data/phase14/outcome.json",
    "p15-outcome": "data/phase15/outcome.json",
    "p15-extraction": "data/phase15/nodes/extraction.json",
    "p15-embedding": "data/phase15/cache/embedding.json",
    "p16-outcome": "data/phase16/outcome.json",
    "p16-extraction": "data/phase16/nodes/extraction.json",
    "p16-embedding": "data/phase16/cache/embedding.json",
    "recipe": "docs/plans/phase_17/17.runpod_recipe.md",
    "spec": "docs/plans/phase_17/17.spec.md",
}


class PaperError(RuntimeError):
    """An artifact lacks something the paper needs; nothing is invented."""


# --------------------------------------------------------------------------------------------
# Sources


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    data: Any


class Sources:
    """The artifacts, by short name. The tests build one from fixtures."""

    def __init__(self, files: dict[str, Source]) -> None:
        self.files = files

    def __getitem__(self, name: str) -> Source:
        if name not in self.files:
            raise PaperError(f"source {name!r} is not loaded")
        return self.files[name]

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def get(self, name: str) -> Source | None:
        return self.files.get(name)


def load_sources(root: Path, names: Sequence[str] | None = None) -> Sources:
    """Read each artifact once, with its SHA-256 (the bytes as they are on disk)."""
    import json

    files: dict[str, Source] = {}
    for name in names or tuple(FILES):
        relative = FILES[name]
        path = Path(root) / relative
        if not path.exists():
            raise PaperError(f"{relative} is missing (source {name!r})")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        files[name] = Source(relative, digest, text if path.suffix == ".md" else json.loads(text))
    return Sources(files)


def lookup(data: Any, key: Sequence[str | int]) -> Any:
    for part in key:
        data = data[part]
    return data


def key_path(key: Sequence[str | int]) -> str:
    return ".".join(str(part) for part in key)


# --------------------------------------------------------------------------------------------
# Formatting. A Val holds the LaTeX and the Markdown spelling of one value.


class Val(NamedTuple):
    tex: str
    md: str


TEX_REPLACE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\u2013": "--",
    "\u2014": "---",
    "\u00d7": r"$\times$",
    "\u2019": "'",
    "\u2018": "`",
    "\u201c": "``",
    "\u201d": "''",
    "\u2265": r"$\geq$",
    "\u2264": r"$\leq$",
    "\u2248": r"$\approx$",
}


def tex_escape(text: str) -> str:
    return "".join(TEX_REPLACE.get(char, char) for char in text)


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def txt(text: str) -> Val:
    return Val(tex_escape(text), md_escape(text))


def count(n: int) -> Val:
    return Val(f"{int(n):,}", f"{int(n):,}")


def pct(ratio: float, unit: bool = False) -> Val:
    """A share (0-1) as a percentage with two decimals."""
    body = f"{100.0 * ratio:.2f}"
    return Val(body + (r"\%" if unit else ""), body + ("%" if unit else ""))


def share(part: int, whole: int, unit: bool = False) -> Val:
    return pct(part / whole, unit)


def delta_pp(value: float) -> Val:
    """A difference in percentage points, explicit sign, two decimals."""
    rounded = round(value, 2)
    if rounded > 0:
        return Val(f"+{rounded:.2f}", f"+{rounded:.2f}")
    if rounded < 0:
        return Val(f"$-${abs(rounded):.2f}", f"-{abs(rounded):.2f}")
    return Val("0.00", "0.00")


def p_value(p: float) -> Val:
    """Two significant digits, scientific notation; `< 10^-300` when the test underflowed."""
    if p == 0.0:
        return Val(r"$<10^{-300}$", "< 1e-300")
    mantissa, exponent = f"{p:.1e}".split("e")
    exp = int(exponent)
    return Val(rf"${mantissa}\times10^{{{exp}}}$", f"{mantissa}e{exp}")


def sci(value: float) -> Val:
    """A small positive measurement (a logit difference) in scientific notation."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return Val(rf"${mantissa}\times10^{{{int(exponent)}}}$", f"{mantissa}e{int(exponent)}")


def num(value: float, digits: int = 2) -> Val:
    body = f"{value:,.{digits}f}"
    return Val(body, body)


def plain_int(n: int) -> Val:
    return Val(str(int(n)), str(int(n)))


def label_val(text: str) -> Val:
    return Val(tex_escape(text), text)


def words(text: str) -> str:
    """`d5-hotpotqa` -> `DFiveHotpotqa`: a macro-safe name for a table stem."""
    pieces = re.split(r"[^A-Za-z0-9]+", text)
    out = []
    for piece in pieces:
        spelled = "".join(DIGIT_WORDS[int(c)] if c.isdigit() else c for c in piece)
        out.append(spelled[:1].upper() + spelled[1:])
    return "".join(out)


# --------------------------------------------------------------------------------------------
# Macros


@dataclass(frozen=True)
class Macro:
    name: str
    text: str
    path: str
    sha12: str
    key: tuple[str | int, ...] | None
    raw: Any
    fmt: Callable[[Any], Val] | None = field(default=None, compare=False)
    source: Source | None = field(default=None, compare=False)
    formula: str = ""


class MacroBuilder:
    def __init__(self) -> None:
        self.macros: list[Macro] = []
        self._names: set[str] = set()

    def _add(self, macro: Macro) -> None:
        if not re.fullmatch(r"[A-Za-z]+", macro.name):
            raise PaperError(f"macro name {macro.name!r} must hold letters only")
        if macro.name in self._names:
            raise PaperError(f"macro {macro.name} defined twice")
        self._names.add(macro.name)
        self.macros.append(macro)

    def direct(
        self, name: str, src: Source, key: Sequence[str | int], fmt: Callable[[Any], Val]
    ) -> Any:
        raw = lookup(src.data, key)
        self._add(Macro(name, fmt(raw).tex, src.path, src.sha256[:12], tuple(key), raw, fmt, src))
        return raw

    def derived(self, name: str, value: Val, sources: Sequence[Source], formula: str) -> None:
        shown = "; ".join(f"{s.path} sha256:{s.sha256[:12]}" for s in sources)
        self._add(Macro(name, value.tex, shown, "", None, None, None, None, formula))


def render_macros(macros: Sequence[Macro], sources: Sources) -> str:
    lines = [
        "% Generated by `cer p17-tables` (Phase 17, R5). Do not edit: every number the prose",
        "% cites is written here from an artifact; each macro carries its artifact path, the",
        "% SHA-256 prefix of that file and the JSON key path (or the formula, when derived).",
        "% Source files (full SHA-256; the outcome files of Phases 9, 10 and 14 are not in git,",
        "% so this line is where their digest is recorded):",
    ]
    for name in sorted(sources.files, key=lambda n: sources.files[n].path):
        src = sources.files[name]
        lines.append(f"%   {src.path} sha256:{src.sha256}")
    lines.append("")
    for macro in macros:
        if macro.key is not None:
            lines.append(f"% {macro.path} sha256:{macro.sha12} {key_path(macro.key)}")
        else:
            lines.append(f"% derived: {macro.formula}; from {macro.path}")
        lines.append(f"\\newcommand{{\\{macro.name}}}{{{macro.text}}}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Tables


class Rule:
    """A horizontal rule inside the body."""


@dataclass(frozen=True)
class Section:
    text: str


Row = list[Val] | Rule | Section


@dataclass
class Table:
    name: str
    caption: str
    header: list[str]
    align: str
    rows: list[Row]
    note: str = ""
    long: bool = False  # rendered as a longtable that carries its own caption macro


def _md_align(align: str) -> list[str]:
    cells = []
    for spec in re.findall(r"p\{[^}]*\}|[lrc]", align):
        cells.append("---:" if spec == "r" else ":---:" if spec == "c" else ":---")
    return cells


def render_long_tex_table(table: Table) -> str:
    """A `longtable` for tables that do not fit one page; the caption is its own macro."""
    head = " & ".join(tex_escape(h) for h in table.header) + " \\\\"
    out = [
        "% Generated by `cer p17-tables`; do not edit.",
        f"% caption: {table.caption}",
        f"\\begin{{longtable}}{{{table.align}}}",
        f"\\caption{{\\cap{words(table.name)}}}\\label{{tab:{table.name}}}\\\\",
        "\\toprule",
        head,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        head,
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for row in table.rows:
        if isinstance(row, (Rule, Section)):
            continue
        out.append(" & ".join(cell.tex for cell in row) + " \\\\")
    out.append("\\end{longtable}")
    return "\n".join(out) + "\n"


def render_tex_table(table: Table) -> str:
    if table.long:
        return render_long_tex_table(table)
    n = len(table.header)
    out = [
        "% Generated by `cer p17-tables`; do not edit.",
        f"% caption: {table.caption}",
        f"\\begin{{tabular}}{{{table.align}}}",
        "\\toprule",
        " & ".join(tex_escape(h) for h in table.header) + " \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(table.rows):
        if isinstance(row, Rule):
            out.append("\\midrule")
        elif isinstance(row, Section):
            if index:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{{n}}}{{l}}{{\\textbf{{{tex_escape(row.text)}}}}} \\\\")
        else:
            out.append(" & ".join(cell.tex for cell in row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def render_md_table(table: Table) -> str:
    n = len(table.header)
    out = [
        "<!-- Generated by `cer p17-tables`; do not edit. -->",
        f"**{md_escape(table.caption)}**",
        "",
        "| " + " | ".join(md_escape(h) for h in table.header) + " |",
        "| " + " | ".join(_md_align(table.align)) + " |",
    ]
    for row in table.rows:
        if isinstance(row, Rule):
            continue
        if isinstance(row, Section):
            out.append("| " + " | ".join([f"**{md_escape(row.text)}**"] + [""] * (n - 1)) + " |")
        else:
            out.append("| " + " | ".join(cell.md for cell in row) + " |")
    if table.note:
        out += ["", md_escape(table.note)]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# Helpers over the Phase 17 artifacts


def sets_of(outcome: Source) -> list[str]:
    present = outcome.data["sets"]
    return [name for name in config.PHASE_17_SETS if name in present]


def judges_of(outcome: Source) -> list[str]:
    return [j for j in JUDGE_TOKENS if j in outcome.data["judges"]]


def systems_of(outcome: Source) -> list[str]:
    return list(outcome.data["systems"])


def line_key(judge: str | None, system: str) -> str:
    return system if judge is None else f"{judge}:{system}"


def line_token(judge: str | None, system: str) -> str:
    return ("" if judge is None else JUDGE_TOKENS[judge]) + SYSTEM_TOKENS[system]


def line_order(outcome: Source, metrics: Source, set_name: str) -> list[tuple[str | None, str]]:
    """The 19 lines present in a set: unjudged systems, then each judge's systems and union."""
    present = metrics.data["sets"][set_name]["lines"]
    order: list[tuple[str | None, str]] = [(None, s) for s in systems_of(outcome)]
    for judge in judges_of(outcome):
        order += [(judge, s) for s in (*systems_of(outcome), config.PHASE_17_UNION)]
    return [(j, s) for j, s in order if line_key(j, s) in present]


def system_label(outcome: Source, system: str) -> str:
    return "union" if system == config.PHASE_17_UNION else str(outcome.data["systems"][system])


def line_label(outcome: Source, judge: str | None, system: str) -> str:
    system_text = system_label(outcome, system)
    if judge is None:
        return system_text
    return f"{outcome.data['judges'][judge]['label']}({system_text})"


def comparison_keys(outcome: Source, set_name: str, judge: str) -> list[str]:
    return list(outcome.data["sets"][set_name]["per_judge"][judge])


# --------------------------------------------------------------------------------------------
# Tables from the Phase 17 artifacts


def table_main(outcome: Source, metrics: Source) -> Table:
    judges = judges_of(outcome)
    header = ["System", "No judge", *(outcome.data["judges"][j]["label"] for j in judges)]
    rows: list[Row] = []
    for set_name in sets_of(outcome):
        block = outcome.data["sets"][set_name]
        n = block["n_questions"]
        rows.append(Section(f"{block['label']}, {n:,} questions"))
        fs = block["full_support_2048"]
        for system in (*systems_of(outcome), config.PHASE_17_UNION):
            row: list[Val] = [txt(system_label(outcome, system))]
            for judge in (None, *judges):
                key = line_key(judge, system)
                if key not in fs:
                    row.append(txt("--"))
                    continue
                cell_count = fs[key]["supported"]
                text = f"{cell_count:,} ({100 * cell_count / n:.2f})"
                mark = " (descriptive)" if fs[key]["descriptive"] else ""
                row.append(Val(text + ("$^\\dagger$" if mark else ""), text + mark))
            rows.append(row)
    return Table(
        "main-result",
        f"Full Support @{config.PHASE_9_BUDGETS[2]:,} tokens: questions supported (share, %), "
        "per system with and without each judge. Union lines are descriptive.",
        header,
        "l" + "r" * (len(judges) + 1),
        rows,
    )


def table_d5(outcome: Source, set_name: str) -> Table:
    block = outcome.data["sets"][set_name]
    header = ["Comparison", "Wins", "Losses", "Ties", "Delta (pp)", "p", "Label"]
    rows: list[Row] = []
    for judge in judges_of(outcome):
        if rows:
            rows.append(Rule())
        for key in comparison_keys(outcome, set_name, judge):
            comp = block["per_judge"][judge][key]
            label = txt(block["hop_label"][judge]) if key == HOP_COMPARISON else txt("")
            rows.append(
                [
                    txt(f"{comp['candidate']} vs {comp['control']}"),
                    count(comp["wins"]),
                    count(comp["losses"]),
                    count(comp["ties"]),
                    delta_pp(comp["delta_percentage_points"]),
                    p_value(comp["exact_two_sided_p"]),
                    label,
                ]
            )
    rows.append(Rule())
    for key in block["cross_judge"]:
        comp = block["cross_judge"][key]
        rows.append(
            [
                txt(f"{comp['candidate']} vs {comp['control']}"),
                count(comp["wins"]),
                count(comp["losses"]),
                count(comp["ties"]),
                delta_pp(comp["delta_percentage_points"]),
                p_value(comp["exact_two_sided_p"]),
                txt(""),
            ]
        )
    return Table(
        f"d5-{set_name}",
        f"{block['label']}, {block['n_questions']:,} questions: the paired comparisons of D5 "
        "(exact two-sided McNemar on Full Support); the label is that of J(P14) against J(P10-B).",
        header,
        "lrrrrrl",
        rows,
    )


def table_d6(outcome: Source, metrics: Source, set_name: str) -> Table:
    ks = list(metrics.data["ks"])
    header = [
        "Line",
        *(f"GR@{k}" for k in ks),
        *(f"FS@{k}" for k in ks),
        "nDCG@10",
    ]
    rows: list[Row] = []
    current: str | None = "-"
    lines = metrics.data["sets"][set_name]["lines"]
    for judge, system in line_order(outcome, metrics, set_name):
        if judge != current:
            current = judge
            title = "No judge" if judge is None else outcome.data["judges"][judge]["label"]
            rows.append(Section(title))
        line = lines[line_key(judge, system)]
        label = line_label(outcome, judge, system)
        mark = line["descriptive"]
        first = Val(tex_escape(label) + ("$^\\dagger$" if mark else ""), label)
        rows.append(
            [
                first,
                *(pct(line["gold_recall_at_k"][str(k)]) for k in ks),
                *(pct(line["full_support_at_k"][str(k)]) for k in ks),
                pct(line["ndcg_at_10"]),
            ]
        )
    label = outcome.data["sets"][set_name]["label"]
    return Table(
        f"d6-{set_name}",
        f"{label}: gold recall (GR), Full Support (FS) at k units, in percent of questions, "
        "and nDCG@10 (x100). Union lines are descriptive.",
        header,
        "l" + "r" * (2 * len(ks) + 1),
        rows,
    )


BUDGET_LINES: list[tuple[str | None, str]] = [
    (None, "dense"),
    (None, "hybrid-bm25"),
    (None, "hybrid-bm25-entity-hop"),
    (None, "hybrid-bm25-seeded-hop"),
    (config.PHASE_17_STRONG, "hybrid-bm25"),
    (config.PHASE_17_STRONG, "hybrid-bm25-entity-hop"),
    (config.PHASE_17_STRONG, "hybrid-bm25-seeded-hop"),
    (config.PHASE_17_STRONG, config.PHASE_17_UNION),
]


def budget_lines(outcome: Source, metrics: Source, set_name: str) -> list[tuple[str | None, str]]:
    present = {line_key(j, s) for j, s in line_order(outcome, metrics, set_name)}
    return [(j, s) for j, s in BUDGET_LINES if line_key(j, s) in present]


def table_budget(outcome: Source, metrics: Source) -> Table:
    budgets = [str(b) for b in metrics.data["budgets"]]
    header = ["Line", *(f"{int(b):,} tokens" for b in budgets)]
    rows: list[Row] = []
    for set_name in sets_of(outcome):
        n = outcome.data["sets"][set_name]["n_questions"]
        rows.append(Section(str(outcome.data["sets"][set_name]["label"])))
        lines = metrics.data["sets"][set_name]["lines"]
        for judge, system in budget_lines(outcome, metrics, set_name):
            line = lines[line_key(judge, system)]
            label = line_label(outcome, judge, system)
            mark = line["descriptive"]
            row: list[Val] = [Val(tex_escape(label) + ("$^\\dagger$" if mark else ""), label)]
            for b in budgets:
                c = line["budgets"][b]["full_support_count"]
                row.append(Val(f"{c:,} ({100 * c / n:.2f})", f"{c:,} ({100 * c / n:.2f})"))
            rows.append(row)
    return Table(
        "budget-curve",
        "Full Support at four token budgets: questions supported (share, %). "
        "Union lines are descriptive.",
        header,
        "l" + "r" * len(budgets),
        rows,
    )


def parse_spec_ceilings(spec_text: str) -> dict[str, dict[str, tuple[int, str]]]:
    """The spec's exploratory figures: set -> system -> (value, `ceiling` or `failures`)."""
    names = {
        "hotpotqa": config.PHASE_17_HOTPOTQA,
        "musique": config.PHASE_17_MUSIQUE,
        "multihop-rag": config.PHASE_17_MULTIHOP_RAG,
    }
    systems = {
        "p10-b": "hybrid-bm25",
        "p10-c": "hybrid-bm25-entity-hop",
        "p14": "hybrid-bm25-seeded-hop",
        "union": config.PHASE_17_UNION,
    }
    out: dict[str, dict[str, tuple[int, str]]] = {}
    inside = False
    current = ""
    for line in spec_text.splitlines():
        if line.startswith("| Corpus | System"):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or set(cells[0]) <= {"-", ":"} and cells[0]:
            continue
        if cells[0]:
            lowered = cells[0].lower()
            current = next((v for k, v in names.items() if lowered.startswith(k)), "")
        system = next((v for k, v in systems.items() if cells[1].lower().startswith(k)), "")
        match = re.match(r"([\d,]+)", cells[3])
        if current and system and match:
            kind = "failures" if "failures" in cells[3] else "ceiling"
            out.setdefault(current, {})[system] = (int(match.group(1).replace(",", "")), kind)
    return out


def ceiling_of(metrics: Source, set_name: str, system: str) -> int:
    """All gold inside the top 100 (the same for a system and each judge's reorder of it)."""
    lines = metrics.data["sets"][set_name]["lines"]
    for judge in (None, *JUDGE_TOKENS):
        key = line_key(judge, system)
        if key in lines:
            return int(lines[key]["ceiling"]["all_gold_inside"])
    raise PaperError(f"no ceiling for {system} in {set_name}")


def table_ceiling(outcome: Source, metrics: Source, spec: Source) -> Table:
    judges = judges_of(outcome)
    spec_figures = parse_spec_ceilings(spec.data)
    header = [
        "System",
        "All gold in top-100",
        "Spec figure",
        "FS no judge",
        *(f"FS {outcome.data['judges'][j]['label']}" for j in judges),
        "Fail. no judge",
        *(f"Fail. {outcome.data['judges'][j]['label']}" for j in judges),
    ]
    rows: list[Row] = []
    for set_name in sets_of(outcome):
        n = outcome.data["sets"][set_name]["n_questions"]
        rows.append(Section(f"{outcome.data['sets'][set_name]['label']}, {n:,} questions"))
        lines = metrics.data["sets"][set_name]["lines"]
        for system in (*systems_of(outcome), config.PHASE_17_UNION):
            ceil = ceiling_of(metrics, set_name, system)
            spec_cell = txt("--")
            figure = spec_figures.get(set_name, {}).get(system)
            if figure:
                spec_cell = txt(
                    f"{figure[0]:,}" + (" (failures)" if figure[1] == "failures" else "")
                )
            fs_cells: list[Val] = []
            fail_cells: list[Val] = []
            for judge in (None, *judges):
                key = line_key(judge, system)
                if key in lines:
                    c = lines[key]["ceiling"]
                    fs_cells.append(count(c["full_support_2048"]))
                    fail_cells.append(count(c["failures_inside"]))
                else:
                    fs_cells.append(txt("--"))
                    fail_cells.append(txt("--"))
            rows.append(
                [
                    txt(system_label(outcome, system)),
                    Val(f"{ceil:,} ({100 * ceil / n:.2f})", f"{ceil:,} ({100 * ceil / n:.2f})"),
                    spec_cell,
                    *fs_cells,
                    *fail_cells,
                ]
            )
    return Table(
        "ceiling",
        "The ceiling: questions with every gold unit inside the first 100 units of the "
        "system's list (union: of the whole pool), Full Support @2,048 achieved, and the "
        "failures that remain inside the ceiling; the spec's exploratory figures beside it.",
        header,
        "lrr" + "r" * (2 * (len(judges) + 1)),
        rows,
    )


def judge_dtype(judge: dict[str, Any]) -> str:
    dtype = judge["dtype"]
    if isinstance(dtype, list):
        return "/".join(dtype)
    return f"{dtype['encoder']} encoder, {dtype['heads']} heads"


def table_judges(outcome: Source, scoring: dict[str, Source]) -> Table:
    header = ["Judge", "Model", "Revision", "Parameters", "Max length", "Precision", "GPU"]
    rows: list[Row] = []
    for judge in judges_of(outcome):
        src = scoring[judge]
        info = src.data["judge"]
        model = str(info["name"])
        params = count(info["parameters"])
        if info["kind"] == config.PHASE_17_BI_ENCODER:
            model += f" over frozen {info['encoder']['name']}"
            params = Val(params.tex + " (heads)", params.md + " (heads)")
        rows.append(
            [
                txt(str(info["label"])),
                txt(model),
                txt(str(info["revision_served"])[:7]),
                params,
                count(info["max_length"]),
                txt(judge_dtype(info)),
                txt(str(src.data["host"]["gpu"])),
            ]
        )
    return Table(
        "judges",
        "The judges as served: pinned model, revision (first seven characters), parameters, "
        "maximum input length in tokens, precision and the GPU that scored.",
        header,
        "lllrrll",
        rows,
    )


def table_judge_cost(outcome: Source, scoring: dict[str, Source]) -> Table:
    header = ["Judge", "Set", "Pairs", "Pairs/s", "Wall (s)", "USD", "Truncated pairs"]
    rows: list[Row] = []
    for judge in judges_of(outcome):
        data = scoring[judge].data
        for set_name in sets_of(outcome):
            block = data["sets"][set_name]
            rows.append(
                [
                    txt(str(data["judge"]["label"])),
                    txt(str(block["label"])),
                    count(block["pairs"]),
                    num(block["pairs_per_second"], 1),
                    num(block["wall_seconds"], 1),
                    num(block["usd"], 4),
                    count(block["truncations"]["count"]),
                ]
            )
    return Table(
        "judge-cost",
        "Scoring cost per judge and set: pairs, throughput, wall time and the attributable "
        "USD at the pod's hourly rate.",
        header,
        "llrrrrr",
        rows,
    )


# --------------------------------------------------------------------------------------------
# Published figures (D7)


BENCHMARK_MARKERS = (
    ("multihop", "multihop"),
    ("musique", "musique"),
    ("hotpot", "hotpotqa"),
    ("wikipedia dump", "hotpotqa"),  # the 2017 abstracts dump of Yang et al. is HotpotQA's own
)


def benchmark_of(corpus: str) -> str:
    lowered = corpus.lower()
    for marker, bench in BENCHMARK_MARKERS:
        if marker in lowered:
            return bench
    raise PaperError(f"cannot place corpus {corpus!r} on a benchmark")


def figure_value(value: float) -> str:
    return f"{value:g}"


def published_tables(published: Source) -> list[Table]:
    rows_all = published.data["rows"]
    tables: list[Table] = []
    short_rows: list[Row] = []
    short_sections: list[tuple[str, int]] = []
    for bench, title in BENCHMARKS.items():
        subset = [r for r in rows_all if benchmark_of(r["corpus"]) == bench]
        if not subset:
            continue
        rows: list[Row] = []
        for r in subset:
            size = r["corpus_size"]
            questions = r["n_questions"]
            corpus = r["corpus"]
            if isinstance(size, int) and f"{size:,}" not in corpus and str(size) not in corpus:
                corpus += f" ({size:,})"
            cite = f"\\cite{{{r['citation_key']}}}"
            rows.append(
                [
                    txt(r["system"]),
                    txt(r["metric_as_published"]),
                    txt(figure_value(r["value"])),
                    txt(f"{questions:,}" if isinstance(questions, int) else "--"),
                    txt(corpus),
                    txt(r["training_or_llm"]),
                    txt(r["differences_from_ours"]),
                    Val(cite, r["citation_key"]),
                ]
            )
        tables.append(
            Table(
                f"published-{bench}",
                f"{title}: published figures, not a paired comparison. Each row was read in the "
                "paper itself; the last column says what differs from our setting.",
                [
                    "System",
                    "Metric",
                    "Value",
                    "Questions",
                    "Corpus",
                    "Training or LLM",
                    "Differences from ours",
                    "Source",
                ],
                "p{2.2cm}p{2.0cm}rrp{2.4cm}p{1.4cm}p{4.2cm}l",
                rows,
                long=True,
            )
        )
        for row in rows:
            if isinstance(row, list):
                short_rows.append([row[0], row[1], row[2], row[3], row[5], row[7]])
        short_sections.append((title, len(short_rows)))
    tables.append(published_short(short_rows, short_sections))
    return tables


def published_short(rows: list[Row], sections: list[tuple[str, int]]) -> Table:
    """One compact table over the three benchmarks: the figure, its unit of count and its
    source. The notes on how each differs from our setting stay in the per-benchmark tables."""
    body: list[Row] = []
    start = 0
    for title, end in sections:
        body.append(Section(title))
        body.extend(rows[start:end])
        start = end
    return Table(
        "published-short",
        "Published figures of other systems on the three benchmarks, not a paired comparison "
        "with ours. Each row was read in the paper itself; the metric and the unit are the "
        "paper's own, and how each differs from our setting is in the appendix.",
        ["System", "Metric", "Value", "Questions", "Training or LLM", "Source"],
        "p{5.2cm}p{6.4cm}rrp{2.1cm}l",
        body,
    )


# --------------------------------------------------------------------------------------------
# Earlier phases: ladder, held-out comparisons, indexing cost


def table_ladder(p16: Source) -> Table:
    ladder = p16.data["d6_ladder"]
    columns = list(ladder["columns"])
    header = ["System", *(LADDER_HEADERS.get(c, c) for c in columns)]
    rows: list[Row] = []
    for rung in ladder["rungs"]:
        row: list[Val] = [txt(rung["label"])]
        for column in columns:
            cell = rung[column]
            row.append(
                Val(
                    f"{cell['supported']:,} ({cell['full_support_percent']:.2f})",
                    f"{cell['supported']:,} ({cell['full_support_percent']:.2f})",
                )
            )
        rows.append(row)
    sizes = {c: ladder["rungs"][0][c]["n_questions"] for c in columns}
    note = "Questions per column: " + ", ".join(
        f"{LADDER_SETS.get(c, c)} {n:,}" for c, n in sizes.items()
    )
    return Table(
        "ladder",
        "The line's ladder at Full Support @2,048 on HotpotQA test-11, MuSiQue validation and "
        "MultiHop-RAG: questions supported (share, %). The columns differ in corpus, "
        "questions and size.",
        header,
        "l" + "r" * len(columns),
        rows,
        note,
    )


@dataclass(frozen=True)
class EarlierRow:
    """One earlier-phase comparison: where it lives and what to call it."""

    token: str
    phase: str
    setting: str
    comparison: str
    source: str
    prefix: tuple[str, ...]
    counts: tuple[str, str] | None = None


EARLIER_ROWS: list[EarlierRow] = [
    EarlierRow(
        "PhNineHop",
        "9",
        "FullWiki, 5,405 train questions",
        "Entity Hop (P9-C) vs Dense",
        "p9-outcome",
        (),
    ),
    EarlierRow(
        "PhNineBm",
        "9",
        "FullWiki, 5,405 train questions",
        "Entity Hop (P9-C) vs Dense + BM25 (descriptive)",
        "p9-outcome",
        ("secondary_entity_vs_bm25",),
    ),
    EarlierRow(
        "PhTenThree",
        "10",
        "FullWiki, test-10, 5,000 train questions",
        "Dense + BM25 + Entity Hop vs Dense + BM25",
        "p10-outcome",
        (),
    ),
    EarlierRow(
        "PhFourteenPTenC",
        "14",
        "FullWiki, test-11",
        "P14 vs P10-C",
        "p14-outcome",
        ("primary_p14_vs_p10c",),
    ),
    EarlierRow(
        "PhFourteenPTenB",
        "14",
        "FullWiki, test-11",
        "P14 vs P10-B",
        "p14-outcome",
        ("secondary_p14_vs_p10b",),
    ),
    EarlierRow(
        "PhFourteenTenCvsB",
        "14",
        "FullWiki, test-11",
        "P10-C vs P10-B (descriptive)",
        "p14-outcome",
        ("descriptive_p10c_vs_p10b",),
    ),
    EarlierRow(
        "PhFifteenPFourteen",
        "15",
        "MuSiQue validation",
        "P14 vs P10-B",
        "p15-outcome",
        ("d5_primary_p14_vs_p10b",),
    ),
    EarlierRow(
        "PhFifteenTenCvsB",
        "15",
        "MuSiQue validation",
        "P10-C vs P10-B",
        "p15-outcome",
        ("d6_secondary", "p10c_vs_p10b"),
    ),
    EarlierRow(
        "PhFifteenFourteenvsC",
        "15",
        "MuSiQue validation",
        "P14 vs P10-C",
        "p15-outcome",
        ("d6_secondary", "p14_vs_p10c"),
    ),
    EarlierRow(
        "PhFifteenBvsA",
        "15",
        "MuSiQue validation",
        "P10-B vs P10-A",
        "p15-outcome",
        ("d6_secondary", "p10b_vs_p10a"),
    ),
    EarlierRow(
        "PhSixteenPFourteen",
        "16",
        "MultiHop-RAG, 2,255 queries",
        "P14 vs P10-B",
        "p16-outcome",
        ("d5_primary_p14_vs_p10b",),
    ),
    EarlierRow(
        "PhSixteenTenCvsB",
        "16",
        "MultiHop-RAG, 2,255 queries",
        "P10-C vs P10-B",
        "p16-outcome",
        ("d6_secondary", "p10c_vs_p10b"),
    ),
    EarlierRow(
        "PhSixteenFourteenvsC",
        "16",
        "MultiHop-RAG, 2,255 queries",
        "P14 vs P10-C",
        "p16-outcome",
        ("d6_secondary", "p14_vs_p10c"),
    ),
    EarlierRow(
        "PhSixteenBvsA",
        "16",
        "MultiHop-RAG, 2,255 queries",
        "P10-B vs P10-A",
        "p16-outcome",
        ("d6_secondary", "p10b_vs_p10a"),
    ),
]


def earlier_fields(row: EarlierRow) -> dict[str, tuple[str, ...]]:
    """JSON keys of wins, losses, ties, delta, p and n inside the comparison block."""
    return {
        "wins": (*row.prefix, "wins"),
        "losses": (*row.prefix, "losses"),
        "ties": (*row.prefix, "ties"),
        "delta": (*row.prefix, "delta_percentage_points"),
        "p": (*row.prefix, "exact_two_sided_p"),
        "n": (*row.prefix, "n_questions"),
    }


def table_earlier(sources: Sources) -> Table:
    header = ["Phase", "Set", "Comparison", "Wins", "Losses", "Ties", "Delta (pp)", "p"]
    rows: list[Row] = []
    for item in EARLIER_ROWS:
        if item.source not in sources:
            continue
        data = sources[item.source].data
        keys = earlier_fields(item)
        rows.append(
            [
                txt(item.phase),
                txt(item.setting),
                txt(item.comparison),
                count(lookup(data, keys["wins"])),
                count(lookup(data, keys["losses"])),
                count(lookup(data, keys["ties"])),
                delta_pp(lookup(data, keys["delta"])),
                p_value(lookup(data, keys["p"])),
            ]
        )
    return Table(
        "earlier-comparisons",
        "Paired comparisons of Phases 9 to 16 on Full Support @2,048 (exact two-sided "
        "McNemar), from their own outcome files.",
        header,
        "llp{4.6cm}rrrrr",
        rows,
    )


COST_SOURCES = [
    ("Nine", "FullWiki, 5,233,329 paragraphs", "p9-extraction", "p9-embedding"),
    ("Fifteen", "MuSiQue pool", "p15-extraction", "p15-embedding"),
    ("Sixteen", "MultiHop-RAG corpus", "p16-extraction", "p16-embedding"),
]


def table_hop_cost(sources: Sources) -> Table:
    header = ["Corpus", "Extractor", "Units", "Wall (s)", "Units/s", "USD", "Embedding USD"]
    rows: list[Row] = []
    if "claude" in sources:
        c = sources["claude"].data
        rows.append(
            [
                txt("Phase 5 pool, 19,366 paragraphs"),
                txt(str(c["model"])),
                count(c["n_units"]),
                txt("--"),
                txt("--"),
                num(float(c["actual_usd"]), 2),
                txt("--"),
            ]
        )
    if "p7-gliner" in sources:
        g = sources["p7-gliner"].data
        rows.append(
            [
                txt("Phase 5 pool, 19,366 paragraphs"),
                txt(str(g["model"])),
                count(g["n_units"]),
                num(g["seconds"], 1),
                num(g["paragraphs_per_second"], 1),
                num(g["usd"], 2),
                txt("--"),
            ]
        )
    for _token, corpus, ext, emb in COST_SOURCES:
        if ext not in sources:
            continue
        e = sources[ext].data
        embed = num(sources[emb].data["attributable_usd"], 2) if emb in sources else txt("--")
        rows.append(
            [
                txt(corpus),
                txt(str(e["model"])),
                count(e["attempted"]),
                num(e["wall_seconds"], 1),
                num(e["paragraphs_per_second"], 1),
                num(e["attributable_usd"], 2),
                embed,
            ]
        )
    return Table(
        "hop-cost",
        "The Entity Hop's indexing cost: one entity-extraction pass over the corpus (and the "
        "BGE-small embedding of the same paragraphs) on a rented RTX 4090; the Claude row is "
        "the Phase 5 batch-priced run on the 19,366-paragraph pool.",
        header,
        "llrrrrr",
        rows,
    )


# --------------------------------------------------------------------------------------------
# Macros


def session_figures(recipe: Source) -> dict[str, Any]:
    text = str(recipe.data)
    balance = re.search(r"([0-9]+\.[0-9]+) to ([0-9]+\.[0-9]+) USD", text)
    duration = re.search(r"(\d+) h (\d+) min (\d+) s", text)
    if not balance or not duration:
        raise PaperError("the pod recipe does not carry the session's balance and duration")
    return {
        "before": float(balance.group(1)),
        "after": float(balance.group(2)),
        "hours": int(duration.group(1)),
        "minutes": int(duration.group(2)),
        "seconds": int(duration.group(3)),
    }


def phase17_macros(b: MacroBuilder, sources: Sources) -> None:
    outcome, metrics = sources["outcome"], sources["metrics"]
    pool = sources.get("pool")
    judges = judges_of(outcome)
    systems = (*systems_of(outcome), config.PHASE_17_UNION)
    for set_name in sets_of(outcome):
        s = SET_TOKENS[set_name]
        base: tuple[str | int, ...] = ("sets", set_name)
        n = b.direct(f"nQuestions{s}", outcome, (*base, "n_questions"), count)
        # Full Support @2,048, the 19 lines
        for judge, system in line_order(outcome, metrics, set_name):
            token = line_token(judge, system)
            fs_key = (*base, "full_support_2048", line_key(judge, system), "supported")
            supported = b.direct(f"fs{token}{s}", outcome, fs_key, count)
            b.derived(
                f"fsPct{token}{s}",
                share(supported, n, unit=True),
                [outcome],
                f"100 * {key_path(fs_key)} / {key_path((*base, 'n_questions'))}",
            )
        # D5
        for judge in judges:
            for cmp_key in comparison_keys(outcome, set_name, judge):
                token = JUDGE_TOKENS[judge] + COMPARISON_TOKENS[cmp_key]
                path = (*base, "per_judge", judge, cmp_key)
                b.direct(f"wins{token}{s}", outcome, (*path, "wins"), count)
                b.direct(f"losses{token}{s}", outcome, (*path, "losses"), count)
                b.direct(f"ties{token}{s}", outcome, (*path, "ties"), count)
                b.direct(f"delta{token}{s}", outcome, (*path, "delta_percentage_points"), delta_pp)
                b.direct(f"p{token}{s}", outcome, (*path, "exact_two_sided_p"), p_value)
            b.direct(
                f"label{JUDGE_TOKENS[judge]}{s}", outcome, (*base, "hop_label", judge), label_val
            )
        for cross_key in outcome.data["sets"][set_name]["cross_judge"]:
            token = "Cross" + CROSS_TOKENS[cross_key]
            path = (*base, "cross_judge", cross_key)
            b.direct(f"wins{token}{s}", outcome, (*path, "wins"), count)
            b.direct(f"losses{token}{s}", outcome, (*path, "losses"), count)
            b.direct(f"ties{token}{s}", outcome, (*path, "ties"), count)
            b.direct(f"delta{token}{s}", outcome, (*path, "delta_percentage_points"), delta_pp)
            b.direct(f"p{token}{s}", outcome, (*path, "exact_two_sided_p"), p_value)
        # D6: per-k metrics and the ceiling, from metrics.json
        lines = metrics.data["sets"][set_name]["lines"]
        for judge, system in line_order(outcome, metrics, set_name):
            token = line_token(judge, system)
            mbase = ("sets", set_name, "lines", line_key(judge, system))
            for k, word in K_TOKENS.items():
                b.direct(f"gr{word}{token}{s}", metrics, (*mbase, "gold_recall_at_k", str(k)), pct)
                b.direct(
                    f"fsAt{word}{token}{s}", metrics, (*mbase, "full_support_at_k", str(k)), pct
                )
            b.direct(f"ndcg{token}{s}", metrics, (*mbase, "ndcg_at_10"), pct)
            b.direct(f"ceilFail{token}{s}", metrics, (*mbase, "ceiling", "failures_inside"), count)
            n_questions = outcome.data["sets"][set_name]["n_questions"]
            for budget, word in BUDGET_TOKENS.items():
                supported = lines[line_key(judge, system)]["budgets"][str(budget)][
                    "full_support_count"
                ]
                b.derived(
                    f"fsBudget{word}{token}{s}",
                    share(supported, n_questions),
                    [metrics],
                    f"sets.{set_name}.lines.{line_key(judge, system)}.budgets.{budget}"
                    ".full_support_count / n_questions",
                )
        for system in systems:
            for judge in (None, *JUDGE_TOKENS):
                key = line_key(judge, system)
                if key in lines:
                    b.direct(
                        f"ceil{SYSTEM_TOKENS[system]}{s}",
                        metrics,
                        ("sets", set_name, "lines", key, "ceiling", "all_gold_inside"),
                        count,
                    )
                    break
        spec_figures = parse_spec_ceilings(sources["spec"].data) if "spec" in sources else {}
        for system, (value, kind) in spec_figures.get(set_name, {}).items():
            token = SYSTEM_TOKENS[system]
            b.derived(
                f"spec{'Fail' if kind == 'failures' else 'Ceil'}{token}{s}",
                count(value),
                [sources["spec"]],
                f"the spec's exploratory table, {set_name} / {system} ({kind})",
            )
        # pool
        if pool is not None and set_name in pool.data["sets"]:
            pbase = ("sets", set_name)
            b.direct(f"poolMean{s}", pool, (*pbase, "size", "mean"), lambda v: num(v, 2))
            b.direct(f"poolMedian{s}", pool, (*pbase, "size", "median"), lambda v: num(v, 0))
            b.direct(f"poolMax{s}", pool, (*pbase, "size", "max"), count)
            b.direct(f"poolUnits{s}", pool, (*pbase, "distinct_units"), count)
            b.direct(f"pairs{s}", pool, (*pbase, "pairs"), count)
    if pool is not None and "pairs" in pool.data:
        b.direct("pairsTotal", pool, ("pairs",), count)
    # judges: cost, throughput, determinism, truncation
    for judge in judges:
        name = f"scoring-{judge}"
        if name not in sources:
            continue
        src = sources[name]
        t = JUDGE_TOKENS[judge]
        b.direct(f"judgeParams{t}", src, ("judge", "parameters"), count)
        b.direct(f"judgeMaxLen{t}", src, ("judge", "max_length"), count)
        b.direct(f"judgeRate{t}", src, ("hourly_rate_usd",), lambda v: num(v, 2))
        for set_name in sets_of(outcome):
            s = SET_TOKENS[set_name]
            base = ("sets", set_name)
            b.direct(f"pairsPerSec{t}{s}", src, (*base, "pairs_per_second"), lambda v: num(v, 1))
            b.direct(f"usd{t}{s}", src, (*base, "usd"), lambda v: num(v, 4))
            b.direct(f"wallSec{t}{s}", src, (*base, "wall_seconds"), lambda v: num(v, 1))
            b.direct(f"truncCount{t}{s}", src, (*base, "truncations", "count"), count)
            det = (*base, "determinism")
            b.direct(f"detMaxDiff{t}{s}", src, (*det, "max_abs_difference"), sci)
            b.direct(f"detOrderChanged{t}{s}", src, (*det, "questions_order_changed"), count)
            b.direct(f"detQuestions{t}{s}", src, (*det, "questions"), count)
    if all(f"scoring-{j}" in sources for j in judges) and judges:
        total = sum(
            sources[f"scoring-{j}"].data["sets"][s]["usd"] for j in judges for s in sets_of(outcome)
        )
        b.derived(
            "usdGpuAttributable",
            num(total, 3),
            [sources[f"scoring-{j}"] for j in judges],
            "sum of sets.*.usd over the judges' manifests",
        )
    if "recipe" in sources:
        sess = session_figures(sources["recipe"])
        recipe = sources["recipe"]
        b.derived(
            "usdSession",
            num(sess["before"] - sess["after"], 4),
            [recipe],
            "balance before minus balance after, as recorded in the pod recipe",
        )
        b.derived("usdBalanceBefore", num(sess["before"], 4), [recipe], "balance before")
        b.derived("usdBalanceAfter", num(sess["after"], 4), [recipe], "balance after")
        b.derived("sessionHours", plain_int(sess["hours"]), [recipe], "session duration, hours")
        b.derived("sessionMinutes", plain_int(sess["minutes"]), [recipe], "duration, minutes")
        b.derived("sessionSeconds", plain_int(sess["seconds"]), [recipe], "duration, seconds")
    if "integrity" in sources:
        integrity = sources["integrity"]
        for set_name, block in integrity.data["d3"].items():
            if set_name not in SET_TOKENS:
                continue
            for system in block:
                b.direct(
                    f"dThree{SYSTEM_TOKENS[system]}{SET_TOKENS[set_name]}",
                    integrity,
                    ("d3", set_name, system, "observed"),
                    count,
                )


def earlier_macros(b: MacroBuilder, sources: Sources) -> None:
    for item in EARLIER_ROWS:
        if item.source not in sources:
            continue
        src = sources[item.source]
        keys = earlier_fields(item)
        b.direct(f"wins{item.token}", src, keys["wins"], count)
        b.direct(f"losses{item.token}", src, keys["losses"], count)
        b.direct(f"ties{item.token}", src, keys["ties"], count)
        b.direct(f"delta{item.token}", src, keys["delta"], delta_pp)
        b.direct(f"p{item.token}", src, keys["p"], p_value)
        b.direct(f"n{item.token}", src, keys["n"], count)
    if "p14-outcome" in sources:
        p14 = sources["p14-outcome"]
        b.direct("phFourteenAlpha", p14, ("selected", "alpha"), lambda v: num(v, 2))
    if "p9-outcome" in sources:
        p9 = sources["p9-outcome"]
        b.direct("phNineDenseCount", p9, ("dense_successes",), count)
        b.direct("phNineHopCount", p9, ("entity_successes",), count)
    if "p10-outcome" in sources:
        p10 = sources["p10-outcome"]
        b.direct("phTenControlCount", p10, ("control_successes",), count)
        b.direct("phTenCandidateCount", p10, ("candidate_successes",), count)
        b.direct(
            "phTenLatencyRatio", p10, ("latency_ratio_candidate_over_control",), lambda v: num(v, 2)
        )
    if "p16-outcome" in sources:
        p16 = sources["p16-outcome"]
        ladder = p16.data["d6_ladder"]
        for index, rung in enumerate(ladder["rungs"]):
            sys_token = SYSTEM_TOKENS[rung["system"]]
            for column, c_token in LADDER_SETS.items():
                if column not in rung:
                    continue
                base = ("d6_ladder", "rungs", index, column)
                supported = b.direct(
                    f"ladder{sys_token}{c_token}", p16, (*base, "supported"), count
                )
                n = rung[column]["n_questions"]
                b.derived(
                    f"ladderPct{sys_token}{c_token}",
                    share(supported, n, unit=True),
                    [p16],
                    f"100 * {key_path((*base, 'supported'))} / n_questions",
                )
        b.direct(
            "phSixteenPOneArticleHoldsGold",
            p16,
            ("d8_article_context", "p1_article_holds_gold_share"),
            pct,
        )
    if "p7-gliner" in sources and "claude" in sources:
        g, c = sources["p7-gliner"], sources["claude"]
        gliner_usd = b.direct("phSevenGlinerUsd", g, ("usd",), lambda v: num(v, 2))
        claude_usd = float(
            b.direct("phSevenClaudeUsd", c, ("actual_usd",), lambda v: num(float(v), 2))
        )
        b.derived(
            "phSevenCostRatio",
            num(claude_usd / gliner_usd, 0),
            [g, c],
            "claude actual_usd / gliner usd (the GLiNER figure is rounded up to 0.03 USD)",
        )
        n_units = c.data["n_units"]
        b.derived(
            "phSevenClaudeProjectedUsd",
            num(claude_usd / n_units * 5233329, 0),
            [c],
            "PROJECTION: linear, claude actual_usd per paragraph x 5,233,329 paragraphs",
        )
    if "p7-test" in sources:
        t = sources["p7-test"]
        dense = lookup(t.data, ("inherited_test_full_support", "dense"))
        claude = lookup(t.data, ("inherited_test_full_support", "hybrid-entity-hop"))
        gliner = lookup(t.data, ("metrics", "budget_2048", "full_support"))
        b.derived(
            "phSevenRetention",
            pct((gliner - dense) / (claude - dense), unit=True),
            [t],
            "(gliner - dense) / (claude - dense) at 2,048 tokens on the 1,400 test questions",
        )
    for token, _corpus, ext, emb in COST_SOURCES:
        if ext in sources:
            e = sources[ext]
            b.direct(f"hopCost{token}Units", e, ("attempted",), count)
            b.direct(f"hopCost{token}Seconds", e, ("wall_seconds",), lambda v: num(v, 1))
            b.derived(
                f"hopCost{token}Hours",
                num(e.data["wall_seconds"] / 3600, 2),
                [e],
                "wall_seconds / 3600",
            )
            b.direct(f"hopCost{token}Usd", e, ("attributable_usd",), lambda v: num(v, 2))
            b.direct(f"hopCost{token}PerSec", e, ("paragraphs_per_second",), lambda v: num(v, 1))
            b.direct(f"hopCost{token}Entities", e, ("entities",), count)
        if emb in sources:
            m = sources[emb]
            b.direct(f"hopCost{token}EmbedUsd", m, ("attributable_usd",), lambda v: num(v, 2))
            b.direct(f"hopCost{token}EmbedSeconds", m, ("wall_seconds",), lambda v: num(v, 1))


def published_macros(b: MacroBuilder, sources: Sources) -> None:
    if "published" not in sources:
        return
    src = sources["published"]
    rows = src.data["rows"]
    b.derived("nPublishedRows", count(len(rows)), [src], "len(rows)")
    b.derived("nPublishedExcluded", count(len(src.data["excluded"])), [src], "len(excluded)")
    for bench in BENCHMARKS:
        n = sum(1 for r in rows if benchmark_of(r["corpus"]) == bench)
        b.derived(f"nPublished{words(bench)}", count(n), [src], f"rows on {bench}")


DIGEST_MACROS = {
    "shaOutcome": "outcome",
    "shaMetrics": "metrics",
    "shaPool": "pool",
    "shaIntegrity": "integrity",
    "shaScoringLight": "scoring-light",
    "shaScoringStrong": "scoring-strong",
    "shaScoringDecision": "scoring-decision",
    "shaPublished": "published",
    "shaPhaseNineOutcome": "p9-outcome",
    "shaPhaseTenOutcome": "p10-outcome",
    "shaPhaseFourteenOutcome": "p14-outcome",
    "shaPhaseFifteenOutcome": "p15-outcome",
    "shaPhaseSixteenOutcome": "p16-outcome",
}


def digest_macros(b: MacroBuilder, sources: Sources) -> None:
    """The full SHA-256 of the artifacts the paper reads, for its reproducibility appendix."""
    for name, key in DIGEST_MACROS.items():
        if key in sources:
            src = sources[key]
            b.derived(name, Val(src.sha256, src.sha256), [src], "full SHA-256 of the file")


def build_macros(sources: Sources) -> list[Macro]:
    b = MacroBuilder()
    phase17_macros(b, sources)
    earlier_macros(b, sources)
    published_macros(b, sources)
    digest_macros(b, sources)
    return b.macros


# --------------------------------------------------------------------------------------------
# Figures (pgfplots, written coordinates)


@dataclass(frozen=True)
class Figure:
    name: str
    tex: str


def _coords(points: Sequence[tuple[float | int, float]]) -> str:
    return " ".join(f"({x},{y:.2f})" for x, y in points)


def _picture(options: str, plots: Sequence[tuple[str, str]], extra: str = "") -> str:
    lines = [
        "% Generated by `cer p17-tables`; do not edit. Coordinates are written from the artifacts.",
        "\\begin{tikzpicture}",
        f"\\begin{{axis}}[{options}]",
    ]
    for coords, legend in plots:
        lines.append(f"\\addplot coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{tex_escape(legend)}}}")
    if extra:
        lines.append(extra)
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


def figure_budget(outcome: Source, metrics: Source, set_name: str) -> Figure:
    budgets = list(metrics.data["budgets"])
    n = outcome.data["sets"][set_name]["n_questions"]
    lines = metrics.data["sets"][set_name]["lines"]
    wanted = [
        (None, "hybrid-bm25"),
        (None, "hybrid-bm25-seeded-hop"),
        (config.PHASE_17_STRONG, "hybrid-bm25"),
        (config.PHASE_17_STRONG, "hybrid-bm25-seeded-hop"),
    ]
    plots = []
    for judge, system in wanted:
        key = line_key(judge, system)
        if key not in lines:
            continue
        points = [
            (b, 100 * lines[key]["budgets"][str(b)]["full_support_count"] / n) for b in budgets
        ]
        plots.append((_coords(points), line_label(outcome, judge, system)))
    title = tex_escape(str(outcome.data["sets"][set_name]["label"]))
    ticks = ",".join(str(b) for b in budgets)
    options = (
        "width=\\columnwidth, height=5.5cm, xmode=log, log basis x=2, "
        f"xtick={{{ticks}}}, xticklabels={{{ticks}}}, xlabel={{Token budget}}, "
        "ylabel={Full Support (\\%)}, grid=major, legend pos=south east, "
        f"legend style={{font=\\scriptsize}}, title={{{title}}}"
    )
    return Figure(f"budget-{set_name}", _picture(options, plots))


def figure_delta(outcome: Source) -> Figure:
    sets = sets_of(outcome)
    plots = []
    for judge in judges_of(outcome):
        points = [
            (
                i,
                outcome.data["sets"][s]["per_judge"][judge]["j_p10b_vs_p10b"][
                    "delta_percentage_points"
                ],
            )
            for i, s in enumerate(sets)
        ]
        plots.append((_coords(points), str(outcome.data["judges"][judge]["label"])))
    labels = ",".join("{" + tex_escape(SET_TICKS.get(s, s)) + "}" for s in sets)
    options = (
        "ybar, width=\\columnwidth, height=6cm, bar width=8pt, enlarge x limits=0.25, "
        f"xtick={{{','.join(str(i) for i in range(len(sets)))}}}, xticklabels={{{labels}}}, "
        "x tick label style={font=\\scriptsize}, ylabel={Delta in Full Support (pp)}, "
        "grid=major, legend style={font=\\scriptsize, at={(0.5,-0.14)}, anchor=north, "
        "legend columns=3}, "
        "title={J(P10-B) against P10-B}"
    )
    return Figure(
        "delta-judge", _picture(options, plots, "\\draw (axis cs:-1,0) -- (axis cs:3,0);")
    )


def figure_ceiling(outcome: Source, metrics: Source) -> Figure:
    sets = sets_of(outcome)
    series = [
        ("P10-B ceiling", None, "hybrid-bm25", "ceiling"),
        ("P10-B", None, "hybrid-bm25", "fs"),
        ("J-strong(P10-B)", config.PHASE_17_STRONG, "hybrid-bm25", "fs"),
        ("P14 ceiling", None, "hybrid-bm25-seeded-hop", "ceiling"),
        ("P14", None, "hybrid-bm25-seeded-hop", "fs"),
        ("J-strong(P14)", config.PHASE_17_STRONG, "hybrid-bm25-seeded-hop", "fs"),
    ]
    plots = []
    for title, judge, system, kind in series:
        points = []
        for i, s in enumerate(sets):
            n = outcome.data["sets"][s]["n_questions"]
            lines = metrics.data["sets"][s]["lines"]
            key = line_key(judge, system)
            if key not in lines:
                continue
            c = lines[key]["ceiling"]
            value = c["all_gold_inside"] if kind == "ceiling" else c["full_support_2048"]
            points.append((i, 100 * value / n))
        plots.append((_coords(points), title))
    labels = ",".join("{" + tex_escape(SET_TICKS.get(s, s)) + "}" for s in sets)
    options = (
        "ybar, width=\\columnwidth, height=6.5cm, bar width=4pt, enlarge x limits=0.2, "
        f"xtick={{{','.join(str(i) for i in range(len(sets)))}}}, xticklabels={{{labels}}}, "
        "x tick label style={font=\\scriptsize}, ylabel={Questions (\\%)}, grid=major, "
        "legend style={font=\\scriptsize, at={(0.5,-0.14)}, anchor=north, legend columns=2}, "
        "title={Ceiling against achieved, Full Support @2,048}"
    )
    return Figure("ceiling", _picture(options, plots))


def build_figures(outcome: Source, metrics: Source) -> list[Figure]:
    figures = [figure_budget(outcome, metrics, s) for s in sets_of(outcome)]
    figures.append(figure_delta(outcome))
    figures.append(figure_ceiling(outcome, metrics))
    return figures


# --------------------------------------------------------------------------------------------
# Everything


def build_tables(sources: Sources) -> list[Table]:
    outcome, metrics = sources["outcome"], sources["metrics"]
    scoring = {j: sources[f"scoring-{j}"] for j in judges_of(outcome) if f"scoring-{j}" in sources}
    tables = [table_main(outcome, metrics)]
    tables += [table_d5(outcome, s) for s in sets_of(outcome)]
    tables += [table_d6(outcome, metrics, s) for s in sets_of(outcome)]
    tables.append(table_budget(outcome, metrics))
    if "spec" in sources:
        tables.append(table_ceiling(outcome, metrics, sources["spec"]))
    if scoring:
        tables.append(table_judges(outcome, scoring))
        tables.append(table_judge_cost(outcome, scoring))
    if "published" in sources:
        tables += published_tables(sources["published"])
    if "p16-outcome" in sources:
        tables.append(table_ladder(sources["p16-outcome"]))
    tables.append(table_earlier(sources))
    tables.append(table_hop_cost(sources))
    return tables


def render_captions(tables: Sequence[Table]) -> str:
    lines = ["% Generated by `cer p17-tables`; do not edit. One caption macro per table."]
    for table in tables:
        lines.append(f"\\newcommand{{\\cap{words(table.name)}}}{{{tex_escape(table.caption)}}}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Written:
    tables: int
    macros: int
    figures: int


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def write_all(sources: Sources, paper_dir: Path) -> Written:
    """Write the generated directories (overwriting their `.tex` and `.md` files)."""
    tables_dir, figures_dir = Path(paper_dir) / "tables", Path(paper_dir) / "figures"
    for directory in (tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in [*directory.glob("*.tex"), *directory.glob("*.md")]:
            old.unlink()
    tables = build_tables(sources)
    macros = build_macros(sources)
    figures = build_figures(sources["outcome"], sources["metrics"])
    for table in tables:
        _write(tables_dir / f"{table.name}.tex", render_tex_table(table))
        _write(tables_dir / f"{table.name}.md", render_md_table(table))
    _write(tables_dir / "numbers.tex", render_macros(macros, sources))
    _write(tables_dir / "captions.tex", render_captions(tables))
    for figure in figures:
        _write(figures_dir / f"{figure.name}.tex", figure.tex)
    return Written(len(tables), len(macros), len(figures))


# --------------------------------------------------------------------------------------------
# R5: the guard over the hand-written files

GENERATED_DIRS = ("tables", "figures")
# Constructs whose arguments may hold digits (identifiers, keys, options, file names) are
# removed before the scan: citations, references, labels, inputs, package and class options,
# pgfplots settings, bibliography, graphics, comments and four-digit years.
ALLOWED = [
    r"\\(?:cite[a-z]*|ref|eqref|autoref|label|input|include|bibliography|bibliographystyle"
    r"|usepackage|usetikzlibrary|usepgfplotslibrary|documentclass|includegraphics"
    r"|pgfplotsset|geometry|setlength|hypersetup|url|href)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}",
    r"(?<!\\)%[^\n]*",
    r"\b(?:19|20)\d{2}\b",
]
NUMBER_PATTERNS = [
    (r"\d\.\d", "a digit, a decimal point and a digit"),
    (r"\d\s*\\?%", "a digit followed by a percent sign"),
    (r"\d,\d{3}", "a thousands separator"),
    (r"\d\{,\}\d{3}", "a thousands separator"),
]


def r5_violations(paper_dir: Path) -> list[str]:
    """Hand-written `.tex` under `paper_dir` (outside tables/ and figures/) holding a number
    with a decimal point, a percent sign or a thousands separator."""
    root = Path(paper_dir)
    found: list[str] = []
    for path in sorted(root.rglob("*.tex")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in GENERATED_DIRS:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in ALLOWED:
            text = re.sub(pattern, " ", text)
        for number, what in NUMBER_PATTERNS:
            for match in re.finditer(number, text):
                line = text.count("\n", 0, match.start()) + 1
                found.append(f"{relative.as_posix()}:{line}: {what}: {match.group(0)!r}")
    return found
