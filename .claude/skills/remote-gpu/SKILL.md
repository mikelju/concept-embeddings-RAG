---
name: remote-gpu
description: How to run work this Windows ARM64 laptop cannot do (GPU extraction, large encodes, packages without win_arm64 wheels) on a rented RunPod Linux x86_64 GPU - setup, what to upload, what to bring back, cost recording and traps. Load when a task needs CUDA, spaCy, a large embedding pass or a rented machine.
---

# Remote GPU runs

## Why

The laptop is Windows on ARM64. That is a development constraint, not a scientific one: never
distort an experiment to fit it. Renting compute is the approved path.

- spaCy cannot install locally (`blis` has no `win_arm64` wheel).
- GLiNER installs but measured 0.558 paragraphs/s on the laptop CPU against 162.178 on an RTX 4090
  (~290x).
- Before adding any model or extractor dependency, check whether it installs here, whether an ONNX
  or other backend exists, and whether the work belongs on the rented machine.

## Environment facts

- `pyproject.toml` resolves `torch` from `pytorch-cu126` on `linux`/`x86_64` and from `pytorch-cpu`
  everywhere else (including win_arm64). Respect that source configuration.
- Python is pinned `>=3.12,<3.13`.
- GLiNER moves itself to CUDA when a device exists and records the GPU name in its manifest.
- Optional extraction dependencies live in the `phase7` group: `uv sync --group phase7`.

## Recipe that has worked (Phase 7, deviation 8.1)

1. Freeze and push the exact commit the pod will run; note its hash.
2. Rent 1x RTX 4090, Secure Cloud, PyTorch template, container disk sized for the job, no
   persistent volume. Record the contracted hourly rate before starting.
3. On the pod: clone the branch into `/workspace`, check out the frozen commit, `uv sync` with the
   groups the job needs.
4. Upload only the frozen inputs the repo does not contain (Phase 7: just `data/pool.json`,
   ~14.5 MB, via Jupyter drag-and-drop).
5. Verify CUDA with a real kernel, not only `torch.cuda.is_available()`.
6. Probe a small sample (for example 500 paragraphs) before committing to a full pass.
7. Run the measured stage. Stages that record cost demand `--actual-cost-usd` whenever
   `--hourly-rate-usd` is nonzero: know the pass duration first or round the charge up.
8. `sha256sum` every expensive artifact on both ends and download it before terminating.
   `/workspace` dies with the pod.
9. Classify or decide on the laptop, not on the pod.

Phase-specific, step-by-step versions with their stop conditions:
`docs/plans/phase_8/8.runpod_recipe.md` and `docs/plans/phase_8/8.1_runpod_recipe.md`. A new phase
that rents compute writes its own recipe before the pod is opened.

## Traps

- The manifest `hardware` block reports the host (for example 256 cores, 1.08 TB RAM), not the
  container allocation (8 vCPU, 31 GB). Record the allocation separately.
- `question_cache_key` does not include the corpus. Any run that changes the corpus must encode
  into its own cache directory or it overwrites the question cache behind a published result.
- `TokenCounter.count_units` tokenizes its whole input in one call; that does not survive
  hundreds of thousands of paragraphs. Batch it.
- Costs: Phase 7 invoiced 0.34 EUR for the whole session. Record the measured cost; never quote a
  projection as a measurement.
