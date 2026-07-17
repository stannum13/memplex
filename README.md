# Fisher Damping

Fisher Damping studies whether spectral statistics can continuously adapt
natural-gradient damping when curvature quality changes.

## Current Status

This repository is a controlled optimizer scaffold, not a broad systems result.
The remaining package tests Fisher-spectrum diagnostics, regime classification,
and adaptive Tikhonov damping on exact linear objectives.

| Area | Status | Claim boundary |
|---|---|---|
| Spectral statistics | Implemented and unit-tested | Computes entropy, rank, concentration, and rank deficiency from eigenvalues |
| Adaptive damping | Implemented and unit-tested | Maps spectral concentration/rank deficiency to Tikhonov damping |
| Optimizer | Implemented | Natural-gradient optimizer with adaptive damping for small PyTorch models |
| Controlled experiments | Implemented as importable functions | Qualitative regression checks only unless raw artifacts are generated and checked in |
| Legacy broad-systems scope | Removed | No systems-level claim is supported here |

The package name remains `memplex` for compatibility with the existing code.
The public research name for the current surface is Fisher Damping.

## Question

Can online Fisher-spectrum statistics choose damping under curvature-quality
shift better than fixed natural gradient, Adam, or hard regime switching when
learning-rate and update-norm confounds are controlled?

## Method

The current scaffold uses synthetic linear objectives where the Fisher/Hessian
structure is inspectable. It includes:

- Renyi entropy and effective-rank summaries of Fisher eigenvalues.
- A continuous damping rule based on spectral concentration and rank deficiency.
- A fixed natural-gradient baseline.
- Adam and matched-step-norm controls in small deterministic experiments.
- Tests that guard the expected rich/blind-regime behavior.

The next promoted result should write raw trajectories, hyperparameter-search
manifests, matched-update-norm tables, and deterministic summaries before any
headline performance claim is made.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Import the controlled experiments:

```python
from memplex.experiments.regimes import run_regime_experiment
from memplex.experiments.transition import run_transition_experiment
from memplex.experiments.lr_confound import run_lr_confound_experiment

regime = run_regime_experiment()
transition = run_transition_experiment()
lr_control = run_lr_confound_experiment()
```

## Repository Map

- `memplex/fisher/spectrum.py`: entropy, effective rank, concentration, and
  adaptive damping.
- `memplex/fisher/regime.py`: discrete Fisher regime classifier.
- `memplex/optimizer.py`: adaptive natural-gradient optimizer.
- `memplex/experiments/`: deterministic controlled experiment functions.
- `tests/`: unit and regression tests for statistics, regimes, and experiments.
- `docs/experiment-ledger.md`: current evidence boundary.
- `docs/limitations.md`: scope and unsupported claims.
- `UPSTREAM.md`: dependency and upstream-substrate notes.

## Evidence Boundary

The checked-in tests verify implementation behavior on deterministic synthetic
objectives. They do not constitute a benchmark result or an OGBench result.
Public numbers should be added only after a checked-in script generates raw
outputs and a deterministic summary from a declared config.

## License

MIT
