# Limitations

Fisher Damping is currently scoped to controlled optimizer experiments.

## Current Scope

- Linear objectives with analytically inspectable Fisher/Hessian structure.
- Small PyTorch modules.
- Exact or small-sample Fisher estimates.
- Spectral-statistic damping and natural-gradient baselines.
- Deterministic unit and regression tests.

## Out of Scope

- Broad systems claims.
- Distributed sharing protocols.
- Compression-frontier claims.
- OGBench or reinforcement-learning benchmark claims.
- LongMemEval, LoCoMo, or other long-memory benchmark claims.
- Production optimizer claims.

## Evidence Boundary

Qualitative regression tests are not benchmark evidence. A public performance
claim requires checked-in raw outputs, declared configs, generated summaries,
matched learning-rate budgets, matched update-norm controls, and seed
sensitivity.
