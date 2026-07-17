# Experiment Ledger

This ledger describes the public evidence boundary for Fisher Damping.

## Current Verdict

The repository has a working implementation scaffold for spectral-statistic
adaptive damping. It does not yet contain a checked-in canonical result with raw
trajectories and generated summary tables.

## Implemented Components

| Component | Status | Evidence boundary |
|---|---|---|
| Fisher spectral statistics | Implemented | Unit tests cover uniform, peaked, sparse, and zero spectra. |
| Adaptive damping rule | Implemented | Unit tests verify low damping for rich spectra and high damping for rank-deficient spectra. |
| Regime detector | Implemented | Unit tests cover rich, sparse, blind, and external-shift states. |
| Adaptive natural gradient | Implemented | Regression tests exercise controlled rich/blind objectives. |
| LR and step-norm controls | Implemented as experiment functions | Not promoted as a public result until generated artifacts are checked in. |

## Removed Scope

The old broad systems framing was removed from the public tree. It is not part
of the Fisher Damping evidence claim.

## Next Valid Result

A promoted result should include:

1. A declared config for controlled rich-to-blind and rich-to-sparse shifts.
2. Adam, fixed natural-gradient, hard-switching, effective-rank damping, and
   oracle damping baselines under matched tuning budgets.
3. Matched update-norm summaries.
4. Raw loss and damping trajectories.
5. Deterministic summaries regenerated from raw outputs.
6. Sensitivity across seeds and learning-rate/damping grids.

Until those artifacts exist, the public claim is limited to implementation and
test coverage for the controlled optimizer scaffold.
