# Upstream and Dependencies

Fisher Damping is a small downstream optimizer scaffold. It does not fork or
vendor an upstream benchmark.

## Current Substrate

- PyTorch provides the model, tensor, autograd, and optimizer baselines.
- NumPy and SciPy provide exact matrix and spectral calculations.
- `curvlinops-for-pytorch` is used for Fisher linear-operator construction in
  the adaptive optimizer path.

## Benchmark Boundary

The current repository uses controlled synthetic objectives only. OGBench or
other agent benchmarks should be added only after the damping rule survives a
declared exact-objective experiment with raw trajectories, matched update-norm
controls, and hyperparameter-search manifests.
