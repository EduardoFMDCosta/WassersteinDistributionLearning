# Codebase notes for LLMs

## What this project does
Learns a data-driven **Wasserstein ambiguity ball** around a discrete centre distribution from samples. Given pretraining samples and evaluation samples it returns an `AmbiguitySet(center, radius)` and an optional `ProbabilityInterval` for the complement mass.

## Package layout
```
src/wasserstein_distribution_learning/
  api.py           — public API (EmpiricalPartition, AmbiguitySetLearner)
  sets.py          — HyperRectangle, Partition ABC, BoundedVoronoiPartition, HyperRectanglePartition
  quantization.py  — Quantization, FullLearningQuantization, ConditionalLearningQuantization
  confidence.py    — ClopperPearsonConfidence
  bound.py         — DataDrivenRadius, fournier_radius
  utils.py         — _detect_modes (GMM BIC selection)
  solvers/         — solver implementations (triangle_inequality_vertex is the default)
experiments/
  configs/
    handlers.py    — parse_arguments / process_args (argparse + parameters.json)
    construct.py   — distribution constructors, get_support_assumption, get_distribution
    parameters.json — per-distribution hyperparameter settings
  examples/
    learn.py       — end-to-end example (EmpiricalPartition → AmbiguitySetLearner)
    partition.py   — visualises Voronoi vs HyperRectangle partition side-by-side (2D only)
```

## API
Two-step public API (`from wasserstein_distribution_learning import ...`):

```python
partition = EmpiricalPartition(
    pretraining_samples,   # (N_pre, d) tensor
    num_clusters=100,
    support=None,          # None or torch.stack([lower, upper]) shape (2, d)
    partition_type='voronoi',  # or 'hyperrectangle'
)

learner = AmbiguitySetLearner(
    partition,             # EmpiricalPartition or raw Partition
    samples,               # (N, d) tensor
    beta=1e-6,
    learning_type='full',  # 'full' / 'full_learning' or 'conditional' / 'conditional_learning'
    method='triangle_inequality_vertex',
    wasserstein_order=2,
)
# learner.ambiguity_set   → AmbiguitySet(center: Quantization, radius: Tensor)
# learner.complement_interval → ProbabilityInterval(lower, upper) or None
# learner.fournier_radius → float (inf when support is None)
```

`get_support_assumption` in `construct.py` returns `torch.stack([-half, half])` (shape `(2, d)`) or `None`.

## Partition types
- **BoundedVoronoiPartition**: K-means (torch-kmeans on CPU, FAISS on GPU) → Voronoi radii clamped by max-sample-distance. `l1_radii = sqrt(2) * l2_radii`.
- **HyperRectanglePartition**: GMM-seeded BSP. Steps: (1) BIC selects #modes, (2) tight bbox per mode, (3) greedy median-split on max-variance dim until M boxes. Regions are disjoint by construction. Centroids = mean of contained samples.

## Quantization modes
- **FullLearningQuantization**: M+1 regions (M bounded + complement), Bonferroni over all. Solver sees M+1 atoms. Use for bounded support.
- **ConditionalLearningQuantization**: Solver sees M atoms only. Complement tracked via separate CP interval. Use for unbounded support (Gaussian etc.).
- `UncertainQuantization` is an alias for `FullLearningQuantization`.

## Solvers
`get_solver(method=...)` returns a `Solver`. Continuous-space methods live in `solvers/`; discrete solvers are wrapped by `IndependentSolver`. Key method: `triangle_inequality_vertex` (default, fast). MILP variants require gurobipy (pinned to 12.0.3).

## fournier_radius
Minimax Fournier–Guilllin bound. Constants from Table 1/2 of Fournier (2023) for L∞ ball support. Adjusted for L2 norm by `* sqrt(d)`. Returns `inf` when `support is None`. Computed using evaluation sample count only (not pretraining).

## DataDrivenRadius
Wraps solver call. Catches NaN/Inf/infeasible exceptions and returns `Result(bound=inf, ...)` instead of crashing — needed for unbounded support + full_learning.

## Key invariants
- The complement region is always the **last** element of all geometry tensors (`locs`, `l2_radii`, distance matrices).
- `len(partition)` = M+1; `len(ConditionalLearningQuantization)` = M.
- `support` inside `Partition` is a `HyperRectangle` object; the public API accepts `torch.stack([lower, upper])` and converts internally.

## Running examples (from project root)
```
py experiments/examples/learn.py
py experiments/examples/partition.py
```
Examples add `experiments/` to `sys.path` at startup so `configs/` and `plotting/` are importable.
