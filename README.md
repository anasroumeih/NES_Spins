# Stochastic NES for discrete lattice spin systems

This is the upgraded project: **Metropolis-sampled Natural Excited States** for discrete spin systems.

It supports:

- 2D lattices by default, with 1D as the exception: `shape=(4,4)` or `shape=(10,)`
- Hamiltonians: `tfim`, `heisenberg`
- ansätze: `ffn`, `resffn`, `toric_resffn`, `rbm`, `toric_rbm`, `cnn`, `vit`
- stochastic NES bundle sampling from `|det A|^2`
- Adam optimization, no SR
- optional NetKet references, plus own dense ED for small systems
- JSON experiment saving and plotting notebooks
- logging of individual energies, trace error, condition number of overlap matrix, sampler acceptance and gradient norm

## Install

From the project root:

```bash
source ~/myenv/bin/activate
pip install -e .
```

If CUDA JAX gives problems, force CPU before importing JAX:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
```

Optional NetKet reference:

```bash
pip install netket
```

The code will still run without NetKet. If NetKet is unavailable, it falls back to the small dense ED reference when possible.

## Main notebook workflow

1. Open `notebooks/01_test_references.ipynb`
2. Open `notebooks/02_train_sampled_nes.ipynb`
3. Open `notebooks/03_plot_results.ipynb`

The workflow is the same as before:

```text
train experiment → save JSON → plot/read JSON later
```

## Important config examples

### 1D TFIM, FFN

```python
cfg = TrainConfig(
    shape=(10,),
    hamiltonian="tfim",
    k=2,
    g=1.0,
    model="ffn",
    hidden=(128,128),
    steps=1000,
    lr=2e-3,
)
```

### 2D TFIM, CNN

```python
cfg = TrainConfig(
    shape=(4,4),
    hamiltonian="tfim",
    k=2,
    g=1.0,
    model="cnn",
    channels=(16,16),
    steps=1500,
    lr=1e-3,
)
```

### 2D Heisenberg, RBM in the Sz=0 sector

```python
cfg = TrainConfig(
    shape=(4,4),
    hamiltonian="heisenberg",
    k=2,
    J=1.0,
    model="rbm",
    rbm_hidden=64,
    steps=2000,
    lr=1e-3,
)
```

For Heisenberg, `magnetization=0` is used by default when the number of sites is even.

### Toric-code ground manifold, sector RBM

Use `toric_rbm` for the four topological ground states. It keeps the ordinary
`rbm` model unchanged, but multiplies each head by exact `B_p=+1` and
Wilson-sector projectors, so the NES determinant spans the four torus sectors
without an overlap penalty.

```python
cfg = TrainConfig(
    shape=(4, 4),
    hamiltonian="toric_code",
    k=4,
    model="toric_rbm",
    rbm_hidden=256,
    init_scale=0.02,
    steps=15000,
    lr=1e-4,
    grad_clip=1.0,
    n_chains=512,
    n_samples=64,
    sweep_steps=128,
    burn_in=1024,
    toric_loop_prob=0.0,
    toric_single_flip_prob=0.0,
    toric_cover_sectors=True,
)
```

### Toric-code ground manifold, sector ResFFN

Use `toric_resffn` for a residual-FFN core with the same exact flux and
Wilson-sector projectors. This keeps the residual ansatz trainable on the
fourfold toric-code ground manifold instead of drifting into the common
one-ground-plus-three-star-excitations plateau of an unconstrained ResFFN.

```python
cfg = TrainConfig(
    shape=(4, 4),
    hamiltonian="toric_code",
    k=4,
    model="toric_resffn",
    hidden=(128, 128),
    init_scale=0.005,
    steps=100,
    lr=1e-4,
    grad_clip=1.0,
    n_chains=64,
    n_samples=4,
    sweep_steps=32,
    burn_in=64,
    toric_loop_prob=0.0,
    toric_single_flip_prob=0.0,
    toric_cover_sectors=True,
)
```

### Toric-code exploratory residual FFN

Use `resffn` when you want a deeper random-initialized ansatz without hard
toric-code sector projectors. The entries in `hidden` define the residual
blocks, so `hidden=(512, 512, 512)` means three width-512 residual blocks.

```python
cfg = TrainConfig(
    shape=(4, 4),
    hamiltonian="toric_code",
    k=4,
    model="resffn",
    hidden=(512, 512, 512),
    init_scale=0.005,
    steps=20000,
    lr=2e-5,
    grad_clip=0.5,
    n_chains=1024,
    n_samples=64,
    sweep_steps=128,
    burn_in=1024,
    toric_loop_prob=0.10,
    toric_single_flip_prob=0.0,
    toric_cover_sectors=True,
)
```

## What is stochastic now?

The old toy version optimized exact span matrices by enumerating all `2^N` basis states. This version trains the NES determinant wavefunction using Metropolis bundles:

```text
X = (sigma_1, ..., sigma_k)
A_ij = psi_i(sigma_j)
Psi_NES(X) = det A
P(X) ∝ |det A|²
```

The local energy is computed from determinant ratios. For a move that changes replica `j`, the ratio is computed efficiently with the column-replacement identity:

```text
det(A with column j replaced by v) / det(A) = (A^{-1} v)_j
```

## Limits to expect

This is now a real stochastic prototype, but it is still a research toy, not production NetKet/FermiNet code.

Expected limitations:

- noisy optimization for larger `k`
- determinant ill-conditioning when states collapse or become too similar
- RBM outputs are positive per state, so harder excited states may need larger hidden size or FFN/CNN
- sampled span-matrix evaluation becomes noisy for large systems
- NetKet reference is optional and version-dependent
- symmetry-sector targeting is limited to Heisenberg `Sz=0` and toric-code
  `toric_resffn`/`toric_rbm` Wilson/flux projectors

Diagnostics to watch:

```text
condition_number_S
sampler_accept_rate
trace_error
abs_errors
train_energy_estimator
```

If it is unstable, first try:

```text
lower lr: 1e-3 or 5e-4
higher n_chains/n_samples
higher hidden/channels/rbm_hidden
smaller k
grad_clip=5.0
```
