# Stochastic NES for discrete lattice spin systems

This is the upgraded project: **Metropolis-sampled Natural Excited States** for discrete spin systems.

It supports:

- 2D lattices by default, with 1D as the exception: `shape=(4,4)` or `shape=(10,)`
- Hamiltonians: `tfim`, `heisenberg`
- ansätze: `ffn`, `rbm`, `cnn`
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
- no symmetry-sector targeting except the default Heisenberg `Sz=0` sector

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
