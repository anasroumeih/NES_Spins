from __future__ import annotations

import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp

from .hamiltonians import HamiltonianSpec, apply_hamiltonian_to_state_values, dense_hamiltonian_np, basis_shape_for_hamiltonian
from .lattice import make_basis
from .sampler import make_config_sampler, init_configs


def exact_span_matrices(apply_fun, params, hspec: HamiltonianSpec, bonds: jnp.ndarray):
    basis_np = make_basis(basis_shape_for_hamiltonian(hspec), hspec.magnetization)
    basis = jnp.asarray(basis_np)
    psi = apply_fun(params, basis)  # (dim,k)
    Hpsi = apply_hamiltonian_to_state_values(apply_fun, params, basis, hspec, bonds)
    S = psi.T @ psi
    H = psi.T @ Hpsi
    return np.asarray(S, dtype=np.float64), np.asarray(H, dtype=np.float64)


def sampled_span_matrices(
    apply_fun,
    params,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
    key,
    n_chains: int = 128,
    n_samples: int = 32,
    sweep_steps: int | None = None,
    burn_in: int | None = None,
):
    N = hspec.N
    if sweep_steps is None:
        sweep_steps = max(1, N)
    if burn_in is None:
        burn_in = 10 * max(1, N)
    sampler = make_config_sampler(apply_fun, hspec.shape, hspec.move_type, n_chains, n_samples, sweep_steps, burn_in, n_sites=hspec.N)
    k1, k2 = jax.random.split(key)
    configs0 = init_configs(k1, n_chains, hspec.shape, hspec.move_type, n_sites=hspec.N)
    samples, _, stats = sampler(params, k2, configs0)
    psi = apply_fun(params, samples)
    Hpsi = apply_hamiltonian_to_state_values(apply_fun, params, samples, hspec, bonds)
    q = jnp.sum(psi * psi, axis=1) + 1e-12
    S = jnp.einsum("ma,mb,m->ab", psi, psi, 1.0 / q) / samples.shape[0]
    H = jnp.einsum("ma,mb,m->ab", psi, Hpsi, 1.0 / q) / samples.shape[0]
    return np.asarray(S, dtype=np.float64), np.asarray(H, dtype=np.float64), {k: float(v) for k, v in stats.items()}


def span_energies_from_matrices(S: np.ndarray, H: np.ndarray, jitter: float = 1e-8):
    S = 0.5 * (S + S.T)
    H = 0.5 * (H + H.T)
    # Small diagonal shift prevents crashes if the learned span is nearly singular.
    S_reg = S + jitter * np.eye(S.shape[0])
    vals = scipy.linalg.eigh(H, S_reg, eigvals_only=True)
    vals.sort()
    cond = float(np.linalg.cond(S_reg))
    return vals, cond


def evaluate_span(
    apply_fun,
    params,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
    key,
    exact_if_sites_leq: int = 16,
    eval_samples: int = 32,
    eval_chains: int = 128,
    jitter: float = 1e-8,
):
    if hspec.N <= exact_if_sites_leq:
        S, H = exact_span_matrices(apply_fun, params, hspec, bonds)
        energies, cond = span_energies_from_matrices(S, H, jitter)
        return energies, cond, {"method": "exact_span", "accept_rate": None}
    S, H, stats = sampled_span_matrices(
        apply_fun, params, hspec, bonds, key,
        n_chains=eval_chains, n_samples=eval_samples,
    )
    energies, cond = span_energies_from_matrices(S, H, jitter)
    stats["method"] = "sampled_span_q"
    return energies, cond, stats


def own_ed_reference(hspec: HamiltonianSpec, k: int, max_sites: int = 14):
    if hspec.N > max_sites:
        return None, f"own ED skipped because N={hspec.N} > max_sites={max_sites}"
    basis = make_basis(basis_shape_for_hamiltonian(hspec), hspec.magnetization)
    H = dense_hamiltonian_np(hspec, basis)
    vals = np.linalg.eigvalsh(H)
    vals.sort()
    return vals[:k], "own_dense_ed"
