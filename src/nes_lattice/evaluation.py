from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .hamiltonians import (
    HamiltonianSpec,
    apply_hamiltonian_to_state_values,
    dense_hamiltonian_np,
    basis_shape_for_hamiltonian,
)
from .lattice import make_basis
from .sampler import make_config_sampler, init_configs


def exact_span_matrices(apply_fun, params, hspec: HamiltonianSpec, bonds: jnp.ndarray):
    basis_np = make_basis(basis_shape_for_hamiltonian(hspec), hspec.magnetization)
    basis = jnp.asarray(basis_np)

    psi = apply_fun(params, basis)  # (dim, k)
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

    sampler = make_config_sampler(
        apply_fun,
        hspec.shape,
        hspec.move_type,
        n_chains,
        n_samples,
        sweep_steps,
        burn_in,
        n_sites=hspec.N,
    )

    k1, k2 = jax.random.split(key)
    configs0 = init_configs(k1, n_chains, hspec.shape, hspec.move_type, n_sites=hspec.N)
    samples, _, stats = sampler(params, k2, configs0)

    psi = apply_fun(params, samples)
    Hpsi = apply_hamiltonian_to_state_values(apply_fun, params, samples, hspec, bonds)

    # Sample from q(sigma) proportional to sum_i |psi_i(sigma)|^2 and estimate
    # S_ij = <psi_i|psi_j>, H_ij = <psi_i|H|psi_j> up to a common factor.
    # The common normalization cancels in the generalized eigenproblem.
    q = jnp.sum(psi * psi, axis=1) + 1e-12
    S = jnp.einsum("ma,mb,m->ab", psi, psi, 1.0 / q) / samples.shape[0]
    H = jnp.einsum("ma,mb,m->ab", psi, Hpsi, 1.0 / q) / samples.shape[0]

    return (
        np.asarray(S, dtype=np.float64),
        np.asarray(H, dtype=np.float64),
        {k: float(v) for k, v in stats.items()},
    )


def span_energies_from_matrices(
    S: np.ndarray,
    H: np.ndarray,
    jitter: float = 1e-6,
    rcond: float = 1e-10,
    return_info: bool = False,
):
    """Solve H c = E S c robustly for noisy sampled span matrices.

    In exact arithmetic S is positive semi-definite. With sampled evaluation,
    finite precision, or nearly collapsed NES states, S can be singular or have
    tiny negative eigenvalues. scipy.linalg.eigh(H, S) uses a Cholesky
    factorization of S and crashes unless S is strictly positive definite.

    We therefore use canonical orthogonalization:
        S = U diag(s) U^T,
        S^{-1/2} H S^{-1/2} y = E y,
    with eigenvalues of S floored to a small positive value. This keeps the
    logging/evaluation stable and records the effective rank/condition number.
    """
    S = np.asarray(S, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    S = 0.5 * (S + S.T)
    H = 0.5 * (H + H.T)

    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(H)):
        k = S.shape[0]
        vals = np.full((k,), np.nan, dtype=np.float64)
        info = {
            "S_min_eig": np.nan,
            "S_max_eig": np.nan,
            "S_rank": 0,
            "S_floor": float(jitter),
            "S_num_clipped": k,
            "S_eigenvalues": [],
        }
        if return_info:
            return vals, np.inf, info
        return vals, np.inf

    s, U = np.linalg.eigh(S)
    s = np.asarray(s, dtype=np.float64)

    smax = float(max(np.max(s), jitter, 1e-30))
    sfloor = float(max(jitter, rcond * smax))
    rank = int(np.sum(s > sfloor))
    num_clipped = int(np.sum(s <= sfloor))

    s_safe = np.maximum(s, sfloor)

    # Build S^{-1/2} without Cholesky, so semi-definite/noisy S cannot crash.
    Sinvhalf = (U * (1.0 / np.sqrt(s_safe))) @ U.T
    Heff = Sinvhalf @ H @ Sinvhalf
    Heff = 0.5 * (Heff + Heff.T)

    vals = np.linalg.eigvalsh(Heff)
    vals.sort()

    cond = float(np.max(s_safe) / np.min(s_safe))
    info = {
        "S_min_eig": float(np.min(s)),
        "S_max_eig": float(np.max(s)),
        "S_rank": rank,
        "S_floor": sfloor,
        "S_num_clipped": num_clipped,
        "S_eigenvalues": [float(x) for x in s],
    }

    if return_info:
        return vals, cond, info
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
    jitter: float = 1e-6,
):
    if hspec.N <= exact_if_sites_leq:
        S, H = exact_span_matrices(apply_fun, params, hspec, bonds)
        energies, cond, info = span_energies_from_matrices(S, H, jitter, return_info=True)
        info["method"] = "exact_span"
        info["accept_rate"] = None
        return energies, cond, info

    S, H, stats = sampled_span_matrices(
        apply_fun,
        params,
        hspec,
        bonds,
        key,
        n_chains=eval_chains,
        n_samples=eval_samples,
    )
    energies, cond, info = span_energies_from_matrices(S, H, jitter, return_info=True)

    stats.update(info)
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
