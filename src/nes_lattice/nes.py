from __future__ import annotations

import jax
import jax.numpy as jnp

from .hamiltonians import HamiltonianSpec, diag_energy


def amplitude_matrix(apply_fun, params, bundle: jnp.ndarray, det_jitter: float = 1e-8) -> jnp.ndarray:
    """A[i, j] = psi_i(sigma_j), shape (k states, k replicas)."""
    vals = apply_fun(params, bundle)  # (k configs, k states)
    A = vals.T
    if det_jitter and det_jitter > 0:
        A = A + det_jitter * jnp.eye(A.shape[0], dtype=A.dtype)
    return A


def signed_logdet_bundle(apply_fun, params, bundle: jnp.ndarray, det_jitter: float = 1e-8):
    A = amplitude_matrix(apply_fun, params, bundle, det_jitter)
    return jnp.linalg.slogdet(A)


def logabsdet_bundle(apply_fun, params, bundle: jnp.ndarray, det_jitter: float = 1e-8):
    return signed_logdet_bundle(apply_fun, params, bundle, det_jitter)[1]


def batch_logabsdet(apply_fun, params, bundles: jnp.ndarray, det_jitter: float = 1e-8):
    return jax.vmap(lambda b: logabsdet_bundle(apply_fun, params, b, det_jitter))(bundles)


def local_energy_bundle(apply_fun, params, bundle: jnp.ndarray, hspec: HamiltonianSpec, bonds: jnp.ndarray, det_jitter: float = 1e-8):
    """NES local energy of the determinant wavefunction for one bundle.

    bundle shape is (k replicas, N sites). H_total = sum_replica H_replica.
    Off-diagonal ratios use the column-replacement determinant identity.
    """
    A = amplitude_matrix(apply_fun, params, bundle, det_jitter)
    Ainv = jnp.linalg.inv(A)
    k, N = bundle.shape

    e = jnp.sum(diag_energy(bundle, hspec, bonds))

    if hspec.name == "tfim":
        for rep in range(k):
            for site in range(N):
                new_config = bundle[rep].at[site].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0]  # state vector psi_i(new)
                ratio = (Ainv @ v)[rep]
                e = e + (-hspec.g) * ratio
    elif hspec.name == "heisenberg":
        for rep in range(k):
            s = bundle[rep]
            for b in range(bonds.shape[0]):
                i = bonds[b, 0]
                j = bonds[b, 1]
                active = s[i] != s[j]
                new_config = s.at[i].multiply(-1).at[j].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0]
                ratio = (Ainv @ v)[rep]
                e = e + jnp.where(active, 0.5 * hspec.J * ratio, 0.0)
    else:
        raise ValueError(hspec.name)
    return e


def batch_local_energy(apply_fun, params, bundles: jnp.ndarray, hspec: HamiltonianSpec, bonds: jnp.ndarray, det_jitter: float = 1e-8):
    return jax.vmap(lambda b: local_energy_bundle(apply_fun, params, b, hspec, bonds, det_jitter))(bundles)


def vmc_surrogate_loss(apply_fun, params, bundles: jnp.ndarray, hspec: HamiltonianSpec, bonds: jnp.ndarray, det_jitter: float = 1e-8):
    """Score-function VMC surrogate for stochastic NES optimization.

    Samples are treated as fixed. The gradient of this scalar is
        2 < (E_L - <E_L>) grad log|det A| >,
    which is the standard real-valued VMC energy gradient estimator.
    """
    e_loc = batch_local_energy(apply_fun, params, bundles, hspec, bonds, det_jitter)
    e_mean = jnp.mean(e_loc)
    logabs = batch_logabsdet(apply_fun, params, bundles, det_jitter)
    centered = jax.lax.stop_gradient(e_loc - e_mean)
    loss = jnp.mean(2.0 * centered * logabs)
    return loss, e_mean
