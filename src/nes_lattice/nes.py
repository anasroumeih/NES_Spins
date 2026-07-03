from __future__ import annotations

import jax
import jax.numpy as jnp

from .hamiltonians import HamiltonianSpec, diag_energy


def amplitude_matrix(apply_fun, params, bundle: jnp.ndarray) -> jnp.ndarray:
    """Physical NES matrix A[i, j] = psi_i(sigma_j), with no determinant jitter."""
    return apply_fun(params, bundle).T


def signed_logdet_bundle(apply_fun, params, bundle: jnp.ndarray):
    return jnp.linalg.slogdet(amplitude_matrix(apply_fun, params, bundle))


def logabsdet_bundle(apply_fun, params, bundle: jnp.ndarray):
    return signed_logdet_bundle(apply_fun, params, bundle)[1]


def batch_logabsdet(apply_fun, params, bundles: jnp.ndarray):
    return jax.vmap(lambda b: logabsdet_bundle(apply_fun, params, b))(bundles)


def _local_energy_bundle_valid(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """NES local energy for a finite nonzero determinant bundle."""
    A = amplitude_matrix(apply_fun, params, bundle)
    k, N = bundle.shape
    Ainv = jnp.linalg.solve(A, jnp.eye(k, dtype=A.dtype))
    e = jnp.sum(diag_energy(bundle, hspec, bonds))

    if hspec.name == "tfim":
        for rep in range(k):
            for site in range(N):
                new_config = bundle[rep].at[site].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0]
                e = e - hspec.g * (Ainv @ v)[rep]
    elif hspec.name == "heisenberg":
        for rep in range(k):
            s = bundle[rep]
            for b in range(bonds.shape[0]):
                i, j = bonds[b, 0], bonds[b, 1]
                active = s[i] != s[j]
                new_config = s.at[i].multiply(-1).at[j].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0]
                e = e + jnp.where(active, 0.5 * hspec.J * (Ainv @ v)[rep], 0.0)
    elif hspec.name == "toric_code":
        stars = bonds[0]
        for rep in range(k):
            s = bundle[rep]
            for a in range(stars.shape[0]):
                new_config = s.at[stars[a]].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0]
                e = e - hspec.Je * (Ainv @ v)[rep]
    else:
        raise ValueError(hspec.name)
    return e


def local_energy_bundle(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """Return NaN only for an impossible (zero-probability) singular bundle."""
    _, logabs = signed_logdet_bundle(apply_fun, params, bundle)
    valid = jnp.isfinite(logabs)
    dtype = amplitude_matrix(apply_fun, params, bundle).dtype
    return jax.lax.cond(
        valid,
        lambda _: _local_energy_bundle_valid(apply_fun, params, bundle, hspec, bonds),
        lambda _: jnp.asarray(jnp.nan, dtype=dtype),
        operand=None,
    )


def batch_local_energy(
    apply_fun,
    params,
    bundles: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    return jax.vmap(
        lambda b: local_energy_bundle(apply_fun, params, b, hspec, bonds)
    )(bundles)


def vmc_surrogate_loss(
    apply_fun,
    params,
    bundles: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """Penalty-free real VMC gradient surrogate for the exact NES determinant."""
    e_loc = batch_local_energy(apply_fun, params, bundles, hspec, bonds)
    e_mean = jnp.mean(e_loc)
    logabs = batch_logabsdet(apply_fun, params, bundles)
    centered = jax.lax.stop_gradient(e_loc - e_mean)
    loss = jnp.mean(2.0 * centered * logabs)
    return loss, e_mean
