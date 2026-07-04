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
    """NES local energy for a bundle with finite, nonzero determinant."""
    A = amplitude_matrix(apply_fun, params, bundle)
    dtype = A.dtype
    k, N = bundle.shape

    Ainv = jnp.linalg.solve(A, jnp.eye(k, dtype=dtype))

    # Keep the full local-energy calculation in the ansatz dtype.
    e = jnp.asarray(jnp.sum(diag_energy(bundle, hspec, bonds)), dtype=dtype)

    if hspec.name == "tfim":
        g = jnp.asarray(hspec.g, dtype=dtype)

        for rep in range(k):
            for site in range(N):
                new_config = bundle[rep].at[site].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0].astype(dtype)
                ratio = (Ainv @ v)[rep]
                e = e - g * ratio

    elif hspec.name == "heisenberg":
        J = jnp.asarray(hspec.J, dtype=dtype)

        for rep in range(k):
            s = bundle[rep]

            for b in range(bonds.shape[0]):
                i = bonds[b, 0]
                j = bonds[b, 1]

                active = s[i] != s[j]
                new_config = s.at[i].multiply(-1).at[j].multiply(-1)

                v = apply_fun(params, new_config[None, :])[0].astype(dtype)
                ratio = (Ainv @ v)[rep]

                e = e + jnp.where(active, 0.5 * J * ratio, 0.0)

    elif hspec.name == "toric_code":
        Je = jnp.asarray(hspec.Je, dtype=dtype)
        stars = bonds[0]

        for rep in range(k):
            s = bundle[rep]

            for a in range(stars.shape[0]):
                idx = stars[a]
                new_config = s.at[idx].multiply(-1)

                v = apply_fun(params, new_config[None, :])[0].astype(dtype)
                ratio = (Ainv @ v)[rep]

                e = e - Je * ratio

    else:
        raise ValueError(hspec.name)

    return jnp.asarray(e, dtype=A.dtype)


def local_energy_bundle(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """NES local energy for a valid determinant-sampled bundle.

    The sampler initializes valid bundles and rejects singular proposals,
    so every bundle reaching this function should have det(A) != 0.
    """
    return _local_energy_bundle_valid(
        apply_fun,
        params,
        bundle,
        hspec,
        bonds,
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
