from __future__ import annotations

import jax
import jax.numpy as jnp

from .hamiltonians import HamiltonianSpec, diag_energy


def amplitude_matrix(apply_fun, params, bundle: jnp.ndarray) -> jnp.ndarray:
    """A[i, j] = psi_i(sigma_j), shape (k states, k replicas).

    This is the physical NES amplitude matrix.
    Do not add a diagonal jitter here.
    """
    vals = apply_fun(params, bundle)  # (k configs, k states)
    return vals.T


def signed_logdet_bundle(apply_fun, params, bundle: jnp.ndarray):
    """Returns sign(det A), log(abs(det A))."""
    A = amplitude_matrix(apply_fun, params, bundle)
    return jnp.linalg.slogdet(A)


def logabsdet_bundle(apply_fun, params, bundle: jnp.ndarray):
    return signed_logdet_bundle(apply_fun, params, bundle)[1]


def batch_logabsdet(apply_fun, params, bundles: jnp.ndarray):
    return jax.vmap(
        lambda b: logabsdet_bundle(apply_fun, params, b)
    )(bundles)


def _local_energy_bundle_valid(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """NES local energy assuming det(A) is finite and nonzero."""
    A = amplitude_matrix(apply_fun, params, bundle)
    k, N = bundle.shape

    # Solve A X = I instead of calling inv(A).
    # This does not modify A; it is only numerically preferable.
    Ainv = jnp.linalg.solve(A, jnp.eye(k, dtype=A.dtype))

    e = jnp.sum(diag_energy(bundle, hspec, bonds))

    if hspec.name == "tfim":
        for rep in range(k):
            for site in range(N):
                new_config = bundle[rep].at[site].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0]
                ratio = (Ainv @ v)[rep]
                e = e - hspec.g * ratio

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

    elif hspec.name == "toric_code":
        stars = bonds[0]

        for rep in range(k):
            s = bundle[rep]

            for a in range(stars.shape[0]):
                idx = stars[a]
                new_config = s.at[idx].multiply(-1)

                v = apply_fun(params, new_config[None, :])[0]
                ratio = (Ainv @ v)[rep]

                e = e - hspec.Je * ratio

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
    """NES local energy for one bundle.

    Singular bundles have zero physical probability under |det A|².
    They should never survive Metropolis sampling; return NaN if one
    somehow reaches this function so the issue is visible.
    """
    _, logabs = signed_logdet_bundle(apply_fun, params, bundle)
    valid = jnp.isfinite(logabs)

    return jax.lax.cond(
        valid,
        lambda _: _local_energy_bundle_valid(
            apply_fun,
            params,
            bundle,
            hspec,
            bonds,
        ),
        lambda _: jnp.array(jnp.nan, dtype=jnp.float32),
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
        lambda b: local_energy_bundle(
            apply_fun,
            params,
            b,
            hspec,
            bonds,
        )
    )(bundles)


def vmc_surrogate_loss(
    apply_fun,
    params,
    bundles: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """Score-function VMC surrogate for the exact NES determinant."""
    e_loc = batch_local_energy(
        apply_fun,
        params,
        bundles,
        hspec,
        bonds,
    )

    e_mean = jnp.mean(e_loc)

    logabs = batch_logabsdet(
        apply_fun,
        params,
        bundles,
    )

    centered = jax.lax.stop_gradient(e_loc - e_mean)

    loss = jnp.mean(2.0 * centered * logabs)

    return loss, e_mean