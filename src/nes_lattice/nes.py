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


def local_energy_matrix_bundle(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """NES local energy matrix for one determinant bundle.

    Let A[i,j] = psi_i(sigma_j).  Let B[i,j] = (H psi_i)(sigma_j), where H acts
    on the j-th replica coordinate.  The bundle-local energy matrix is

        E_L(X) = B A^{-1}.

    Its trace is the scalar expanded-system local energy used by the current
    NES objective.  Keeping the full matrix lets us estimate individual-state
    energy variances following Sec. S12 of the NES paper.
    """
    A = amplitude_matrix(apply_fun, params, bundle)
    dtype = A.dtype
    k, N = bundle.shape
    Ainv = jnp.linalg.solve(A, jnp.eye(k, dtype=dtype))

    # Diagonal Hamiltonian contribution.  If d_j is the diagonal energy of
    # configuration sigma_j, then B[:, j] = d_j A[:, j].
    d = diag_energy(bundle, hspec, bonds).astype(dtype)
    B = A * d[None, :]

    if hspec.name == "tfim":
        g = jnp.asarray(hspec.g, dtype=dtype)
        for rep in range(k):
            for site in range(N):
                new_config = bundle[rep].at[site].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0].astype(dtype)
                B = B.at[:, rep].add(-g * v)

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
                B = B.at[:, rep].add(jnp.where(active, 0.5 * J, 0.0) * v)

    elif hspec.name == "toric_code":
        Je = jnp.asarray(hspec.Je, dtype=dtype)
        stars = bonds[0]
        for rep in range(k):
            s = bundle[rep]
            for a in range(stars.shape[0]):
                idx = stars[a]
                new_config = s.at[idx].multiply(-1)
                v = apply_fun(params, new_config[None, :])[0].astype(dtype)
                B = B.at[:, rep].add(-Je * v)
    else:
        raise ValueError(hspec.name)

    return B @ Ainv


def batch_local_energy_matrix(
    apply_fun,
    params,
    bundles: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    return jax.vmap(
        lambda b: local_energy_matrix_bundle(apply_fun, params, b, hspec, bonds)
    )(bundles)


def _local_energy_bundle_valid(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """Scalar NES local energy for a bundle with finite, nonzero determinant."""
    return jnp.trace(local_energy_matrix_bundle(apply_fun, params, bundle, hspec, bonds))


def local_energy_bundle(
    apply_fun,
    params,
    bundle: jnp.ndarray,
    hspec: HamiltonianSpec,
    bonds: jnp.ndarray,
):
    """NES scalar local energy for a valid determinant-sampled bundle.

    The sampler initializes valid bundles and rejects singular proposals, so
    every bundle reaching this function should have det(A) != 0.
    """
    return _local_energy_bundle_valid(apply_fun, params, bundle, hspec, bonds)


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
