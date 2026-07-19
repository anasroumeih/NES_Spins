from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .nes import batch_local_energy_matrix


def nes_energy_matrix_stats(local_energy_mats, *, compute_variance: bool = True) -> dict:
    """Estimate NES energies, and optionally S12 variances, from local energy matrices.

    For sampled determinant bundles X, the NES local energy matrix is
        E_L(X) = B(X) A(X)^{-1}.
    We estimate the physical k-state energy matrix by averaging E_L(X) over
    |det A|^2 samples, then diagonalize this mean k x k matrix.

    If compute_variance=True, also implement the practical Sec. S12 estimator:
    transform every sampled local energy matrix to the eigenbasis of the mean
    matrix and take the sample variance of the diagonal elements.

    Args:
      local_energy_mats: array with shape (n_samples, k, k).
      compute_variance: whether to compute the S12 per-state variance estimate.

    Returns:
      JSON-serializable statistics.
    """
    mats = np.asarray(local_energy_mats, dtype=np.float64)
    if mats.ndim != 3:
        raise ValueError(f"local_energy_mats must have shape (n,k,k), got {mats.shape}")
    n, k, k2 = mats.shape
    if k != k2:
        raise ValueError(f"local_energy_mats must be square in last axes, got {mats.shape}")

    mean_mat = np.mean(mats, axis=0)
    eigvals, eigvecs = np.linalg.eig(mean_mat)
    order = np.argsort(eigvals.real)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    out = {
        "energy_matrix_mean": mean_mat.real.tolist(),
        "energy_matrix_eigvals": eigvals.real.tolist(),
        "energy_matrix_eigvals_imag": eigvals.imag.tolist(),
        "n_energy_matrix_samples": int(n),
        "energy_matrix_eigvec_cond": float(np.linalg.cond(eigvecs)),
    }

    if compute_variance:
        eigvecs_inv = np.linalg.inv(eigvecs)
        demixed = np.einsum("ab,nbc,cd->nad", eigvecs_inv, mats, eigvecs)
        diag_samples = np.diagonal(demixed, axis1=1, axis2=2)

        if n > 1:
            state_variances = np.var(diag_samples.real, axis=0, ddof=1)
        else:
            state_variances = np.zeros((k,), dtype=np.float64)
        state_std_errors = np.sqrt(np.maximum(state_variances, 0.0) / max(n, 1))

        out.update(
            {
                "state_energy_variances": state_variances.real.tolist(),
                "state_energy_std_errors": state_std_errors.real.tolist(),
                "n_variance_samples": int(n),
            }
        )

    return out


def energy_matrix_from_bundles(
    apply_fun,
    params,
    bundles,
    hspec,
    bonds,
    *,
    compute_variance: bool = False,
) -> dict:
    """Compute sampled NES energy-matrix stats on determinant bundles."""
    mats = batch_local_energy_matrix(apply_fun, params, bundles, hspec, bonds)
    return nes_energy_matrix_stats(
        np.asarray(jax.device_get(mats)),
        compute_variance=compute_variance,
    )


def variance_from_bundles(apply_fun, params, bundles, hspec, bonds) -> dict:
    """Backward-compatible S12 variance wrapper."""
    return energy_matrix_from_bundles(
        apply_fun,
        params,
        bundles,
        hspec,
        bonds,
        compute_variance=True,
    )


# Backward-compatible old name.
def nes_energy_matrix_variance(local_energy_mats) -> dict:
    return nes_energy_matrix_stats(local_energy_mats, compute_variance=True)


@jax.jit
def scalar_local_energy_variance(apply_fun, params, bundles, hspec, bonds):
    """Cheap scalar variance of Tr(E_L), mostly useful as a sanity check."""
    mats = batch_local_energy_matrix(apply_fun, params, bundles, hspec, bonds)
    traces = jnp.trace(mats, axis1=-2, axis2=-1)
    return jnp.var(traces), jnp.mean(traces)
