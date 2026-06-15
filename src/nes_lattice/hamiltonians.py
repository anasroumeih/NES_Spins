from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .lattice import nearest_neighbor_bonds, normalize_shape, num_sites

HamType = Literal["tfim", "heisenberg"]


@dataclass(frozen=True)
class HamiltonianSpec:
    name: HamType = "tfim"
    shape: tuple[int, ...] = (4, 4)
    J: float = 1.0
    g: float = 1.0       # TFIM transverse field
    pbc: bool = True
    magnetization: int | None = None

    @property
    def N(self) -> int:
        return num_sites(self.shape)

    @property
    def bonds_np(self) -> np.ndarray:
        return nearest_neighbor_bonds(self.shape, self.pbc)

    @property
    def move_type(self) -> str:
        # TFIM changes magnetization; Heisenberg conserves it, so use pair flips in Sz=0.
        if self.magnetization is not None:
            return "pair_flip"
        return "single_flip"


def make_hamiltonian_spec(
    name: str = "tfim",
    shape: int | tuple[int, ...] = (4, 4),
    J: float = 1.0,
    g: float = 1.0,
    pbc: bool = True,
    magnetization: int | None = None,
) -> HamiltonianSpec:
    shape = normalize_shape(shape)
    name = name.lower()
    if name not in ("tfim", "heisenberg"):
        raise ValueError("name must be 'tfim' or 'heisenberg'")
    if name == "heisenberg" and magnetization is None:
        # For spin-1/2 AF Heisenberg, the low-energy sector is usually Sz=0.
        # Keeping this default also makes the Metropolis moves ergodic inside the target sector.
        magnetization = 0 if num_sites(shape) % 2 == 0 else None
    return HamiltonianSpec(name=name, shape=shape, J=J, g=g, pbc=pbc, magnetization=magnetization)


def diag_energy(spins: jnp.ndarray, hspec: HamiltonianSpec, bonds: jnp.ndarray) -> jnp.ndarray:
    """Diagonal energy for one or many configs. spins shape (..., N)."""
    s_i = spins[..., bonds[:, 0]]
    s_j = spins[..., bonds[:, 1]]
    if hspec.name == "tfim":
        return -hspec.J * jnp.sum(s_i * s_j, axis=-1)
    if hspec.name == "heisenberg":
        # s=±1 encodes 2 S^z, hence S^z_i S^z_j = s_i s_j / 4.
        return 0.25 * hspec.J * jnp.sum(s_i * s_j, axis=-1)
    raise ValueError(hspec.name)


def apply_hamiltonian_to_state_values(apply_fun, params, configs: jnp.ndarray, hspec: HamiltonianSpec, bonds: jnp.ndarray):
    """Compute (H psi_b)(sigma) for all b at configs.

    Returns an array with shape (n_configs, k). This is used for sampled/exact
    span matrices H_ab = sum_sigma psi_a(sigma) (H psi_b)(sigma).
    """
    psi = apply_fun(params, configs)  # (M, k)
    out = diag_energy(configs, hspec, bonds)[:, None] * psi
    M, N = configs.shape
    if hspec.name == "tfim":
        for site in range(N):
            flipped = configs.at[:, site].multiply(-1)
            out = out + (-hspec.g) * apply_fun(params, flipped)
    elif hspec.name == "heisenberg":
        for b in range(bonds.shape[0]):
            i, j = int(bonds[b, 0]), int(bonds[b, 1])
            opposite = configs[:, i] != configs[:, j]
            flipped = configs.at[:, i].multiply(-1).at[:, j].multiply(-1)
            out = out + (0.5 * hspec.J) * apply_fun(params, flipped) * opposite[:, None]
    else:
        raise ValueError(hspec.name)
    return out


def dense_hamiltonian_np(hspec: HamiltonianSpec, basis: np.ndarray) -> np.ndarray:
    """Dense Hamiltonian in a provided basis. Intended only for small ED checks."""
    bonds = hspec.bonds_np
    index = {tuple(row.tolist()): a for a, row in enumerate(basis)}
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.float64)
    for a, s in enumerate(basis):
        H[a, a] += float(np.asarray(diag_energy(jnp.asarray(s[None, :]), hspec, jnp.asarray(bonds)))[0])
        if hspec.name == "tfim":
            for site in range(hspec.N):
                sp = s.copy()
                sp[site] *= -1
                b = index.get(tuple(sp.tolist()))
                if b is not None:
                    H[b, a] += -hspec.g
        elif hspec.name == "heisenberg":
            for i, j in bonds:
                if s[i] != s[j]:
                    sp = s.copy()
                    sp[i] *= -1
                    sp[j] *= -1
                    b = index.get(tuple(sp.tolist()))
                    if b is not None:
                        H[b, a] += 0.5 * hspec.J
    return H
