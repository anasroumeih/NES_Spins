from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .lattice import (
    make_basis,
    nearest_neighbor_bonds,
    normalize_shape,
    num_sites,
    toric_code_num_edges,
    toric_code_terms,
)

HamType = Literal["tfim", "heisenberg", "toric_code"]


@dataclass(frozen=True)
class HamiltonianSpec:
    name: HamType = "tfim"
    shape: tuple[int, ...] = (4, 4)
    J: float = 1.0
    g: float = 1.0       # TFIM transverse field
    pbc: bool = True
    magnetization: int | None = None
    Je: float = 1.0      # toric-code star coupling: -Je sum_s A_s
    Jm: float = 1.0      # toric-code plaquette coupling: -Jm sum_p B_p

    @property
    def N(self) -> int:
        if self.name == "toric_code":
            return toric_code_num_edges(self.shape)
        return num_sites(self.shape)

    @property
    def n_cells(self) -> int:
        return num_sites(self.shape)

    @property
    def bonds_np(self) -> np.ndarray:
        if self.name == "toric_code":
            stars, plaquettes = toric_code_terms(self.shape, self.pbc)
            return np.stack([stars, plaquettes], axis=0)
        return nearest_neighbor_bonds(self.shape, self.pbc)

    @property
    def move_type(self) -> str:
        # TFIM and toric code do not conserve magnetization.
        # Heisenberg conserves Sz, so use pair flips in the Sz=0 sector.
        if self.name == "toric_code":
            # Star and winding-loop proposals preserve the plaquette-flux
            # sector.  This is the intended sampler for the topological
            # ground-space calculation; an optional rare single-edge move can
            # still be enabled from TrainConfig for flux excitations.
            return "toric"
        if self.magnetization is not None:
            return "pair_flip"
        return "single_flip"

    @property
    def model_input_channels(self) -> int:
        # Toric-code variables are edge qubits: horizontal/vertical edge channels per cell.
        return 2 if self.name == "toric_code" else 1


def make_hamiltonian_spec(
    name: str = "tfim",
    shape: int | tuple[int, ...] = (4, 4),
    J: float = 1.0,
    g: float = 1.0,
    pbc: bool = True,
    magnetization: int | None = None,
    Je: float | None = None,
    Jm: float | None = None,
) -> HamiltonianSpec:
    shape = normalize_shape(shape)
    name = name.lower()
    if name in ("toric", "tc"):
        name = "toric_code"
    if name not in ("tfim", "heisenberg", "toric_code"):
        raise ValueError("name must be 'tfim', 'heisenberg', or 'toric_code'")
    if name == "heisenberg" and magnetization is None:
        # For spin-1/2 AF Heisenberg, the low-energy sector is usually Sz=0.
        # Keeping this default also makes the Metropolis moves ergodic inside the target sector.
        magnetization = 0 if num_sites(shape) % 2 == 0 else None
    if name == "toric_code":
        if len(shape) != 2:
            raise ValueError("toric_code requires a 2D shape, e.g. shape=(2,2) or shape=(4,4).")
        if not pbc:
            raise NotImplementedError("toric_code currently supports only pbc=True.")
        if magnetization is not None:
            raise ValueError("toric_code should not use a fixed-magnetization sector.")
    return HamiltonianSpec(
        name=name,
        shape=shape,
        J=J,
        g=g,
        pbc=pbc,
        magnetization=magnetization,
        Je=J if Je is None else Je,
        Jm=J if Jm is None else Jm,
    )


def _toric_terms(terms: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # terms has shape (2, n_cells, 4): [stars, plaquettes]
    return terms[0], terms[1]


def diag_energy(spins: jnp.ndarray, hspec: HamiltonianSpec, bonds: jnp.ndarray) -> jnp.ndarray:
    """Diagonal energy for one or many configs. spins shape (..., N)."""
    if hspec.name == "tfim":
        s_i = spins[..., bonds[:, 0]]
        s_j = spins[..., bonds[:, 1]]
        return -hspec.J * jnp.sum(s_i * s_j, axis=-1)
    if hspec.name == "heisenberg":
        s_i = spins[..., bonds[:, 0]]
        s_j = spins[..., bonds[:, 1]]
        # s=±1 encodes 2 S^z, hence S^z_i S^z_j = s_i s_j / 4.
        return 0.25 * hspec.J * jnp.sum(s_i * s_j, axis=-1)
    if hspec.name == "toric_code":
        _, plaquettes = _toric_terms(bonds)
        b_p = jnp.prod(spins[..., plaquettes], axis=-1)
        return -hspec.Jm * jnp.sum(b_p, axis=-1)
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
    elif hspec.name == "toric_code":
        stars, _ = _toric_terms(bonds)
        for a in range(stars.shape[0]):
            idx = stars[a]
            flipped = configs.at[:, idx].multiply(-1)
            out = out + (-hspec.Je) * apply_fun(params, flipped)
    else:
        raise ValueError(hspec.name)
    return out


def basis_shape_for_hamiltonian(hspec: HamiltonianSpec) -> tuple[int, ...]:
    """Shape argument to make_basis for the actual number of spin variables."""
    return (hspec.N,) if hspec.name == "toric_code" else hspec.shape


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
        elif hspec.name == "toric_code":
            stars = bonds[0]
            for idx in stars:
                sp = s.copy()
                sp[idx] *= -1
                b = index.get(tuple(sp.tolist()))
                if b is not None:
                    H[b, a] += -hspec.Je
        else:
            raise ValueError(hspec.name)
    return H


def toric_code_exact_ground_energy(hspec: HamiltonianSpec) -> float:
    if hspec.name != "toric_code":
        raise ValueError("only valid for toric_code")
    n_terms = num_sites(hspec.shape)
    return float(-hspec.Je * n_terms - hspec.Jm * n_terms)
