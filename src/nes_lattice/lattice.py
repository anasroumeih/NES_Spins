from __future__ import annotations

from itertools import product
from math import prod
from typing import Iterable

import numpy as np


def normalize_shape(shape: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(x) for x in shape)


def num_sites(shape: int | Iterable[int]) -> int:
    return int(prod(normalize_shape(shape)))


def coord_to_index(coord: tuple[int, ...], shape: tuple[int, ...]) -> int:
    idx = 0
    stride = 1
    for c, L in zip(reversed(coord), reversed(shape)):
        idx += c * stride
        stride *= L
    return idx


def index_to_coord(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coord = []
    for L in reversed(shape):
        coord.append(index % L)
        index //= L
    return tuple(reversed(coord))


def nearest_neighbor_bonds(shape: int | Iterable[int], pbc: bool = True) -> np.ndarray:
    """Undirected nearest-neighbor bonds for 1D or 2D lattices.

    2D is the intended/default use. 1D is supported as the simple exception.
    Duplicates are removed, which avoids double-counting tiny periodic systems.
    """
    shape = normalize_shape(shape)
    if len(shape) not in (1, 2):
        raise ValueError(f"Only 1D or 2D shapes are supported, got shape={shape}")
    bonds: set[tuple[int, int]] = set()
    for coord in product(*[range(L) for L in shape]):
        i = coord_to_index(coord, shape)
        for axis, L_axis in enumerate(shape):
            nb = list(coord)
            nb[axis] += 1
            if nb[axis] >= L_axis:
                if not pbc:
                    continue
                nb[axis] = 0
            j = coord_to_index(tuple(nb), shape)
            if i != j:
                bonds.add(tuple(sorted((i, j))))
    return np.array(sorted(bonds), dtype=np.int32)


def toric_code_edge_index(x: int, y: int, direction: int, shape: int | Iterable[int]) -> int:
    """Index an edge spin of the periodic square-lattice toric code.

    shape=(Lx,Ly).  Each unit cell has two edge qubits:
        direction=0: horizontal edge from (x,y) to (x+1,y)
        direction=1: vertical edge from (x,y) to (x,y+1)

    The flat spin index is 2 * cell_index + direction.
    """
    Lx, Ly = normalize_shape(shape)
    x %= Lx
    y %= Ly
    return 2 * (x * Ly + y) + int(direction)


def toric_code_num_edges(shape: int | Iterable[int]) -> int:
    """Number of edge qubits for the periodic square-lattice toric code."""
    shape = normalize_shape(shape)
    if len(shape) != 2:
        raise ValueError("Toric code requires a 2D shape, e.g. shape=(Lx,Ly).")
    return 2 * num_sites(shape)


def toric_code_terms(shape: int | Iterable[int], pbc: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return star and plaquette terms for the periodic 2D toric code.

    Returns
    -------
    stars, plaquettes:
        Arrays with shape (Lx*Ly, 4).  Each row contains the four edge-qubit
        indices belonging to a star A_s or plaquette B_p.

    Notes
    -----
    The toric-code ground-state degeneracy targeted here requires periodic
    boundary conditions.  Open boundaries have different edge counting and are
    intentionally not implemented in this small research code.
    """
    shape = normalize_shape(shape)
    if len(shape) != 2:
        raise ValueError("Toric code requires a 2D shape, e.g. shape=(Lx,Ly).")
    if not pbc:
        raise NotImplementedError("This toric-code implementation currently supports only pbc=True.")

    Lx, Ly = shape
    stars = []
    plaquettes = []

    for x in range(Lx):
        for y in range(Ly):
            # Star at vertex (x,y): the four incident edges.
            stars.append([
                toric_code_edge_index(x, y, 0, shape),      # horizontal to the right
                toric_code_edge_index(x - 1, y, 0, shape),  # horizontal from the left
                toric_code_edge_index(x, y, 1, shape),      # vertical upward
                toric_code_edge_index(x, y - 1, 1, shape),  # vertical from below
            ])

            # Plaquette with lower-left corner (x,y): four boundary edges.
            plaquettes.append([
                toric_code_edge_index(x, y, 0, shape),      # bottom edge
                toric_code_edge_index(x + 1, y, 1, shape),  # right edge
                toric_code_edge_index(x, y + 1, 0, shape),  # top edge
                toric_code_edge_index(x, y, 1, shape),      # left edge
            ])

    return np.asarray(stars, dtype=np.int32), np.asarray(plaquettes, dtype=np.int32)


def make_basis(shape: int | Iterable[int], magnetization: int | None = None) -> np.ndarray:
    """Enumerate spin configurations with values ±1.

    magnetization is sum(spins). Use magnetization=0 for the Heisenberg Sz=0 sector.
    For models with a non-site number of spins, pass shape=(N,).
    """
    N = num_sites(shape)
    basis = np.array(list(product([-1, 1], repeat=N)), dtype=np.int8)
    if magnetization is not None:
        basis = basis[basis.sum(axis=1) == magnetization]
    return basis


def random_configs(key, n: int, shape: int | Iterable[int], magnetization: int | None = None):
    """JAX random spin configurations. Imported lazily to keep this file NumPy-light."""
    import jax
    import jax.numpy as jnp

    N = num_sites(shape)
    if magnetization is None:
        return 2 * jax.random.bernoulli(key, 0.5, (n, N)).astype(jnp.int8) - 1
    if magnetization != 0:
        raise NotImplementedError("random fixed-magnetization init currently supports only magnetization=0")
    if N % 2 != 0:
        raise ValueError("magnetization=0 requires an even number of sites")
    base = jnp.concatenate([jnp.ones(N // 2, dtype=jnp.int8), -jnp.ones(N // 2, dtype=jnp.int8)])
    keys = jax.random.split(key, n)
    return jax.vmap(lambda kk: jax.random.permutation(kk, base))(keys)


def toric_code_move_masks(shape: int | Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return Z-basis flip masks for toric-code Monte Carlo moves.

    Returns
    -------
    star_masks, winding_masks:
        ``star_masks`` has shape ``(Lx*Ly, 2*Lx*Ly)``.  Row ``s`` flips the
        four edge spins of the star operator :math:`A_s`.

        ``winding_masks`` has shape ``(2, 2*Lx*Ly)``.  The two rows are
        non-contractible dual-lattice loops:

        * row 0 flips every horizontal edge at fixed ``x=0``;
        * row 1 flips every vertical edge at fixed ``y=0``.

    Every one of these moves preserves all plaquette eigenvalues ``B_p``.
    The winding masks are not products of star masks and therefore connect the
    four topological sectors on the torus.
    """
    shape = normalize_shape(shape)
    if len(shape) != 2:
        raise ValueError("Toric-code moves require a 2D shape, e.g. shape=(Lx, Ly).")

    Lx, Ly = shape
    n_edges = toric_code_num_edges(shape)
    stars, _ = toric_code_terms(shape, pbc=True)

    star_masks = np.zeros((stars.shape[0], n_edges), dtype=np.int8)
    for s, edges in enumerate(stars):
        # XOR keeps this correct even in pathological tiny periodic cases.
        for edge in edges:
            star_masks[s, int(edge)] ^= np.int8(1)

    winding_masks = np.zeros((2, n_edges), dtype=np.int8)

    # A dual loop wrapping in the y direction crosses all horizontal edges at
    # a fixed x.  Every plaquette intersects this set in 0 or 2 edges.
    for y in range(Ly):
        winding_masks[0, toric_code_edge_index(0, y, 0, shape)] = 1

    # A dual loop wrapping in the x direction crosses all vertical edges at a
    # fixed y.  Again each plaquette intersects it in 0 or 2 edges.
    for x in range(Lx):
        winding_masks[1, toric_code_edge_index(x, 0, 1, shape)] = 1

    return star_masks, winding_masks


def toric_code_plaquette_values(configs: np.ndarray, shape: int | Iterable[int]) -> np.ndarray:
    """Return all B_p values for NumPy spin configurations of shape (..., N_edges)."""
    _, plaquettes = toric_code_terms(shape, pbc=True)
    configs = np.asarray(configs)
    return np.prod(configs[..., plaquettes], axis=-1)
