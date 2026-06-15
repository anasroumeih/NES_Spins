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


def make_basis(shape: int | Iterable[int], magnetization: int | None = None) -> np.ndarray:
    """Enumerate spin configurations with values ±1.

    magnetization is sum(spins). Use magnetization=0 for the Heisenberg Sz=0 sector.
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
