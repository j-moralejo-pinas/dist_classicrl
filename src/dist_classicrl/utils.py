"""
Utilities for working with discrete action spaces.
"""

import numpy as np
from numpy.typing import NDArray


def compute_radix(nvec: NDArray[np.int32]) -> NDArray[np.int32]:
    radix = np.empty_like(nvec)
    radix[1:] = nvec[::-1][:-1]
    radix[0] = 1
    return np.cumprod(radix, dtype=np.int32)[::-1]


def encode_multi_discrete(multidiscrete_vector: NDArray[np.int32], radix: NDArray[np.int32]) -> int:
    return int(np.dot(multidiscrete_vector, radix))


def encode_multi_discretes(
    multidiscrete_vectors: NDArray[np.int32], radixes: NDArray[np.int32]
) -> NDArray[int]:
    return np.sum(multidiscrete_vectors * radixes, axis=1)


def decode_to_multi_discrete(
    nvec: NDArray[np.int32], index: int, radix: NDArray[np.int32]
) -> NDArray[np.int32]:
    return (index // radix) % nvec


def decode_to_multi_discretes(
    nvecs: NDArray[np.int32], indices: NDArray[int], radixes: NDArray[np.int32]
) -> NDArray[np.int32]:
    return (indices // radixes) % nvecs
