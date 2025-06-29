"""Utilities for working with discrete action spaces."""

import numpy as np
from numpy.typing import NDArray


def compute_radix(nvec: NDArray[np.int32]) -> NDArray[np.int32]:
    """
    Compute the radix for encoding multi-discrete values.

    Parameters
    ----------
    nvec : NDArray[np.int32]
        Array representing the number of discrete values for each dimension.

    Returns
    -------
    NDArray[np.int32]
        Radix array used for encoding multi-discrete values into a single integer.
    """
    radix = np.empty_like(nvec)
    radix[1:] = nvec[::-1][:-1]
    radix[0] = 1
    return np.cumprod(radix, dtype=np.int32)[::-1]


def encode_multi_discrete(multidiscrete_vector: NDArray[np.int32], radix: NDArray[np.int32]) -> int:
    """
    Encode a multi-discrete vector into a single integer.

    Parameters
    ----------
    multidiscrete_vector : NDArray[np.int32]
        Multi-discrete vector to encode.
    radix : NDArray[np.int32]
        Radix array for encoding.

    Returns
    -------
    int
        Encoded integer representation of the multi-discrete vector.
    """
    return int(np.dot(multidiscrete_vector, radix))


def encode_multi_discretes(
    multidiscrete_vectors: NDArray[np.int32], radixes: NDArray[np.int32]
) -> NDArray[np.int32]:
    """
    Encode multiple multi-discrete vectors into integers.

    Parameters
    ----------
    multidiscrete_vectors : NDArray[np.int32]
        Array of multi-discrete vectors to encode.
    radixes : NDArray[np.int32]
        Array of radix values for encoding.

    Returns
    -------
    NDArray[np.int32]
        Array of encoded integers.
    """
    return np.sum(multidiscrete_vectors * radixes, axis=1)


def decode_to_multi_discrete(
    nvec: NDArray[np.int32], index: int, radix: NDArray[np.int32]
) -> NDArray[np.int32]:
    """
    Decode an integer back to a multi-discrete vector.

    Parameters
    ----------
    nvec : NDArray[np.int32]
        Array representing the number of discrete values for each dimension.
    index : int
        Integer to decode.
    radix : NDArray[np.int32]
        Radix array used for decoding.

    Returns
    -------
    NDArray[np.int32]
        Decoded multi-discrete vector.
    """
    return (index // radix) % nvec


def decode_to_multi_discretes(
    nvecs: NDArray[np.int32], indices: NDArray[np.int32], radixes: NDArray[np.int32]
) -> NDArray[np.int32]:
    """
    Decode multiple integers back to multi-discrete vectors.

    Parameters
    ----------
    nvecs : NDArray[np.int32]
        Array representing the number of discrete values for each dimension.
    indices : NDArray[np.int32]
        Array of integers to decode.
    radixes : NDArray[np.int32]
        Array of radix values for decoding.

    Returns
    -------
    NDArray[np.int32]
        Array of decoded multi-discrete vectors.
    """
    return (indices // radixes) % nvecs
