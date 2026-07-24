from typing import Literal

import numpy as np


def low_rank_approximation(
    matrix, rank, method: Literal["svd", "eigen"] = "svd"
) -> np.ndarray:
    """
    Perform low-rank approximation of a matrix using Singular Value Decomposition (SVD).

    Args:
        matrix (np.ndarray): Input matrix to approximate.
        rank (int): Desired rank for the approximation.

    Returns:
        np.ndarray: Approximated matrix of the specified rank.
    """
    if method == "svd":
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        U_truncated = U[:, :rank]
        s_truncated = s[:rank]
        Vt_truncated = Vt[:rank, :]

    return U_truncated @ np.diag(s_truncated) @ Vt_truncated
