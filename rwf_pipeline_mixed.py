"""
Generalization of the RWF pipeline (Definition 2) to subsystems of
heterogeneous local dimension.

For H -> ZZ: subsystems are (L: d=9, S1: d=3, S2: d=3).

Key functions:
    - pairwise_reduced_mixed(rho, keep, dims): trace out all but `keep`
    - log_negativity_mixed(rho, dims): log-negativity for a bipartite state
      of arbitrary local dims
    - pairwise_negativity_matrix_mixed(rho, dims): n x n matrix W
    - flag complex + persistent homology from existing rwf_pipeline.py
"""
import sys
sys.path.insert(0, "/home/claude")

import numpy as np
from itertools import combinations

from rwf_pipeline import flag_complex_persistence


def partial_transpose_mixed(rho, subsystem, dims):
    """
    Partial transpose of a bipartite density matrix with local dims (dA, dB)
    w.r.t. `subsystem` (0 for A, 1 for B).

    Works for arbitrary dims, not just (2,2).
    """
    dA, dB = dims
    assert rho.shape == (dA * dB, dA * dB)
    rho_tensor = rho.reshape(dA, dB, dA, dB)
    if subsystem == 0:
        return rho_tensor.transpose(2, 1, 0, 3).reshape(dA * dB, dA * dB)
    elif subsystem == 1:
        return rho_tensor.transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)
    raise ValueError


def log_negativity_mixed(rho, dims):
    """
    Logarithmic negativity for a bipartite state with arbitrary local dims.
    """
    rho_pt = partial_transpose_mixed(rho, 0, dims)
    # rho_pt is Hermitian (PT preserves Hermiticity); eigvalsh is valid.
    eigs = np.linalg.eigvalsh(rho_pt)
    trace_norm = np.sum(np.abs(eigs))
    return np.log2(trace_norm)


def pairwise_reduced_mixed(rho, keep, dims):
    """
    Trace out all subsystems except those in `keep`.

    Parameters
    ----------
    rho : (D, D) density matrix, D = prod(dims)
    keep : sorted list of indices to keep
    dims : list of local dimensions

    Returns
    -------
    rho_reduced : density matrix on the kept subsystems, dim = prod(dims[i] for i in keep)
    """
    keep = sorted(keep)
    n = len(dims)
    assert rho.shape == (int(np.prod(dims)),) * 2

    # Reshape into a tensor of shape (d0, d1, ..., d_{n-1}, d0, d1, ..., d_{n-1})
    shape = list(dims) + list(dims)
    t = rho.reshape(shape)

    # Trace out each subsystem not in keep, iteratively.
    remaining = list(range(n))
    remaining_dims = list(dims)
    for idx in sorted(set(range(n)) - set(keep), reverse=True):
        pos = remaining.index(idx)
        t = np.trace(t, axis1=pos, axis2=pos + len(remaining))
        remaining.pop(pos)
        remaining_dims.pop(pos)

    m = int(np.prod(remaining_dims))
    return t.reshape(m, m)


def pairwise_negativity_matrix_mixed(rho, dims):
    """
    Pairwise log-negativity matrix W for a state with heterogeneous local dims.

    Returns
    -------
    W : (n, n) symmetric array, W[i,j] = E_N(subsystem_i : subsystem_j)
    """
    n = len(dims)
    W = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        rho_ij = pairwise_reduced_mixed(rho, [i, j], dims)
        dij = (dims[i], dims[j])
        en = log_negativity_mixed(rho_ij, dij)
        W[i, j] = W[j, i] = max(en, 0.0)
    return W


def topological_signature_mixed(rho, dims, max_dim=2):
    """
    Full topological signature of an n-partite state with heterogeneous dims.
    """
    W = pairwise_negativity_matrix_mixed(rho, dims)
    pers = flag_complex_persistence(W, max_dim=max_dim)
    diagrams = {k: [] for k in range(max_dim + 1)}
    for dim, interval in pers:
        if dim <= max_dim:
            diagrams[dim].append(interval)
    return {"W": W, "persistence": pers, "diagrams": diagrams}
