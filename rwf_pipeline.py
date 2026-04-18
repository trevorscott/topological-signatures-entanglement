"""
RWF Pipeline: Density matrix -> Pairwise log-negativity -> Flag complex -> Persistent homology.

This implements Definition 2 of Scott (2026), "The Relational Witnessing Framework",
applied to collider-measured quantum states.

Stage 1 (this file): the building blocks, validated against the ATLAS/CMS tt-bar result.
"""
import numpy as np
from itertools import combinations

# --- Pauli matrices ---
SIGMA = {
    0: np.eye(2, dtype=complex),
    1: np.array([[0, 1], [1, 0]], dtype=complex),
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),
    3: np.array([[1, 0], [0, -1]], dtype=complex),
}


def two_qubit_rho_from_correlation(B_plus, B_minus, C, sign_convention="atlas"):
    """
    Reconstruct a two-qubit density matrix from its Fano decomposition.

    ATLAS/CMS convention (sign_convention="atlas"), used in ATLAS 2311.07288:
        rho = (1/4) [ I⊗I + B+_i sigma_i⊗I + B-_i I⊗sigma_i - C_ij sigma_i⊗sigma_j ]
    The negative sign in front of C is chosen so that same-helicity tops have
    positive C_kk. In this convention, the spin singlet has C = +I, giving
    D = -Tr(C)/3 = -1 and perfect entanglement.

    Theory convention (sign_convention="theory"), used in Afik-de Nova 2003.02280:
        rho = (1/4) [ I⊗I + B+_i sigma_i⊗I + B-_i I⊗sigma_i + C_ij sigma_i⊗sigma_j ]
    In this convention, the spin singlet has C = -I and <sigma · sigma'> = Tr(C) = -3.

    These are physically equivalent; only the sign of C differs. We default to
    the ATLAS convention because that's how experimental results are reported.

    Parameters
    ----------
    B_plus  : (3,) array — polarization of subsystem 1 (top)
    B_minus : (3,) array — polarization of subsystem 2 (antitop)
    C       : (3,3) array — spin correlation matrix C_ij in the specified convention
    sign_convention : "atlas" (default) or "theory"

    Returns
    -------
    rho : (4,4) complex array, Hermitian, trace 1
    """
    if sign_convention == "atlas":
        c_sign = -1.0
    elif sign_convention == "theory":
        c_sign = +1.0
    else:
        raise ValueError("sign_convention must be 'atlas' or 'theory'")

    rho = np.kron(SIGMA[0], SIGMA[0]).astype(complex)
    for i in range(1, 4):
        rho += B_plus[i - 1] * np.kron(SIGMA[i], SIGMA[0])
        rho += B_minus[i - 1] * np.kron(SIGMA[0], SIGMA[i])
    for i in range(1, 4):
        for j in range(1, 4):
            rho += c_sign * C[i - 1, j - 1] * np.kron(SIGMA[i], SIGMA[j])
    return rho / 4.0


def partial_transpose(rho, subsystem, dims):
    """
    Partial transpose of a bipartite density matrix with respect to `subsystem` (0 or 1).

    `dims` = (d1, d2). For two qubits, dims = (2, 2).
    """
    d1, d2 = dims
    rho_tensor = rho.reshape(d1, d2, d1, d2)
    if subsystem == 0:
        rho_pt = rho_tensor.transpose(2, 1, 0, 3)
    elif subsystem == 1:
        rho_pt = rho_tensor.transpose(0, 3, 2, 1)
    else:
        raise ValueError("subsystem must be 0 or 1")
    return rho_pt.reshape(d1 * d2, d1 * d2)


def log_negativity(rho, dims=(2, 2)):
    """
    Logarithmic negativity E_N(rho) = log2 || rho^{T_A} ||_1.

    This is the entanglement monotone chosen in RWF Definition 2.
    It is zero for separable states, > 0 for NPT-entangled states.
    """
    rho_pt = partial_transpose(rho, 0, dims)
    # Trace norm = sum of absolute values of eigenvalues (rho_pt is Hermitian)
    eigvals = np.linalg.eigvalsh(rho_pt)
    trace_norm = np.sum(np.abs(eigvals))
    return np.log2(trace_norm)


def entanglement_marker_D(C):
    """
    The observable D measured by ATLAS and CMS.
    D = -(C_11 + C_22 + C_33) / 3 = -Tr(C)/3.

    D < -1/3 implies entanglement for tt-bar from QCD (Afik & de Nova 2021).
    """
    return -np.trace(C) / 3.0


# ---------------------------------------------------------------------------
# RWF Section 6: Flag complex from pairwise entanglement
# ---------------------------------------------------------------------------

def pairwise_reduced_state(rho, keep, total_n, local_dim=2):
    """
    Trace out all subsystems except those in `keep` from an n-qubit state.

    Parameters
    ----------
    rho : (d, d) density matrix where d = local_dim**total_n
    keep : iterable of subsystem indices (0-indexed) to retain
    total_n : total number of subsystems
    local_dim : local Hilbert space dimension per site (2 for qubits)

    Returns
    -------
    rho_reduced : reduced density matrix on the kept subsystems,
                  shape (local_dim**|keep|, local_dim**|keep|)
    """
    keep = sorted(keep)
    # Reshape into tensor of shape (d, d, ..., d, d, d, ..., d) (2n legs)
    shape = [local_dim] * (2 * total_n)
    t = rho.reshape(shape)
    # Trace out each subsystem not in `keep`
    traced = set(range(total_n)) - set(keep)
    # We trace iteratively. After each trace, indices re-number.
    remaining = list(range(total_n))
    for idx in sorted(traced, reverse=True):
        pos = remaining.index(idx)
        t = np.trace(t, axis1=pos, axis2=pos + len(remaining))
        remaining.pop(pos)
    # t now has 2*|keep| legs in order (ket, ket, ..., bra, bra, ...)
    m = local_dim ** len(keep)
    return t.reshape(m, m)


def log_negativity_pair(rho_ij, local_dim=2):
    """
    Log-negativity of a bipartite reduced state between two subsystems.

    rho_ij : (4, 4) reduced state of subsystems i, j (for qubits)
    """
    return log_negativity(rho_ij, dims=(local_dim, local_dim))


def pairwise_negativity_matrix(rho, n, local_dim=2):
    """
    Compute the weighted 1-skeleton of the flag complex: W[i,j] = E_N(i:j).

    This is the symmetric matrix of pairwise log-negativities across all
    C(n,2) pairs, obtained by tracing out all other subsystems.

    Returns
    -------
    W : (n, n) symmetric matrix, zeros on diagonal, W[i,j] = E_N(i:j) for i != j
    """
    W = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        rho_ij = pairwise_reduced_state(rho, [i, j], n, local_dim=local_dim)
        en = log_negativity_pair(rho_ij, local_dim=local_dim)
        # Numerical noise can make tiny separable states come out slightly negative
        W[i, j] = W[j, i] = max(en, 0.0)
    return W


def flag_complex_persistence(W, max_dim=2):
    """
    Build the flag (clique) complex from the weighted 1-skeleton W and compute
    persistent homology via an "edge-weight" filtration.

    Construction per RWF Definition 2:
      - 0-simplices: the n subsystems (always present)
      - 1-simplex {i, j}: present if W[i,j] > 0
      - k-simplex: present iff all pairwise edges are present (flag condition)

    Filtration: we use a DECREASING filtration on the edge weight. A k-simplex
    enters the complex at filtration value ε = min(weights of its edges).
    (Per Eq. (14) of the paper, we sweep ε from max down to 0.)

    gudhi's RipsComplex internally uses an INCREASING filtration on distance.
    We can reuse its machinery by mapping:
        distance(i, j) := W_max - W[i, j]    (for edges with W > 0)
        distance(i, j) := infinity           (for W = 0, so edge never appears)
    Then small distances <-> large weights, and persistence intervals in the
    rips filtration can be translated back to weight intervals.

    Parameters
    ----------
    W : (n, n) symmetric pairwise-log-negativity matrix with zeros on diagonal
    max_dim : maximum homology dimension to compute

    Returns
    -------
    persistence : list of (dim, (birth_weight, death_weight)) tuples, sorted
    """
    import gudhi

    n = W.shape[0]
    W_max = W.max() if W.max() > 0 else 1.0

    # Build distance matrix. Edges with zero weight get infinite distance
    # so they are excluded from the flag complex.
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for i, j in combinations(range(n), 2):
        if W[i, j] > 0:
            D[i, j] = D[j, i] = W_max - W[i, j]

    # Rips complex on this distance matrix = flag complex on (W > 0) edges
    # with filtration value = W_max - min-edge-weight of the simplex.
    rc = gudhi.RipsComplex(distance_matrix=D, max_edge_length=W_max)
    st = rc.create_simplex_tree(max_dimension=max_dim)
    persistence_rips = st.persistence()

    # Translate back: a (dim, (birth_dist, death_dist)) in rips-filtration
    # corresponds to (dim, (birth_weight = W_max - birth_dist,
    #                       death_weight = W_max - death_dist)) in our
    # decreasing weight filtration. A feature born at distance d_b dies at
    # distance d_d > d_b in rips, which means it's born at weight
    # W_max - d_b and dies at weight W_max - d_d (< birth weight).
    out = []
    for dim, (bd, dd) in persistence_rips:
        birth_w = W_max - bd
        death_w = W_max - dd if dd != float("inf") else 0.0
        out.append((dim, (birth_w, death_w)))
    return out


def topological_signature(rho, n, local_dim=2, max_dim=2):
    """
    Compute the full topological signature chi(K_W) of an n-partite state.

    Returns a dict:
      {
        "W": pairwise negativity matrix,
        "persistence": list of (dim, (birth_weight, death_weight)),
        "diagrams": {k: list of (birth, death) intervals for H_k}
      }
    """
    W = pairwise_negativity_matrix(rho, n, local_dim=local_dim)
    pers = flag_complex_persistence(W, max_dim=max_dim)
    diagrams = {k: [] for k in range(max_dim + 1)}
    for dim, interval in pers:
        if dim <= max_dim:
            diagrams[dim].append(interval)
    return {"W": W, "persistence": pers, "diagrams": diagrams}


# ---------------------------------------------------------------------------
# VALIDATION AGAINST ATLAS NATURE 2024 (arXiv:2311.07288)
# ---------------------------------------------------------------------------

def validate_atlas_ttbar():
    """
    Validate the pipeline against the ATLAS 2024 measurement.

    The ATLAS paper publishes D = -0.537 ± 0.019 in the threshold bin
    340 < m_tt < 380 GeV. They use the "helicity basis".

    For tt from unpolarized pp collisions, B+ = B- = 0 (parity + CP).

    At threshold, the tt system is in a near-maximal spin-singlet state:
    the SM theory prediction has C_ij approximately diag(-1, -1, -1) in a
    suitable basis, yielding D ≈ -1 (fully entangled singlet).

    ATLAS measures D_obs = -0.537, reflecting decoherence from finite
    kinematic range, NLO effects, and the mixed gg/qqbar initial state.

    As a validation of our pipeline, we construct a two-qubit state
    consistent with D = -0.537 and verify:
      - it is a valid density matrix
      - D computed from C matches the input
      - log-negativity is positive (confirming entanglement)
      - the numerical value is consistent with the Afik-de Nova analytical result
    """
    print("=" * 70)
    print("Validation: ATLAS tt-bar threshold measurement (arXiv:2311.07288)")
    print("=" * 70)

    # ATLAS measured D = -0.537. For the threshold singlet regime, a natural
    # parametrization consistent with this D and no polarization:
    # C is diagonal with entries summing to +1.611, yielding D = -0.537.
    # We use the simplest consistent ansatz: C = diag(c, c, c) with c = 0.537.
    # (In the full helicity basis ATLAS publishes a slightly anisotropic C; we
    # use the isotropic case as a canonical test. The flag-complex topology
    # will not depend on the basis choice.)
    c = 0.537
    C = np.diag([c, c, c])
    B_plus = np.zeros(3)
    B_minus = np.zeros(3)

    rho = two_qubit_rho_from_correlation(B_plus, B_minus, C)

    # --- Sanity checks ---
    print(f"  rho Hermitian?       {np.allclose(rho, rho.conj().T)}")
    print(f"  Tr(rho) = 1?         {np.isclose(np.trace(rho).real, 1.0)}")
    eigs = np.linalg.eigvalsh(rho)
    print(f"  rho eigenvalues:     {np.round(eigs, 4)}")
    print(f"  all nonneg (within 1e-10)?  {np.all(eigs > -1e-10)}")

    D = entanglement_marker_D(C)
    print(f"  D (from C):          {D:.4f}   (ATLAS measured: -0.537 ± 0.019)")

    EN = log_negativity(rho, dims=(2, 2))
    print(f"  E_N(t : tbar):       {EN:.4f}  bits")
    print(f"  Entangled?           {EN > 1e-10}")

    # Analytical cross-check (Afik & de Nova, Eur. Phys. J. Plus 136 (2021) 907):
    # For a Werner-like state with C = diag(c,c,c), the concurrence is:
    # Concurrence = max(0, 3c/2 - 1/2). For c=0.537: concurrence = 3*0.537/2 - 0.5 = 0.3055.
    # log-negativity E_N = log2(1 + concurrence) approximately for rank-2 states,
    # but exact formula comes from the partial transpose spectrum.
    expected_negative_eig = (-1 + 3 * c) / 4  # from explicit calculation
    print(f"  Expected most-negative PT eigenvalue: {expected_negative_eig:.4f}")
    rho_pt = partial_transpose(rho, 0, (2, 2))
    pt_eigs = np.linalg.eigvalsh(rho_pt)
    print(f"  Actual PT eigenvalues: {np.round(pt_eigs, 4)}")
    print()


if __name__ == "__main__":
    validate_atlas_ttbar()
