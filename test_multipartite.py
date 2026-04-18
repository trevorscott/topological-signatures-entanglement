"""
Sanity tests for the RWF pipeline on known n-qubit states.

The physics/math expectations:

1. GHZ_n = (|0...0> + |1...1>)/sqrt(2)
   - Tracing out any 1 qubit leaves a separable (classical) mixture on the rest.
   - Pairwise log-negativity E_N(i:j) = 0 for all pairs (this is the classic
     "monogamy of entanglement" in its extreme form).
   - Therefore the flag complex has NO edges. Only n connected components.
   - So: H_0 has n disconnected points that all live forever, H_k = 0 for k > 0.
   - This confirms Remark 8 of the paper: the flag complex fails to capture
     genuinely multipartite GHZ entanglement.

2. 4-qubit cluster state |C_4> has nearest-neighbor entanglement on a chain.
   - Pairwise E_N is nonzero only between nearest neighbors.
   - The 1-skeleton is the path graph 0-1-2-3.
   - The flag complex is just the path: no higher simplices form.
   - H_0 has 1 connected component (the path is connected), H_1 = 0 (no cycle).

3. 3-qubit W state = (|100> + |010> + |001>)/sqrt(3)
   - Has nonzero pairwise entanglement for every pair.
   - The 1-skeleton is the complete graph K_3.
   - With 3 edges, the flag complex can contain the 2-simplex {0,1,2}.
   - H_0 = 1, H_1 = 0, H_2 = 0.

4. A 4-qubit state with entangled pairs (0-1, 2-3) but no cross-entanglement.
   - 1-skeleton: two disjoint edges.
   - H_0 = 2 (two connected components), H_1 = 0.
"""
import sys
sys.path.insert(0, "/home/claude")

import numpy as np
from rwf_pipeline import (
    pairwise_negativity_matrix,
    topological_signature,
    log_negativity,
)


def ket(bits):
    """Make a computational basis ket |bits> as a (2**n,) array."""
    n = len(bits)
    vec = np.zeros(2**n, dtype=complex)
    idx = int(bits, 2) if isinstance(bits, str) else int("".join(str(b) for b in bits), 2)
    vec[idx] = 1.0
    return vec


def density(psi):
    return np.outer(psi, psi.conj())


def ghz_state(n):
    psi = (ket("0" * n) + ket("1" * n)) / np.sqrt(2)
    return density(psi)


def w_state_3():
    psi = (ket("100") + ket("010") + ket("001")) / np.sqrt(3)
    return density(psi)


def two_disjoint_bell_pairs():
    """|Phi+>_{01} (x) |Phi+>_{23} on 4 qubits."""
    phi_plus = (ket("00") + ket("11")) / np.sqrt(2)
    psi = np.kron(phi_plus, phi_plus)
    return density(psi)


def cluster_state_4():
    """
    4-qubit linear cluster state: |C_4> = CZ_{01} CZ_{12} CZ_{23} |+>^4
    """
    plus = np.array([1, 1]) / np.sqrt(2)
    psi = np.kron(np.kron(np.kron(plus, plus), plus), plus)
    # Apply CZ_{01}
    CZ = np.diag([1, 1, 1, -1])
    I = np.eye(2)
    # CZ acting on qubits 0,1:  CZ_{01} otimes I_2 otimes I_3
    CZ01 = np.kron(np.kron(CZ, I), I)
    # CZ acting on qubits 1,2:  I_0 otimes CZ_{12} otimes I_3
    CZ12 = np.kron(np.kron(I, CZ), I)
    # Hmm this isn't quite right because CZ on qubits 1,2 needs the right tensor order.
    # Let me be more careful. Qubits are ordered [0,1,2,3]. CZ_{ij} is diagonal in the
    # computational basis, giving -1 when qubits i and j are both 1.

    n = 4
    def CZ_on(i, j, n):
        op = np.zeros((2**n, 2**n))
        for k in range(2**n):
            bits = [(k >> (n - 1 - b)) & 1 for b in range(n)]
            sign = -1 if (bits[i] == 1 and bits[j] == 1) else 1
            op[k, k] = sign
        return op

    CZ01 = CZ_on(0, 1, n)
    CZ12 = CZ_on(1, 2, n)
    CZ23 = CZ_on(2, 3, n)
    psi = CZ23 @ CZ12 @ CZ01 @ psi
    return density(psi)


def show_signature(name, rho, n):
    print(f"--- {name} (n={n}) ---")
    sig = topological_signature(rho, n, local_dim=2, max_dim=min(n - 1, 2))
    W = sig["W"]
    print("  Pairwise log-negativity matrix W[i,j]:")
    for row in W:
        print("    ", "  ".join(f"{w:6.3f}" for w in row))

    for dim, diag in sig["diagrams"].items():
        if len(diag) == 0:
            continue
        # Sort by birth descending (large weight = early in filtration)
        diag_sorted = sorted(diag, key=lambda x: -x[0])
        print(f"  H_{dim}: {len(diag)} feature(s)")
        for (b, d) in diag_sorted:
            life = b - d if d != 0 else b
            print(f"     birth={b:.4f}, death={d:.4f}  (persistence={life:.4f})")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Sanity tests: RWF flag-complex pipeline on known multipartite states")
    print("=" * 70)
    print()

    # 1. GHZ_3: zero pairwise log-neg expected
    show_signature("GHZ_3", ghz_state(3), 3)

    # 2. GHZ_4: same, for 4 qubits
    show_signature("GHZ_4", ghz_state(4), 4)

    # 3. W_3: all 3 pairs entangled, forms K_3 flag complex
    show_signature("W_3", w_state_3(), 3)

    # 4. Two disjoint Bell pairs: two disconnected edges
    show_signature("Two disjoint Bell pairs on 4 qubits", two_disjoint_bell_pairs(), 4)

    # 5. 4-qubit linear cluster state: path graph
    show_signature("Cluster state |C_4>", cluster_state_4(), 4)
