"""
Apply the full RWF Section 6 pipeline to H -> ZZ.

Subsystems: (L, S_{Z1}, S_{Z2}) with dims (9, 3, 3).

Physics expectations (Aguilar-Saavedra 2403.13942):
  - The state is GENUINELY tripartite-entangled (no bipartition separable).
  - Tracing over L leaves a highly entangled (S1,S2) state.
  - At the characteristic mZ_off ~ 30 GeV, the amplitudes are mostly balanced.
  - We scan mV_off from 10 to 70 GeV to see how the topology evolves.

We output:
  - E_N(L:S1), E_N(L:S2), E_N(S1:S2) (the three edges of the flag complex)
  - Persistent homology diagrams
  - Whether the 2-simplex {L,S1,S2} is present (all three edges > 0)
"""
import numpy as np
import sys
sys.path.insert(0, "/home/claude")

from hvv_state import build_hvv_state, sm_hvv_amplitudes
from rwf_pipeline_mixed import topological_signature_mixed


def analyze_hvv(mV_off, MV=91.1876, MH=125.10, label=""):
    a11, a00, am1m1 = sm_hvv_amplitudes(mV_off, MV, MH)
    # Normalize
    scale = max(abs(a11), abs(a00), abs(am1m1))
    a11 /= scale; a00 /= scale; am1m1 /= scale

    psi = build_hvv_state(a11, a00, am1m1)
    rho = np.outer(psi, psi.conj())
    dims = [9, 3, 3]

    sig = topological_signature_mixed(rho, dims, max_dim=2)

    print(f"--- {label} (mV_off = {mV_off} GeV) ---")
    print(f"  Amplitudes: a_11={a11:.3f}, a_00={a00:.3f}, a_-1-1={am1m1:.3f}")
    W = sig["W"]
    print(f"  Pairwise log-negativity matrix (subsystems = [L, S1, S2]):")
    labels = ["L ", "S1", "S2"]
    print(f"         {'  '.join(labels)}")
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{W[i,j]:5.3f}" for j in range(3))
        print(f"    {row_label}  {row}")

    e_LS1 = W[0, 1]; e_LS2 = W[0, 2]; e_S1S2 = W[1, 2]
    print(f"  E_N(L : S1)   = {e_LS1:.4f} bits")
    print(f"  E_N(L : S2)   = {e_LS2:.4f} bits")
    print(f"  E_N(S1: S2)   = {e_S1S2:.4f} bits")

    all_edges_present = min(e_LS1, e_LS2, e_S1S2) > 1e-6
    print(f"  All 3 edges > 0 (2-simplex in flag complex)? {all_edges_present}")

    print(f"  Persistence diagrams:")
    for dim, diag in sig["diagrams"].items():
        if len(diag) == 0:
            continue
        diag_sorted = sorted(diag, key=lambda x: -x[0])
        for (b, d) in diag_sorted:
            persistence = b - d
            print(f"    H_{dim}: birth = {b:.4f}, death = {d:.4f}, persistence = {persistence:.4f}")
    print()
    return sig


def main():
    print("=" * 70)
    print("RWF Section 6 applied to H -> ZZ tripartite state")
    print("=" * 70)
    print("Subsystems: (L, S_Z1, S_Z2), dimensions (9, 3, 3)")
    print("Tripartite pure state per Aguilar-Saavedra, arXiv:2403.13942")
    print()

    MZ = 91.1876
    MH = 125.10

    # Scan invariant mass of off-shell Z from threshold to near m_H - m_Z
    mV_off_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    results = {}
    for mV in mV_off_values:
        label = f"H -> ZZ*"
        sig = analyze_hvv(mV, MZ, MH, label=label)
        results[mV] = sig

    # Summary table
    print("=" * 70)
    print("Summary: topological signature as a function of m_Z_off")
    print("=" * 70)
    print(f"{'mV_off':>8}  {'E_N(L:S1)':>11}  {'E_N(L:S2)':>11}  {'E_N(S1:S2)':>11}  "
          f"{'H_0 count':>10}  {'H_1 count':>10}")
    for mV in mV_off_values:
        W = results[mV]["W"]
        h0 = len(results[mV]["diagrams"].get(0, []))
        h1 = len(results[mV]["diagrams"].get(1, []))
        print(f"{mV:>8.1f}  {W[0,1]:>11.4f}  {W[0,2]:>11.4f}  {W[1,2]:>11.4f}  "
              f"{h0:>10d}  {h1:>10d}")


if __name__ == "__main__":
    main()
