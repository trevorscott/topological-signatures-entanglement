"""
Diagnostic: look at the eigenvalues of the reconstructed rho in each m_tt bin
to see if we're losing an entanglement signal or if the state really is
separable by the log-negativity criterion.

Also compute D-tilde (= -(C_nn - C_rr - C_kk)/3), which is CMS's high-mass
entanglement observable, for each bin, since the published >5σ entanglement
result in the boosted regime uses D-tilde, not D.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from rwf_pipeline import (
    two_qubit_rho_from_correlation,
    partial_transpose,
    log_negativity,
    entanglement_marker_D,
)
from parse_cms_differential import parse_measurements, COEFF_ORDER
from cms_differential_analysis import build_C_matrix, compute_EN


def D_tilde(C):
    """D-tilde observable used by CMS in the boosted regime.
    At high m(tt) the state transitions from singlet-like to triplet-like,
    and the sign structure of C flips on the r and k axes. D-tilde flips
    the same signs to recover an entanglement witness:
        D-tilde = -(C_nn - C_rr - C_kk) / 3.
    Entanglement if D-tilde > 1/3 (or equivalently, the quantity -D-tilde < -1/3).
    """
    # Index order in C: (k, n, r) = (0, 1, 2)
    return (C[1, 1] - C[2, 2] - C[0, 0]) / 3.0


def analyze_bin(bin_data, bin_label):
    print(f"--- {bin_label} ---")
    values = {c: bin_data[c]["value"] for c in COEFF_ORDER}

    P1 = np.array([values["P1k"], values["P1n"], values["P1r"]])
    P2 = np.array([values["P2k"], values["P2n"], values["P2r"]])
    C  = build_C_matrix(
        values["Ckk"], values["Cnn"], values["Crr"],
        values["Cnr+"], values["Crk+"], values["Cnk+"],
        values["Cnr-"], values["Crk-"], values["Cnk-"],
    )

    rho = two_qubit_rho_from_correlation(P1, P2, C, sign_convention="atlas")
    rho_eigs = np.linalg.eigvalsh(rho)
    rho_PT = partial_transpose(rho, 0, (2, 2))
    pt_eigs = np.linalg.eigvalsh(rho_PT)

    D = entanglement_marker_D(C)
    Dt = D_tilde(C)

    print(f"  C matrix:")
    for i, lab in enumerate(["k", "n", "r"]):
        row = "  ".join(f"{C[i, j]:+.4f}" for j in range(3))
        print(f"    {lab}   {row}")
    print()
    print(f"  D              = {D:+.4f}   (entanglement: D < -1/3)")
    print(f"  D-tilde        = {Dt:+.4f}   (CMS high-mass witness: D-tilde > 1/3)")
    print()
    print(f"  rho eigenvalues: {np.round(rho_eigs, 4)}")
    print(f"  min eig (rho)  = {rho_eigs.min():+.6f}  "
          f"{'[unphysical]' if rho_eigs.min() < -1e-6 else '[ok]'}")
    print(f"  rho^TA eigenvalues: {np.round(pt_eigs, 4)}")
    print(f"  min eig (rho^TA) = {pt_eigs.min():+.6f}  "
          f"{'[entangled]' if pt_eigs.min() < -1e-6 else '[separable]'}")
    print(f"  E_N = log2(||rho^TA||_1) = {log_negativity(rho, (2,2)):.4f}")
    print()


def main():
    # Find the CSV
    import os
    candidates = [
        "data/cms_full_matrix_differential_mtt.csv",
        "cms_full_matrix_differential_mtt.csv",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print("ERROR: CSV not found in data/ or cwd")
        return

    bins, bin_order = parse_measurements(path)
    print(f"Parsed {len(bin_order)} bins\n")

    for b in bin_order:
        if len(bins[b]) >= 16:
            analyze_bin(bins[b], b)


if __name__ == "__main__":
    main()
