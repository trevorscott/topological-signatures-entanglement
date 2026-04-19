"""
Real CMS measured tt-bar analysis, from HEPData record ins2829523.

Paper: CMS Collaboration, Phys. Rev. D 110 (2024) 112016, arXiv:2409.11067.
Data:  https://www.hepdata.net/record/ins2829523
       Table t1 (full matrix inclusive from m(tt))

This script:
  1. Loads the measured 15 coefficients + stat/syst uncertainties.
  2. Reconstructs the 4x4 tt-bar spin density matrix in the helicity basis.
  3. Computes log-negativity E_N from the reconstructed rho.
  4. Propagates uncertainties via Monte Carlo: sample coefficients from their
     uncertainty distributions, recompute E_N, report central value and 68% CL.
  5. Compares against four SM theory predictions (POWHEG+P8, POWHEG+H7,
     MG5_aMC+P8, MiNNLO+P8) also provided in the HEPData record.

Key physical point: the ATLAS and CMS helicity-basis C matrix has the structure
    C = diag(C_kk, C_nn, C_rr)  PLUS  C_rk (= C_kr) off-diagonal
with C_nr = C_nk = 0 at LO by parity.  CMS publishes C^+ and C^- combinations
which correspond to:
    C^+_ij = (C_ij + C_ji)/2   (symmetric part, physical)
    C^-_ij = (C_ij - C_ji)/2   (antisymmetric, should be zero at LO)
"""
import numpy as np
import sys
sys.path.insert(0, ".")  # so it works regardless of where it's run
from rwf_pipeline import (
    two_qubit_rho_from_correlation,
    log_negativity,
    partial_transpose,
    entanglement_marker_D,
)


# ============================================================================
# CMS measured values (inclusive, from m(tt) fit)
# Source: HEPData ins2829523, table t1
# ============================================================================
# Central values
P1  = np.array([0.01347,    0.0080024,  -0.0036959])  # P_k, P_n, P_r  (top)
P2  = np.array([0.014267,   0.0036164,  -0.016752])   # Pbar_k, Pbar_n, Pbar_r (antitop)

C_kk = 0.3053
C_nn = 0.33004
C_rr = 0.027871

# Symmetric (physical) and antisymmetric combinations
# CMS convention: C^+_{ij} = (C_{ij} + C_{ji})/2,  C^-_{ij} = (C_{ij} - C_{ji})/2
C_nr_plus  =  0.014206
C_rk_plus  = -0.20758
C_nk_plus  =  0.0089442
C_nr_minus = -0.015895
C_rk_minus = -0.0088241
C_nk_minus =  0.021695

# Uncertainties (stat, syst) — symmetric in the CMS table
stat = {
    "P1k": 0.0042501, "P1n": 0.0060895, "P1r": 0.0072204,
    "P2k": 0.0042312, "P2n": 0.0060441, "P2r": 0.0071795,
    "Ckk": 0.016719,  "Cnn": 0.0083106, "Crr": 0.015677,
    "Cnr+": 0.018296, "Crk+": 0.032952, "Cnk+": 0.025831,
    "Cnr-": 0.016533, "Crk-": 0.029678, "Cnk-": 0.024966,
}
syst = {
    "P1k": 0.0053078, "P1n": 0.0014546, "P1r": 0.002796,
    "P2k": 0.0051544, "P2n": 0.0013917, "P2r": 0.0030435,
    "Ckk": 0.011012,  "Cnn": 0.0061331, "Crr": 0.0057788,
    "Cnr+": 0.0039482,"Crk+": 0.011615, "Cnk+": 0.0058753,
    "Cnr-": 0.0035233,"Crk-": 0.0066426,"Cnk-": 0.0057692,
}

# SM theory predictions for comparison
SM_PREDICTIONS = {
    "POWHEG+P8":   {"Ckk": 0.30792, "Cnn": 0.31645, "Crr": 0.042193,
                    "Crk+": -0.20621, "Cnk+": 5.4577e-05, "Cnr+": 0.00044545,
                    "Crk-": 0.00089932, "Cnk-": 0.00022864, "Cnr-": -6.59e-05},
    "POWHEG+H7":   {"Ckk": 0.29076, "Cnn": 0.30427, "Crr": 0.032414,
                    "Crk+": -0.2022, "Cnk+": 0.00070355, "Cnr+": 0.00028725,
                    "Crk-": 0.00070154, "Cnk-": -0.00047611, "Cnr-": 0.00011049},
    "MG5_aMC+P8":  {"Ckk": 0.31431, "Cnn": 0.31687, "Crr": 0.04305,
                    "Crk+": -0.20528, "Cnk+": 0.0011332, "Cnr+": -0.00030782,
                    "Crk-": -0.0010084, "Cnk-": 0.0020196, "Cnr-": 0.0010337},
    "MiNNLO+P8":   {"Ckk": 0.31629, "Cnn": 0.31108, "Crr": 0.05273,
                    "Crk+": -0.1985, "Cnk+": -0.0012651, "Cnr+": 0.00029282,
                    "Crk-": -0.0010541, "Cnk-": 0.00027111, "Cnr-": -0.0010673},
}


def build_C_matrix(ckk, cnn, crr, cnr_plus, crk_plus, cnk_plus,
                   cnr_minus, crk_minus, cnk_minus):
    """
    Build the 3x3 C matrix in the helicity basis {k, n, r} from the
    measured coefficients.  CMS publishes symmetric (C^+) and antisymmetric
    (C^-) combinations; we recover the full C by:
        C_{ij} = C^+_{ij} + C^-_{ij}
        C_{ji} = C^+_{ij} - C^-_{ij}

    Index order: row/col = (k, n, r) = (0, 1, 2).
    """
    C = np.zeros((3, 3))
    # Diagonal
    C[0, 0] = ckk
    C[1, 1] = cnn
    C[2, 2] = crr
    # Off-diagonal: nr pair (indices n=1, r=2)
    C[1, 2] = cnr_plus + cnr_minus
    C[2, 1] = cnr_plus - cnr_minus
    # rk pair (indices r=2, k=0)
    C[2, 0] = crk_plus + crk_minus
    C[0, 2] = crk_plus - crk_minus
    # nk pair (indices n=1, k=0)
    C[1, 0] = cnk_plus + cnk_minus
    C[0, 1] = cnk_plus - cnk_minus
    return C


def compute_EN_from_coeffs(P1, P2, C):
    """Build rho (ATLAS sign convention) and compute log-negativity."""
    rho = two_qubit_rho_from_correlation(P1, P2, C, sign_convention="atlas")
    # Check positivity — measured matrix may violate positivity due to
    # statistical fluctuations, but should be close.
    eigs = np.linalg.eigvalsh(rho)
    return log_negativity(rho, dims=(2, 2)), eigs.min()


def monte_carlo_EN(n_samples=10000, seed=42):
    """
    Propagate stat+syst uncertainties to E_N by Monte Carlo sampling.

    Each coefficient is sampled from a Gaussian with width = sqrt(stat^2 + syst^2).
    This ignores correlations between coefficients (we don't have the full
    covariance matrix here; to do this properly we'd use HEPData table t2).
    The resulting uncertainty is therefore an approximation that treats stat
    and syst uncertainties as independent across coefficients.
    """
    rng = np.random.default_rng(seed)

    def total(key):
        return np.sqrt(stat[key]**2 + syst[key]**2)

    EN_samples = []
    D_samples = []
    min_eig_samples = []

    for _ in range(n_samples):
        P1s = np.array([
            rng.normal(P1[0], total("P1k")),
            rng.normal(P1[1], total("P1n")),
            rng.normal(P1[2], total("P1r")),
        ])
        P2s = np.array([
            rng.normal(P2[0], total("P2k")),
            rng.normal(P2[1], total("P2n")),
            rng.normal(P2[2], total("P2r")),
        ])
        C = build_C_matrix(
            rng.normal(C_kk, total("Ckk")),
            rng.normal(C_nn, total("Cnn")),
            rng.normal(C_rr, total("Crr")),
            rng.normal(C_nr_plus,  total("Cnr+")),
            rng.normal(C_rk_plus,  total("Crk+")),
            rng.normal(C_nk_plus,  total("Cnk+")),
            rng.normal(C_nr_minus, total("Cnr-")),
            rng.normal(C_rk_minus, total("Crk-")),
            rng.normal(C_nk_minus, total("Cnk-")),
        )
        EN, min_eig = compute_EN_from_coeffs(P1s, P2s, C)
        EN_samples.append(EN)
        D_samples.append(entanglement_marker_D(C))
        min_eig_samples.append(min_eig)

    return np.array(EN_samples), np.array(D_samples), np.array(min_eig_samples)


def analyze_prediction(name, coeffs):
    """Compute E_N from a theoretical SM prediction (zero polarization at LO)."""
    P_zero = np.zeros(3)
    C = build_C_matrix(
        coeffs["Ckk"], coeffs["Cnn"], coeffs["Crr"],
        coeffs["Cnr+"], coeffs["Crk+"], coeffs["Cnk+"],
        coeffs["Cnr-"], coeffs["Crk-"], coeffs["Cnk-"],
    )
    EN, min_eig = compute_EN_from_coeffs(P_zero, P_zero, C)
    D = entanglement_marker_D(C)
    return EN, D, min_eig


def main():
    print("=" * 72)
    print("CMS measured tt-bar log-negativity from full 15-coefficient matrix")
    print("HEPData ins2829523, arXiv:2409.11067, Phys. Rev. D 110 (2024) 112016")
    print("=" * 72)
    print()

    # ---- Central-value analysis ----
    C_central = build_C_matrix(
        C_kk, C_nn, C_rr,
        C_nr_plus, C_rk_plus, C_nk_plus,
        C_nr_minus, C_rk_minus, C_nk_minus,
    )
    print("Measured C matrix (helicity basis {k, n, r}):")
    for i, row_label in enumerate(["k", "n", "r"]):
        row = "  ".join(f"{C_central[i, j]:+.4f}" for j in range(3))
        print(f"  {row_label}   {row}")
    print()

    print(f"Measured polarizations: P_top = {P1}, P_antitop = {P2}")
    print()

    D_central = entanglement_marker_D(C_central)
    EN_central, min_eig = compute_EN_from_coeffs(P1, P2, C_central)
    print(f"D = -Tr(C)/3              = {D_central:+.4f}")
    print(f"Entanglement threshold    : D < -1/3 implies entanglement")
    print(f"Entangled at central value? {D_central < -1/3}")
    print()
    print(f"Minimum eigenvalue of rho : {min_eig:+.6f}")
    print(f"(Slight negativity is expected from statistical fluctuations.)")
    print()
    print(f"E_N(t : tbar) central     = {EN_central:.4f} bits")
    print()

    # ---- Monte Carlo uncertainty propagation ----
    print("Running Monte Carlo uncertainty propagation (10000 samples)...")
    EN_samples, D_samples, min_eig_samples = monte_carlo_EN(n_samples=10000)

    EN_median = np.median(EN_samples)
    EN_low  = np.percentile(EN_samples, 16)
    EN_high = np.percentile(EN_samples, 84)
    EN_mean = np.mean(EN_samples)
    EN_std  = np.std(EN_samples)

    D_median = np.median(D_samples)
    D_low, D_high = np.percentile(D_samples, [16, 84])

    # Count fraction of samples with E_N > 0 (significant entanglement)
    frac_entangled = np.mean(EN_samples > 1e-6)

    print()
    print("=" * 72)
    print("UNCERTAINTY PROPAGATION RESULTS (stat+syst, 68% CL)")
    print("=" * 72)
    print(f"D            = {D_central:+.4f}  "
          f"({D_low:+.4f} to {D_high:+.4f})")
    print(f"E_N(t:tbar)  = {EN_central:.4f} bits  "
          f"({EN_low:.4f} to {EN_high:.4f})")
    print(f"Mean ± std   = {EN_mean:.4f} ± {EN_std:.4f} bits")
    print(f"Fraction of MC samples with E_N > 0: {frac_entangled:.3f}")
    print()
    print("Note: uncorrelated-Gaussian approximation to uncertainties. The full")
    print("covariance matrix (HEPData table t2) would give a more accurate interval.")
    print()

    # ---- Comparison against SM predictions ----
    print("=" * 72)
    print("COMPARISON AGAINST SM PREDICTIONS (zero polarization at LO)")
    print("=" * 72)
    print(f"{'Generator':<15} {'D':>8} {'E_N (bits)':>12} {'min eig':>10}")
    print("-" * 50)
    for name, coeffs in SM_PREDICTIONS.items():
        EN, D, min_e = analyze_prediction(name, coeffs)
        print(f"{name:<15} {D:+.4f}   {EN:>8.4f}       {min_e:+.4f}")
    print()
    print(f"{'Measured':<15} {D_central:+.4f}   {EN_central:>8.4f}       {min_eig:+.4f}")
    print()


if __name__ == "__main__":
    main()
