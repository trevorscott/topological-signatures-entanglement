"""
Generate the final paper figure: E_N, D, and D-tilde across m(tt-bar)
bins, from the measured CMS data with covariance-propagated uncertainties.

This replaces the original fig_en_vs_mtt.png with a 3-panel figure that
tells the complete story of how the quantum structure of the tt-bar state
transitions across the kinematic phase space.

Physical narrative:
    - Threshold (300-400 GeV): near-singlet, strong entanglement by D (<-1/3)
    - Intermediate (400-600 GeV): transition region, separable
    - 600-800 GeV: D-tilde crosses zero, moving toward triplet structure
    - >800 GeV: triplet-like, D-tilde positive but below +1/3 in this
      inclusive |cos theta| selection
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from rwf_pipeline import two_qubit_rho_from_correlation, log_negativity, entanglement_marker_D
from parse_cms_differential import parse_measurements, parse_covariance, build_covariance_matrix, COEFF_ORDER
from cms_differential_analysis import build_C_matrix, compute_EN, bin_midpoint, bin_range


def D_tilde_from_C(C):
    # Index: k=0, n=1, r=2
    return (C[1, 1] - C[2, 2] - C[0, 0]) / 3.0


def monte_carlo_observables(bin_data, cov_matrix=None, n_samples=20000, seed=42):
    """MC on all three observables: E_N, D, D-tilde. Rejects unphysical samples."""
    rng = np.random.default_rng(seed)

    central = np.array([bin_data[c]["value"] for c in COEFF_ORDER])
    totals = np.array([bin_data[c]["total"] for c in COEFF_ORDER])

    if cov_matrix is not None:
        eigs_cov = np.linalg.eigvalsh(cov_matrix)
        if eigs_cov.min() < -1e-10:
            cov_matrix = cov_matrix + (abs(eigs_cov.min()) + 1e-8) * np.eye(len(central))
        samples_vec = rng.multivariate_normal(central, cov_matrix, size=n_samples)
    else:
        samples_vec = rng.normal(
            loc=central[None, :],
            scale=totals[None, :],
            size=(n_samples, len(central)),
        )

    EN, D, Dt, phys = [], [], [], []
    for s in range(n_samples):
        v = {c: samples_vec[s, i] for i, c in enumerate(COEFF_ORDER)}
        P1 = np.array([v["P1k"], v["P1n"], v["P1r"]])
        P2 = np.array([v["P2k"], v["P2n"], v["P2r"]])
        C = build_C_matrix(v["Ckk"], v["Cnn"], v["Crr"],
                           v["Cnr+"], v["Crk+"], v["Cnk+"],
                           v["Cnr-"], v["Crk-"], v["Cnk-"])
        rho = two_qubit_rho_from_correlation(P1, P2, C, sign_convention="atlas")
        rho_eigs = np.linalg.eigvalsh(rho)
        if rho_eigs.min() < -1e-6:
            phys.append(False)
            # Project to physical region: clip eigenvalues and renormalize
            eigs_clip = np.maximum(rho_eigs, 0)
            eigs_clip /= eigs_clip.sum()
            # Reconstruct rho from its eigendecomposition
            _, vecs = np.linalg.eigh(rho)
            rho_phys = vecs @ np.diag(eigs_clip) @ vecs.conj().T
            EN.append(log_negativity(rho_phys, (2, 2)))
        else:
            phys.append(True)
            EN.append(log_negativity(rho, (2, 2)))
        D.append(entanglement_marker_D(C))
        Dt.append(D_tilde_from_C(C))

    return np.array(EN), np.array(D), np.array(Dt), np.array(phys)


def summarize(arr):
    return {
        "central": None,  # will be filled separately from central values
        "median": float(np.median(arr)),
        "low":  float(np.percentile(arr, 16)),
        "high": float(np.percentile(arr, 84)),
    }


def central_observables(bin_data):
    v = {c: bin_data[c]["value"] for c in COEFF_ORDER}
    P1 = np.array([v["P1k"], v["P1n"], v["P1r"]])
    P2 = np.array([v["P2k"], v["P2n"], v["P2r"]])
    C = build_C_matrix(v["Ckk"], v["Cnn"], v["Crr"],
                       v["Cnr+"], v["Crk+"], v["Cnk+"],
                       v["Cnr-"], v["Crk-"], v["Cnk-"])
    rho = two_qubit_rho_from_correlation(P1, P2, C, sign_convention="atlas")
    return {
        "EN": log_negativity(rho, (2, 2)),
        "D": entanglement_marker_D(C),
        "Dt": D_tilde_from_C(C),
    }


def main():
    meas_file = "data/cms_full_matrix_differential_mtt.csv"
    cov_file = "data/cms_full_matrix_differential_mtt_covariance.csv"

    bins, bin_order = parse_measurements(meas_file)
    cov_dict = parse_covariance(cov_file) if os.path.exists(cov_file) else None

    rows = []
    for b in bin_order:
        if len(bins[b]) < 16:
            continue
        cov_M = build_covariance_matrix(cov_dict, b) if cov_dict else None
        EN_arr, D_arr, Dt_arr, phys_arr = monte_carlo_observables(bins[b], cov_M)
        central = central_observables(bins[b])
        rows.append({
            "bin": b,
            "mid": bin_midpoint(b),
            "lo": bin_range(b)[0],
            "hi": bin_range(b)[1],
            "EN": {"c": central["EN"], **summarize(EN_arr)},
            "D":  {"c": central["D"],  **summarize(D_arr)},
            "Dt": {"c": central["Dt"], **summarize(Dt_arr)},
            "frac_phys": float(np.mean(phys_arr)),
        })
        print(f"{b:<30}  E_N={central['EN']:+.4f}  "
              f"D={central['D']:+.4f}  D-tilde={central['Dt']:+.4f}  "
              f"(phys frac: {np.mean(phys_arr):.2f})")

    # ---- Figure: 3-panel ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    xs      = [r["mid"] for r in rows]
    xerr_lo = [r["mid"] - r["lo"] for r in rows]
    xerr_hi = [r["hi"] - r["mid"] for r in rows]

    # Panel 1: E_N
    ax = axes[0]
    ys  = [r["EN"]["c"] for r in rows]
    elo = [max(0, r["EN"]["c"] - r["EN"]["low"])  for r in rows]
    ehi = [max(0, r["EN"]["high"] - r["EN"]["c"]) for r in rows]
    ax.errorbar(xs, ys, xerr=[xerr_lo, xerr_hi], yerr=[elo, ehi],
                fmt="o", color="#2c3e50", capsize=4, markersize=8, lw=1.2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$m(t\bar{t})$ [GeV]")
    ax.set_ylabel(r"$E_N(t:\bar{t})$ [bits]")
    ax.set_title(r"Log-negativity")
    ax.grid(alpha=0.3)
    ax.set_xlim(250, 1250)

    # Panel 2: D
    ax = axes[1]
    ys  = [r["D"]["c"] for r in rows]
    elo = [r["D"]["c"] - r["D"]["low"]  for r in rows]
    ehi = [r["D"]["high"] - r["D"]["c"] for r in rows]
    ax.errorbar(xs, ys, xerr=[xerr_lo, xerr_hi], yerr=[elo, ehi],
                fmt="s", color="#c0392b", capsize=4, markersize=8, lw=1.2,
                label="CMS measurement")
    ax.axhline(-1/3, color="green", lw=1.5, ls="--",
               label=r"Threshold: $D = -1/3$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$m(t\bar{t})$ [GeV]")
    ax.set_ylabel(r"$D = -\mathrm{Tr}(C)/3$")
    ax.set_title(r"Singlet witness")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(250, 1250)

    # Panel 3: D-tilde
    ax = axes[2]
    ys  = [r["Dt"]["c"] for r in rows]
    elo = [r["Dt"]["c"] - r["Dt"]["low"]  for r in rows]
    ehi = [r["Dt"]["high"] - r["Dt"]["c"] for r in rows]
    ax.errorbar(xs, ys, xerr=[xerr_lo, xerr_hi], yerr=[elo, ehi],
                fmt="^", color="#16a085", capsize=4, markersize=8, lw=1.2,
                label="CMS measurement")
    ax.axhline(1/3, color="purple", lw=1.5, ls="--",
               label=r"Threshold: $\tilde{D} = +1/3$")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"$m(t\bar{t})$ [GeV]")
    ax.set_ylabel(r"$\tilde{D} = (C_{nn} - C_{kk} - C_{rr})/3$")
    ax.set_title(r"Triplet witness (boosted regime)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(250, 1250)

    fig.suptitle(r"Quantum entanglement of the measured $t\bar{t}$ state across "
                 r"$m(t\bar{t})$ bins, CMS 138 fb$^{-1}$, $\sqrt{s} = 13$ TeV",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig_ttbar_3panel.png", dpi=140, bbox_inches="tight")
    print("\nSaved figures/fig_ttbar_3panel.png")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/cms_three_observables.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved results/cms_three_observables.json")


if __name__ == "__main__":
    main()
