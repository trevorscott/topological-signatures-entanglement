"""
CMS differential tt-bar log-negativity analysis.

This is the main new analysis: compute the bipartite log-negativity
E_N(t:tbar) in each kinematic bin of m(tt-bar), using the full 15-coefficient
spin density matrix from CMS's differential measurement, with proper
covariance-matrix uncertainty propagation.

Input files (both from HEPData ins2829523, CMS 2024, arXiv:2409.11067):
    data/cms_full_matrix_differential_mtt.csv           — table t9 (values)
    data/cms_full_matrix_differential_mtt_covariance.csv — table t10 (covariance)

Outputs:
    figures/fig_en_vs_mtt.png     — E_N vs m(tt) with error bars
    figures/fig_entanglement_significance.png — significance map
    Terminal: table of E_N per bin with 68% intervals and entanglement significance

Physical interpretation:
    - Low-mass (threshold) bin: near-singlet state, strong entanglement
    - Intermediate-mass bin: SM predicts triplet-like mixture with D close to
      or above the -1/3 threshold — possibly no entanglement
    - High-mass bin: triplet state, entanglement returns (via the sign structure
      of C_kk, C_nn, C_rr which flips at high m(tt))

The topological signature of the RWF framework applied to this bipartite
system reduces to a single persistent H_0 feature per bin (with birth = E_N,
death = 0), so the information content is fully captured by the E_N(m_tt) curve.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from rwf_pipeline import (
    two_qubit_rho_from_correlation,
    log_negativity,
    entanglement_marker_D,
)
from parse_cms_differential import (
    parse_measurements,
    parse_covariance,
    build_covariance_matrix,
    COEFF_ORDER,
)

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 110


def build_C_matrix(ckk, cnn, crr, cnr_plus, crk_plus, cnk_plus,
                   cnr_minus, crk_minus, cnk_minus):
    """3x3 C matrix from the 9 measured correlation coefficients."""
    C = np.zeros((3, 3))
    # Index order: (k, n, r) = (0, 1, 2)
    C[0, 0] = ckk
    C[1, 1] = cnn
    C[2, 2] = crr
    # nr pair
    C[1, 2] = cnr_plus + cnr_minus
    C[2, 1] = cnr_plus - cnr_minus
    # rk pair
    C[2, 0] = crk_plus + crk_minus
    C[0, 2] = crk_plus - crk_minus
    # nk pair
    C[1, 0] = cnk_plus + cnk_minus
    C[0, 1] = cnk_plus - cnk_minus
    return C


def compute_EN(values):
    """Given a dict of coefficient values, build rho and return E_N and D."""
    P1 = np.array([values["P1k"], values["P1n"], values["P1r"]])
    P2 = np.array([values["P2k"], values["P2n"], values["P2r"]])
    C = build_C_matrix(
        values["Ckk"], values["Cnn"], values["Crr"],
        values["Cnr+"], values["Crk+"], values["Cnk+"],
        values["Cnr-"], values["Crk-"], values["Cnk-"],
    )
    rho = two_qubit_rho_from_correlation(P1, P2, C, sign_convention="atlas")
    eigs = np.linalg.eigvalsh(rho)
    EN = log_negativity(rho, dims=(2, 2))
    D = entanglement_marker_D(C)
    return EN, D, eigs.min()


def monte_carlo_bin(bin_data, cov_matrix=None, n_samples=20000, seed=42):
    """
    Propagate uncertainties for a single bin using either:
    - Full covariance matrix (if cov_matrix is provided), via multivariate
      normal sampling
    - Uncorrelated Gaussians (fallback)

    Returns:
        dict with EN_central, EN_median, EN_16, EN_84, D_central, etc.,
        plus the full array of E_N samples.
    """
    rng = np.random.default_rng(seed)

    # Central values in the order of COEFF_ORDER
    central = np.array([bin_data[c]["value"] for c in COEFF_ORDER])
    totals = np.array([bin_data[c]["total"] for c in COEFF_ORDER])

    if cov_matrix is not None:
        # Use full covariance
        # Check it's positive-semidefinite; if not, regularize
        eigs_cov = np.linalg.eigvalsh(cov_matrix)
        if eigs_cov.min() < -1e-10:
            # Add small ridge to make it PSD (should be unnecessary in practice)
            cov_matrix = cov_matrix + (abs(eigs_cov.min()) + 1e-8) * np.eye(len(central))
        samples_vec = rng.multivariate_normal(central, cov_matrix, size=n_samples)
    else:
        # Uncorrelated fallback
        samples_vec = rng.normal(
            loc=central[None, :],
            scale=totals[None, :],
            size=(n_samples, len(central)),
        )

    EN_samples = np.zeros(n_samples)
    D_samples = np.zeros(n_samples)
    positivity_viol = 0

    for s in range(n_samples):
        v = {c: samples_vec[s, i] for i, c in enumerate(COEFF_ORDER)}
        EN, D, min_eig = compute_EN(v)
        EN_samples[s] = EN
        D_samples[s] = D
        if min_eig < -1e-6:
            positivity_viol += 1

    # Central (from central values, not MC mean)
    v_central = {c: bin_data[c]["value"] for c in COEFF_ORDER}
    EN_central, D_central, min_eig_central = compute_EN(v_central)

    return {
        "EN_central": EN_central,
        "EN_median": np.median(EN_samples),
        "EN_16": np.percentile(EN_samples, 16),
        "EN_84": np.percentile(EN_samples, 84),
        "EN_mean": np.mean(EN_samples),
        "EN_std": np.std(EN_samples),
        "EN_samples": EN_samples,
        "D_central": D_central,
        "D_16": np.percentile(D_samples, 16),
        "D_84": np.percentile(D_samples, 84),
        "min_eig_central": min_eig_central,
        "frac_entangled": np.mean(EN_samples > 1e-6),
        "frac_below_threshold": np.mean(D_samples < -1/3),  # D < -1/3 entanglement criterion
        "positivity_violations": positivity_viol,
        "n_samples": n_samples,
    }


def bin_midpoint(bin_label):
    """Extract a representative m(tt) value for plotting. Uses bin midpoint,
    or for 'm(ttbar) > 800 GeV', uses 1000 GeV as representative."""
    import re
    # Try "NNN < m(ttbar) < MMM GeV"
    m = re.search(r"(\d+)\s*<\s*m\(ttbar\)\s*<\s*(\d+)", bin_label)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.search(r"m\(ttbar\)\s*>\s*(\d+)", bin_label)
    if m:
        # For open-ended high-mass bin, use 1000 GeV as representative
        return int(m.group(1)) + 200
    return None


def bin_range(bin_label):
    """Return (lo, hi) for the m(tt) range, for x-axis error bars."""
    import re
    m = re.search(r"(\d+)\s*<\s*m\(ttbar\)\s*<\s*(\d+)", bin_label)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"m\(ttbar\)\s*>\s*(\d+)", bin_label)
    if m:
        return (int(m.group(1)), int(m.group(1)) + 400)
    return (0, 0)


def main():
    # --- Locate data files ---
    data_dir = "data"
    meas_file = os.path.join(data_dir, "cms_full_matrix_differential_mtt.csv")
    cov_file  = os.path.join(data_dir, "cms_full_matrix_differential_mtt_covariance.csv")

    if not os.path.exists(meas_file):
        print(f"ERROR: Cannot find {meas_file}")
        print("Please pull from HEPData first:")
        print(f"  cd data/")
        print("  curl -L -o cms_full_matrix_differential_mtt.csv \\")
        print("    \"https://www.hepdata.net/download/table/ins2829523/Results%20for%20full%20matrix%20m(tt)%20differential/csv\"")
        sys.exit(1)

    bins, bin_order = parse_measurements(meas_file)
    print(f"Parsed {len(bin_order)} bins from {meas_file}")
    for b in bin_order:
        print(f"  {b}")
    print()

    # Try to load covariance
    use_cov = False
    cov_dict = None
    if os.path.exists(cov_file):
        print(f"Loading covariance from {cov_file}...")
        cov_dict = parse_covariance(cov_file)
        print(f"  {len(cov_dict)} covariance entries loaded")
        use_cov = True
    else:
        print(f"WARNING: no covariance file at {cov_file}; using uncorrelated approximation")
    print()

    # --- Per-bin analysis ---
    results = {}
    print("=" * 80)
    print(f"{'Bin':<30}  {'D':>8}  {'E_N (bits)':>18}  {'P(entangled)':>12}")
    print("=" * 80)
    for bin_label in bin_order:
        # Skip bins that don't have all 16 coefficients
        if len(bins[bin_label]) < 16:
            print(f"  {bin_label:<28}: only {len(bins[bin_label])} coeffs, skipping")
            continue

        cov_M = None
        if use_cov:
            cov_M = build_covariance_matrix(cov_dict, bin_label)

        r = monte_carlo_bin(bins[bin_label], cov_matrix=cov_M, n_samples=20000)
        results[bin_label] = r

        EN_str = f"{r['EN_central']:.4f} [{r['EN_16']:.4f}, {r['EN_84']:.4f}]"
        print(f"  {bin_label:<28}  {r['D_central']:+.4f}  {EN_str:>18}  "
              f"{r['frac_entangled']:.2f}")

    print("=" * 80)
    print()
    print("Notes:")
    print("  - D < -1/3 is the Peres-Horodecki entanglement criterion for tt-bar")
    print("  - E_N > 0 is the (stronger) log-negativity entanglement condition")
    print("  - P(entangled) = MC fraction of samples with E_N > 0")
    print()

    # --- Figures ---
    os.makedirs("figures", exist_ok=True)

    # Figure: E_N vs m(tt)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    xs, ys, yerr_low, yerr_high = [], [], [], []
    xerr_low, xerr_high = [], []
    d_ys, d_err_lo, d_err_hi = [], [], []
    for bin_label, r in results.items():
        mid = bin_midpoint(bin_label)
        lo, hi = bin_range(bin_label)
        if mid is None:
            continue
        xs.append(mid)
        xerr_low.append(mid - lo)
        xerr_high.append(hi - mid)
        ys.append(r["EN_central"])
        yerr_low.append(max(0, r["EN_central"] - r["EN_16"]))
        yerr_high.append(max(0, r["EN_84"] - r["EN_central"]))
        d_ys.append(r["D_central"])
        d_err_lo.append(r["D_central"] - r["D_16"])
        d_err_hi.append(r["D_84"] - r["D_central"])

    xs = np.array(xs)
    ys = np.array(ys)

    # Left panel: E_N
    ax1.errorbar(xs, ys,
                 xerr=[xerr_low, xerr_high],
                 yerr=[yerr_low, yerr_high],
                 fmt="o", color="#2c3e50", capsize=4, markersize=8,
                 label="CMS measurement (68% CL)")
    ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax1.set_xlabel(r"$m(t\bar{t})$ [GeV]")
    ax1.set_ylabel(r"$E_N(t:\bar{t})$ [bits]")
    ax1.set_title(r"Log-negativity across $m(t\bar{t})$ phase space")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(250, 1250)

    # Right panel: D
    ax2.errorbar(xs, d_ys,
                 xerr=[xerr_low, xerr_high],
                 yerr=[d_err_lo, d_err_hi],
                 fmt="s", color="#c0392b", capsize=4, markersize=8,
                 label="CMS measurement")
    ax2.axhline(-1/3, color="green", lw=1.5, ls="--",
                label=r"Entanglement threshold $D = -1/3$")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax2.set_xlabel(r"$m(t\bar{t})$ [GeV]")
    ax2.set_ylabel(r"$D = -\mathrm{Tr}(C)/3$")
    ax2.set_title(r"Peres-Horodecki observable $D$")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(250, 1250)

    plt.tight_layout()
    fig_path = "figures/fig_en_vs_mtt.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    # Save results for later use
    import json
    summary = {}
    for b, r in results.items():
        summary[b] = {
            "D_central": float(r["D_central"]),
            "D_16": float(r["D_16"]),
            "D_84": float(r["D_84"]),
            "EN_central": float(r["EN_central"]),
            "EN_16": float(r["EN_16"]),
            "EN_84": float(r["EN_84"]),
            "frac_entangled": float(r["frac_entangled"]),
            "frac_below_threshold": float(r["frac_below_threshold"]),
            "positivity_violations": int(r["positivity_violations"]),
            "n_samples": int(r["n_samples"]),
        }
    with open("results/cms_differential_results.json", "w") as f:
        os.makedirs("results", exist_ok=True)
        json.dump(summary, f, indent=2)
    print(f"Saved results/cms_differential_results.json")


if __name__ == "__main__":
    main()
