"""
Parser for the CMS differential m(tt) spin correlation measurement.

Input: the CSV file from HEPData ins2829523, table t9:
    `cms_full_matrix_differential_mtt.csv`

And optionally the covariance:
    `cms_full_matrix_differential_mtt_covariance.csv`

Output: a dict keyed by bin label with each bin's 15 coefficients + uncertainties.

The CSV has a header with metadata (lines starting with #:), then a single data
block where each row is
    "<Coefficient name> for <bin label>", value, stat+, stat-, syst+, syst-

and the bin labels are things like "300 < m(t t-bar) < 400  GeV".
"""
import csv
import re
from collections import defaultdict


# Canonical short names for the 15 coefficients (in the order they appear in
# CMS's tables, per the inclusive file we already have).
COEFF_ORDER = [
    "c",
    "P1r", "P1n", "P1k",
    "P2r", "P2n", "P2k",
    "Crr", "Cnn", "Ckk",
    "Cnr+", "Crk+", "Cnk+",
    "Cnr-", "Crk-", "Cnk-",
]


# Regex patterns to parse "<coeff name> for <bin label>"
# The coefficient labels in the CSV have LaTeX-style formatting:
#   $c$, $P_{r}$, $P_{n}$, $P_{k}$
#   $\bar{P}_{r}$, $\bar{P}_{n}$, $\bar{P}_{k}$
#   $C_{rr}$, $C_{nn}$, $C_{kk}$
#   $C_{nr}^{+}$, $C_{rk}^{+}$, $C_{nk}^{+}$
#   $C_{nr}^{-}$, $C_{rk}^{-}$, $C_{nk}^{-}$
COEFF_PATTERNS = {
    r"\$c\$":              "c",
    r"\$P_\{r\}\$":        "P1r",
    r"\$P_\{n\}\$":        "P1n",
    r"\$P_\{k\}\$":        "P1k",
    r"\$\\bar\{P\}_\{r\}\$": "P2r",
    r"\$\\bar\{P\}_\{n\}\$": "P2n",
    r"\$\\bar\{P\}_\{k\}\$": "P2k",
    r"\$C_\{rr\}\$":       "Crr",
    r"\$C_\{nn\}\$":       "Cnn",
    r"\$C_\{kk\}\$":       "Ckk",
    r"\$C_\{nr\}\^\{\+\}\$": "Cnr+",
    r"\$C_\{rk\}\^\{\+\}\$": "Crk+",
    r"\$C_\{nk\}\^\{\+\}\$": "Cnk+",
    r"\$C_\{nr\}\^\{-\}\$":  "Cnr-",
    r"\$C_\{rk\}\^\{-\}\$":  "Crk-",
    r"\$C_\{nk\}\^\{-\}\$":  "Cnk-",
}


def parse_label(s):
    """
    Given a label like "$C_{kk}$ for $300 < m(t\\bar{t}) < 400 $ GeV",
    return (short_coeff_name, bin_label).
    """
    # Find which coefficient this is
    coeff = None
    for pat, name in COEFF_PATTERNS.items():
        if re.match(pat, s):
            coeff = name
            break
    if coeff is None:
        return None, None

    # Extract the bin label: everything after " for "
    bin_match = re.search(r" for (.*)", s)
    if not bin_match:
        return coeff, None

    bin_str = bin_match.group(1).strip()
    # Normalize the bin label by stripping LaTeX
    bin_str = bin_str.replace("$", "").replace(r"\bar{t}", "tbar").replace("\\", "")
    bin_str = re.sub(r"\s+", " ", bin_str).strip()
    return coeff, bin_str


def parse_measurements(csv_path):
    """
    Parse the measured-values CSV (HEPData table t9).

    Returns
    -------
    bins : dict mapping bin_label -> dict of {coeff_name: {value, stat, syst}}
    bin_order : list of bin labels in the order they appear in the file
    """
    bins = defaultdict(dict)
    bin_order = []

    with open(csv_path, "r") as f:
        # Skip metadata lines (start with #: or are blank)
        data_lines = []
        for line in f:
            if line.startswith("#:"):
                continue
            if not line.strip():
                continue
            data_lines.append(line.rstrip("\n"))

    # Now we have alternating header + data blocks. Find the first header:
    # "Coefficient name,Measured coefficient value,stat +,stat -,syst +,syst -"
    reader = csv.reader(data_lines)
    header = None
    for row in reader:
        if not row:
            continue
        if row[0].strip() == "Coefficient name":
            header = row
            # Check if this is the "measured" block (has stat/syst columns)
            if "Measured coefficient value" in row[1]:
                # Start of measurement block
                break
            else:
                # Theory prediction block — stop, we only want measurements
                break

    # Continue parsing until we hit another header row
    for row in reader:
        if not row:
            continue
        if row[0].strip() == "Coefficient name":
            # We've hit the start of a theory-prediction block; stop.
            break

        label = row[0].strip()
        coeff, bin_label = parse_label(label)
        if coeff is None or bin_label is None:
            continue

        try:
            value = float(row[1])
            stat_plus  = float(row[2])
            syst_plus  = float(row[4])
        except (ValueError, IndexError):
            continue

        if bin_label not in bins:
            bin_order.append(bin_label)

        bins[bin_label][coeff] = {
            "value": value,
            "stat": abs(stat_plus),
            "syst": abs(syst_plus),
            "total": (stat_plus**2 + syst_plus**2)**0.5,
        }

    return dict(bins), bin_order


def parse_covariance(csv_path):
    """
    Parse the covariance matrix CSV (HEPData table t10).

    Returns a dict mapping (coeff_i, bin_i, coeff_j, bin_j) -> covariance value.
    The covariance is symmetric so we store both (i,j) and (j,i).

    For downstream use, most callers will want the full 64x64 matrix (16 coeffs
    x 4 bins), which we construct on demand.
    """
    cov = {}

    with open(csv_path, "r") as f:
        data_lines = [line.rstrip("\n") for line in f
                      if not line.startswith("#:") and line.strip()]

    reader = csv.reader(data_lines)
    # Skip header
    header_found = False
    for row in reader:
        if not row:
            continue
        if not header_found and "x-axis" in row[0]:
            header_found = True
            continue
        if not header_found:
            continue

        if len(row) < 3:
            continue

        row_label = row[0].strip()
        col_label = row[1].strip()
        try:
            val = float(row[2])
        except ValueError:
            continue

        coeff_i, bin_i = parse_label(row_label)
        coeff_j, bin_j = parse_label(col_label)
        if coeff_i is None or coeff_j is None:
            continue

        key = (coeff_i, bin_i, coeff_j, bin_j)
        cov[key] = val

    return cov


def build_covariance_matrix(cov_dict, bin_label):
    """
    Extract the 16x16 covariance matrix for a single m(tt) bin.

    Returns:
        cov : (16, 16) numpy array ordered by COEFF_ORDER
    """
    import numpy as np

    n = len(COEFF_ORDER)
    M = np.zeros((n, n))
    for i, ci in enumerate(COEFF_ORDER):
        for j, cj in enumerate(COEFF_ORDER):
            key = (ci, bin_label, cj, bin_label)
            if key in cov_dict:
                M[i, j] = cov_dict[key]
            else:
                # Try the symmetric entry
                key_sym = (cj, bin_label, ci, bin_label)
                if key_sym in cov_dict:
                    M[i, j] = cov_dict[key_sym]
    # Symmetrize in case only one triangle is stored
    M = 0.5 * (M + M.T)
    return M


if __name__ == "__main__":
    import sys

    meas_file = sys.argv[1] if len(sys.argv) > 1 else "data/cms_full_matrix_differential_mtt.csv"
    cov_file  = sys.argv[2] if len(sys.argv) > 2 else "data/cms_full_matrix_differential_mtt_covariance.csv"

    print(f"Loading measurements from: {meas_file}")
    bins, bin_order = parse_measurements(meas_file)
    print(f"Parsed {len(bins)} bins:")
    for b in bin_order:
        print(f"  {b}  ({len(bins[b])} coefficients)")

    # Show the structure of the first bin
    first = bin_order[0]
    print(f"\n--- {first} ---")
    for coeff in COEFF_ORDER:
        if coeff in bins[first]:
            d = bins[first][coeff]
            print(f"  {coeff:6s} = {d['value']:+.5f}  ± {d['total']:.5f}  "
                  f"(stat {d['stat']:.5f}, syst {d['syst']:.5f})")

    # Load covariance if available
    try:
        print(f"\nLoading covariance from: {cov_file}")
        cov = parse_covariance(cov_file)
        print(f"Parsed {len(cov)} covariance entries")

        import numpy as np
        M = build_covariance_matrix(cov, first)
        print(f"\nCovariance matrix for bin {first}: shape {M.shape}")
        print(f"Diagonal (variances): {np.diag(M)}")
    except FileNotFoundError:
        print("Covariance file not found — uncorrelated uncertainties will be used.")
