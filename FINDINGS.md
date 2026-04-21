# Development Log: Empirical Findings & Pipeline Evolution

_Updated: April 2026_

This document tracks the evolution of the data pipeline, detailing how the processing architecture adapted to the realities of the CMS differential measurement data (HEPData ins2829523). It serves as a technical log of the assumptions made, the data realities encountered, and the architectural solutions implemented.

---

## TL;DR

1. **Initial Assumption Failure:** The early `atlas_ttbar_signature.py` script yielded $E_N = 0.385$ bits using an isotropic $C$-matrix ansatz. This failed when confronted with the highly anisotropic structure of the real CMS data.
2. **The "Inclusive" Trap:** The inclusive $t\bar{t}$ dataset is an average over vastly different kinematic regimes, resulting in a blended state that washes out entanglement signatures ($D = -0.221$).
3. **Differential Resolution:** Slicing the data differentially is mathematically required to extract the signal. At the threshold bin (300 < $m_{t\bar{t}}$ < 400 GeV), the pipeline successfully reconstructs $E_N(t:\bar{t}) = 0.182$ bits [0.154, 0.316] at 68% CL.
4. **Architectural Pivot:** The pipeline was overhauled from a static calculator into a Monte Carlo engine capable of propagating $16 \times 16$ covariance matrices and applying Positive-Semidefinite (PSD) projections to handle the noise inherent in collider measurements.

---

## The Engineering Journey

### Phase 1: The Isotropic Model
Early iterations of the pipeline attempted to construct the $\rho_{t\bar{t}}$ density matrix from a single abstract scalar ($D = -0.537$) using an isotropic ansatz: `C = diag(0.537, 0.537, 0.537)`. This produced a clean Werner state.

**The Reality Check:** The real CMS data at the kinematic threshold is heavily anisotropic:
* $C_{kk} \approx 0.43$, $C_{nn} \approx 0.54$, $C_{rr} \approx 0.27$
* Large off-diagonal elements ($C_{rk} \approx +0.04$) indicate complex parity-conserving structures.

Feeding the naive isotropic model real data broke the assumptions. The pipeline had to be rewritten to accept the full 15-coefficient $4 \times 4$ reconstruction. 

### Phase 2: Handling Unphysical Tomography
Unconstrained quantum state tomography—especially when data is subjected to detector cuts and unfolding—frequently yields coefficient matrices that do not correspond to physically possible states (e.g., density matrices with negative eigenvalues).

In the most critical data bins, the pipeline encountered positivity violation rates between 40% and 54%. 

**The Solution:** The pipeline was upgraded to implement a positivity-preserving projection. Unphysical matrices are projected onto the nearest Positive-Semidefinite (PSD) cone by clipping negative eigenvalues at zero and renormalizing. This preserves Hermiticity and unit trace without artificially inflating the Logarithmic Negativity metric.

### Phase 3: Uncertainty Propagation
Collider data does not exist as isolated central values; the 15 angular coefficients are highly correlated. 

**The Solution:** Rather than computing a single static $E_N$ value, the core pipeline (`cms_differential_analysis.py`) was rebuilt around a Monte Carlo engine. It draws 20,000 samples per kinematic bin from a multivariate Gaussian using the published $16 \times 16$ intra-bin covariance matrices. This allows the pipeline to output rigorous 68% confidence intervals rather than brittle point estimates.

---

## Final Pipeline Output

Processing the CMS 138 fb⁻¹ lepton+jets channel through the finalized architecture yields the following empirical topology:

| $m_{t\bar{t}}$ range (GeV) | $D$ Witness | $E_N$ (bits, 68% CL) | Pipeline State | Physical Interpretation |
|---|---|---|---|---|
| 300–400 (threshold) | −0.412 ± 0.08 | **0.18 [0.15, 0.32]** | Validated | Near-singlet, highly entangled |
| 400–600 (intermediate) | −0.204 ± 0.03 | 0.00 | Validated | $C_{rr}$ sign flip, separability |
| 600–800 | −0.089 ± 0.02 | 0.00 | Validated | Triplet-like mixture |
| > 800 (boosted) | −0.005 ± 0.02 | 0.00 | Validated | Requires specialized $|cos~\theta|$ cuts |

*(Note: Uncertainties reflect full covariance matrix propagation from HEPData table t10).*

## Next Steps for the Codebase

* **High-Mass Boosted Regime:** Implement the specialized $|cos~\theta| < 0.4$ selection filters (HEPData t11/t12) to capture the $>5\sigma$ boosted entanglement observation.
* **Alternative Projections:** Introduce Maximum Likelihood Estimation (MLE) or Frobenius norm projections as alternatives to eigenvalue clipping to allow users to test reconstruction sensitivity.