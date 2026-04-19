# Deprecated scripts

## `atlas_ttbar_signature.py` (removed)

An earlier version of this pipeline included `atlas_ttbar_signature.py`, which computed $E_N(t{:}\bar{t}) = 0.385$ bits from $D = -0.537$ (the ATLAS threshold scalar) under an isotropic assumption $C = \text{diag}(0.537, 0.537, 0.537)$.

**That number was wrong** — the measured $C$ matrix is strongly anisotropic. The isotropic state satisfying $D = -0.537$ is not the physical tt̄ state at threshold.

The correct value, computed from the full 16-coefficient CMS measurement with covariance-propagated uncertainties:

$$E_N(t{:}\bar{t}) = 0.18\,[0.13, 0.29]\ \text{bits at 68\% CL}$$

in the $300 < m_{t\bar{t}} < 400$ GeV bin.

See `cms_differential_analysis.py` and `plot_ttbar_3panel.py` for the correct analysis.
