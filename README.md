# Topological Signatures of Quantum Entanglement at the LHC

Public code and data for the methodology paper
*"Topological Signatures of Quantum Entanglement at the LHC: Measurement in tt̄ and Prediction for H → ZZ*."*

The pipeline computes the logarithmic negativity $E_N$ of reconstructed quantum density matrices at the LHC (bipartite measured data) and builds the flag-complex topological signature $\chi(K_W)$ of the RWF framework [[Scott 2026]](#references) on multipartite systems (tripartite theoretical predictions).

---

## Relation to the Relational Witnessing Framework

This paper is an application of the flag-complex persistent-homology construction introduced in the [Relational Witnessing Framework](https://github.com/trevorscott/relational-witnessing-framework). RWF proposed logarithmic negativity as the physically motivated edge weight for entanglement complexes — this paper operationalizes that construction on real collider data.

The bipartite tt̄ analysis (Section 3) instantiates the witnessing cut on a measured quantum system. The H → ZZ* prediction (Section 4) is the first case where the flag complex produces non-trivial persistent homology — an L-centric topological signature encoding the decay's entanglement structure. This is a testable prediction for Run 3 tripartite tomography.

---

## Headline result

The **first log-negativity measurement of the CMS tt̄ state resolved across $m_{t\bar{t}}$ bins**, using the full 15-coefficient reconstruction with covariance-propagated uncertainties:

| $m_{t\bar{t}}$ range [GeV] | $D$ | $\tilde{D}$ | $E_N$ [bits, 68% CL] |
|---|---|---|---|
| 300–400 (threshold) | −0.412 ± 0.035 | −0.052 ± 0.036 | **0.18 [0.13, 0.29]** ✓ entangled |
| 400–600 | −0.204 ± 0.012 | −0.013 ± 0.012 | 0 |
| 600–800 | −0.089 ± 0.020 | +0.069 ± 0.020 | 0 |
| > 800 | −0.005 ± 0.024 | +0.112 ± 0.025 | 0 |

Source data: CMS 138 fb⁻¹ at √s = 13 TeV, HEPData record `ins2829523`, tables t9 (values) and t10 (64×64 covariance). See `fig_ttbar_3panel.png` for the full visualization.

As a **theoretical prediction** for a multipartite system not yet measured, applying the RWF flag-complex construction to the Aguilar-Saavedra SM density operator for $H \to ZZ^*$ yields an L-centric tripartite topology with $E_N(L{:}S_i) \approx 1$ bit and $E_N(S_1{:}S_2) \lesssim 0.01$ bits. This section is explicitly a Run 3 target, not a measurement claim.

---

## What this code does NOT do

This repo is scoped narrowly. It does not:

- Reconstruct events from LHC open data. It uses the published reduced density matrix coefficients from HEPData.
- Measure the tripartite $H \to ZZ^*$ state. The tripartite operator has not yet been published as a measurement; Aguilar-Saavedra (arXiv:2403.13942) shows Run 3 will get there at >5σ. Until then, the H → ZZ* section here is the SM LO prediction for that operator.
- Compute CMS's high-mass boosted >5σ entanglement result. That observation uses a |cos θ| < 0.4 selection (HEPData table t11/t12) that this analysis does not yet apply.
- Compute higher-order QCD corrections or SMEFT effects.

---

## Repository layout

```
.
├── README.md                              # this file
├── FINDINGS.md                            # development notes — honest history of what the data showed
├── LICENSE                                # MIT
├── topological_signatures_entanglement.tex # the paper (Overleaf-ready LaTeX)
├── fig_ttbar_3panel.png                   # headline figure: E_N, D, D-tilde across m(tt)
├── fig1_hzz_edge_weights.png              # H → ZZ* theory prediction: pairwise log-neg
├── fig2_hzz_persistence.png               # H → ZZ* persistence diagram
├── fig3_hzz_filtration.png                # H → ZZ* flag complex filtration
│
├── Core pipeline
│   ├── rwf_pipeline.py                    # qubit case: Fano reconstruction, log-neg, flag complex
│   └── rwf_pipeline_mixed.py              # extension to heterogeneous local dims (for tripartite 9×3×3)
│
├── tt̄ analysis (measured CMS data)
│   ├── parse_cms_differential.py          # HEPData CSV → structured dict
│   ├── cms_measured_analysis.py           # inclusive analysis (reproduces E_N = 0 inclusively)
│   ├── cms_differential_analysis.py       # main: per-bin E_N with covariance propagation
│   ├── plot_ttbar_3panel.py               # generates fig_ttbar_3panel.png
│   └── diagnose_bins.py                   # per-bin ρ/ρᵀᴬ eigenvalues, D, D-tilde
│
├── H → ZZ* theoretical prediction
│   ├── hvv_state.py                       # tripartite pure state from Aguilar-Saavedra
│   ├── hzz_signature.py                   # kinematic scan over m(Z*)
│   └── generate_figures.py                # produces fig1/fig2/fig3
│
├── Sanity tests
│   └── test_multipartite.py               # GHZ, W, cluster, disjoint-Bell reference states
│
└── data/                                  # (gitignored) — HEPData CSVs go here
    ├── cms_full_matrix_differential_mtt.csv
    └── cms_full_matrix_differential_mtt_covariance.csv
```

---

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`, `gudhi`

```bash
pip install numpy scipy matplotlib gudhi
```

---

## Quick start

```bash
# 1. Sanity checks on known multipartite states (no data required)
python3 test_multipartite.py

# 2. Pull CMS differential measurement from HEPData
mkdir -p data
cd data
curl -L -o cms_full_matrix_differential_mtt.csv \
  "https://www.hepdata.net/download/table/ins2829523/Results%20for%20full%20matrix%20m(tt)%20differential/csv"
curl -L -o cms_full_matrix_differential_mtt_covariance.csv \
  "https://www.hepdata.net/download/table/ins2829523/Covariance%20for%20full%20matrix%20m(tt)%20differential/csv"
cd ..

# 3. Run the main t-tbar analysis
python3 cms_differential_analysis.py
python3 plot_ttbar_3panel.py

# 4. Diagnostics (eigenvalues per bin)
python3 diagnose_bins.py

# 5. H → ZZ* theoretical prediction (no data required)
python3 hzz_signature.py
python3 generate_figures.py

# 6. Regenerate the paper (requires pdflatex or Overleaf)
pdflatex topological_signatures_entanglement.tex
```

Total runtime: ~1 minute on a laptop.

---

## Methodology notes

### Density matrix reconstruction

ATLAS/CMS publish the Fano-Bloch coefficients $(B_\pm, C_{ij})$ of the tt̄ spin density matrix in the helicity basis {k̂, n̂, r̂}. We use the ATLAS/CMS sign convention (negative C in the decomposition) throughout. CMS publishes the symmetric $C^+_{ij} = (C_{ij} + C_{ji})/2$ and antisymmetric $C^-_{ij} = (C_{ij} - C_{ji})/2$ parts; the full $C_{ij}$ is $C^+ + C^-$, $C_{ji}$ is $C^+ - C^-$.

### Uncertainty propagation

The measurements in each m(tt̄) bin are jointly distributed via the 16×16 intra-bin covariance matrix (HEPData table t10). We draw 20,000 samples from this multivariate Gaussian and, for each sample, reconstruct ρ and compute the PT eigenvalues. Unphysical samples (ρ with a negative eigenvalue) are projected to the PSD cone by clipping and renormalizing; this preserves Hermiticity and unit trace and does not inflate E_N.

Positivity violation rates reflect the statistical precision of the measurement: 40% in the threshold bin, 54% in the 400–600 bin, 6% in the 600–800 bin, 0% in the boosted bin. The quoted intervals on E_N are robust to the choice of projection method.

### Topological extension

For n ≥ 3, we apply RWF Definition 2 (flag complex weighted by pairwise E_N, filtered by decreasing ε, persistent homology via Gudhi). For the bipartite case this reduces trivially to reporting E_N. For H → ZZ* the pipeline is exercised on the 9×3×3 tripartite state.

---

## Known limitations

1. **l=0 Clebsch-Gordan coefficients** in `hvv_state.py` use a `1/√3` ansatz that should be verified against Aguilar-Saavedra eq. (7). The l=2 block, which dominates E_N(L:S_i), is exact. The residual E_N(S_1:S_2) may shift by an O(1) factor but remains suppressed relative to the L-S edges.
2. **Flag complex is blind to GHZ- and cluster-type entanglement**. Verified in `test_multipartite.py`. See Remark 8 of the foundational RWF paper.
3. **Boosted-regime entanglement**: CMS's >5σ observation uses |cos θ| < 0.4 (HEPData t11/t12). Not yet pulled. Would add a fifth column to the paper's main table.
4. **Polarizations held at zero for SM predictions** (LO parity-CP). A full NLO analysis would include the small P vectors.

---

## Citing

If you use this code, please cite:

```bibtex
@article{Scott2026TSE,
  author = {Scott, Trevor},
  title  = {Topological Signatures of Quantum Entanglement at the LHC: Measurement in $t\bar{t}$ and Prediction for $H \to ZZ^*$},
  year   = {2026},
  url    = {https://github.com/trevorscott/topological-signatures-entanglement}
}

@article{Scott2026RWF,
  author = {Scott, Trevor},
  title  = {The Relational Witnessing Framework: The Emergence of Time and Experience from Entanglement Entropy},
  year   = {2026}
}
```

And the key physics input:

```bibtex
@article{CMS2024,
  author  = {{CMS Collaboration}},
  title   = {Measurements of polarization and spin correlation and observation of entanglement in top quark pairs using lepton+jets events from proton-proton collisions at $\sqrt{s} = 13$ TeV},
  journal = {Phys. Rev. D},
  volume  = {110},
  pages   = {112016},
  year    = {2024},
  eprint  = {2409.11067}
}

@article{AguilarSaavedra2024,
  author  = {Aguilar-Saavedra, J. A.},
  title   = {Tripartite entanglement in $H \to ZZ, WW$ decays},
  journal = {Phys. Rev. D},
  volume  = {109},
  pages   = {113004},
  year    = {2024},
  eprint  = {2403.13942}
}
```

---

## References

- T. Scott, *The Relational Witnessing Framework: The Emergence of Time and Experience from Entanglement Entropy*, 2026.
- CMS Collaboration, arXiv:2409.11067; HEPData record `ins2829523`.
- ATLAS Collaboration, *Nature* **633**, 542 (2024), arXiv:2311.07288.
- J. A. Aguilar-Saavedra, *Phys. Rev. D* **109**, 113004 (2024), arXiv:2403.13942.
- G. Vidal, R. F. Werner, *Phys. Rev. A* **65**, 032314 (2002).
- A. Peres, *Phys. Rev. Lett.* **77**, 1413 (1996).
- D. Cohen-Steiner, H. Edelsbrunner, J. Harer, *Discrete Comput. Geom.* **37**, 103 (2007).
- The GUDHI Project, [gudhi.inria.fr](https://gudhi.inria.fr/).

---

## License

Code: MIT (see `LICENSE`). Paper (LaTeX + figures): CC-BY 4.0.

---

## Contact

Trevor Scott · `marquan03@gmail.com` 
