# Collider Density Matrix & Topology Pipeline

Public code and data pipeline for processing $t\bar{t}$ spin correlation measurements into information-theoretic and topological visualizations.

This repository provides an open-source Python toolchain to compute the logarithmic negativity $E_N$ of reconstructed quantum density matrices at the LHC from published HEPData. It also includes an exploratory module applying Topological Data Analysis (TDA) via flag-complex persistent homology to $n$-partite quantum states.

---

## ⚠️ Physics & Methodology Limitations

This pipeline is an exploratory methodological tool, and the outputs should be interpreted as **inferences under leading-order (LO) assumptions**, not direct measurements. Users should be aware of the following physical and mathematical constraints:

1. **The PSD Reconstruction Bias:** Unconstrained coefficient tomography often yields unphysical density matrices (e.g., negative eigenvalues). This pipeline projects samples onto the positive-semidefinite (PSD) cone by clipping eigenvalues and renormalizing. At high violation rates (e.g., 40-54% in threshold bins), this projection introduces uncharacterized systematic biases into the $E_N$ calculation.
2. **Breakdown of the LO Mapping:** Reconstructing the density matrix from angular coefficients assumes a strict tree-level mapping. In reality, fiducial detector selections and Next-to-Leading Order (NLO) corrections heavily distort these coefficients (see Grossi et al., *JHEP* 12 (2024) 120).
3. **Topology at Low-$N$:** The flag-complex construction relies strictly on pairwise marginals. For small systems like bipartite ($N=2$) or tripartite ($N=3$), the persistent homology reduces to the pairwise scalars and is blind to genuine multipartite entanglement. It is included here as a visualization tool and scaffolding for future high-multiplicity ($N \gg 3$) environments. 

---

## Pipeline Capabilities

**1. Data Ingestion & Reconstruction**
* Parses published Fano-Bloch coefficients $(B_\pm, C_{ij})$ from ATLAS/CMS differential measurements via HEPData.
* Reconstructs the $t\bar{t}$ spin density matrix in the helicity basis.

**2. Covariance-Propagated Uncertainties**
* Draws 20,000 samples from the multivariate Gaussian utilizing the full $16 \times 16$ intra-bin covariance matrices.
* Applies PSD projection to maintain Hermiticity and unit trace.

**3. Quantum Information & Topology Modules**
* Calculates Logarithmic Negativity ($E_N$), alongside conventional scalar witnesses ($D$, $\tilde{D}$).
* Implements Gudhi-based persistent homology to generate filtration complexes and persistence diagrams for $n$-partite states.

---

## Repository layout

```
.
├── README.md                              # this file
├── FINDINGS.md                            # development notes — honest history of what the data showed
├── LICENSE                                # MIT
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

Trevor Scott 
