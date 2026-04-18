# RWF Topological Signatures at the LHC

A pipeline applying the Relational Witnessing Framework's flag-complex construction (Scott 2026, Section 6) to multipartite quantum states measured at the LHC. Validates on the ATLAS $t\bar{t}$ entanglement measurement and computes the tripartite topological signature of $H \to ZZ^*$.

## What this is

Given an $n$-partite quantum state $\rho$ on $\mathcal{H}_1 \otimes \cdots \otimes \mathcal{H}_n$, this code:

1. Computes the pairwise logarithmic negativity $E_N(i{:}j)$ between every pair of subsystems
2. Builds the flag (clique) complex $K_W$ weighted by these values
3. Runs persistent homology on the decreasing-threshold filtration
4. Outputs the topological signature $\chi(K_W)$ as a collection of persistence diagrams

The construction is a standalone quantum-information tool independent of the RWF's foundational axioms — it takes a density matrix in, and returns a basis-independent topological summary of its multipartite entanglement geometry.

## Results demonstrated here

**Bipartite baseline (ATLAS $t\bar{t}$):**
- Reproduces the measured $D = -0.537$ observable
- Computes $E_N(t{:}\bar{t}) = 0.385$ bits from the reconstructed Werner state
- Singlet fraction $\approx 0.65$

**Tripartite result ($H \to ZZ^*$, subsystems $L \otimes S_1 \otimes S_2$ with dims $9 \times 3 \times 3$):**
- $E_N(L{:}S_1) = E_N(L{:}S_2) \approx 1.04$ bits
- $E_N(S_1{:}S_2) \approx 0.01$ bits
- Flag complex is $L$-centric: $V$-shaped 1-skeleton across most of the filtration; 2-simplex closes only at $\epsilon \lesssim 0.01$

The $100\times$ hierarchy in pairwise entanglements is a direct topological encoding of the fact that the two $Z$-spins communicate *via* orbital angular momentum in the Higgs decay, not directly with each other.

## Files

| File | Purpose |
|---|---|
| `rwf_pipeline.py` | Core bipartite pipeline: Fano reconstruction, partial transpose, log-negativity, flag complex + persistence (qubit case). Includes ATLAS $t\bar{t}$ validation. |
| `rwf_pipeline_mixed.py` | Generalization to heterogeneous subsystem dimensions (needed for the $9 \times 3 \times 3$ Higgs case). |
| `hvv_state.py` | Constructs the tripartite pure state $\ket{\psi}_{LS_1S_2}$ for $H \to VV$ from the three helicity amplitudes, following Aguilar-Saavedra arXiv:2403.13942. |
| `test_multipartite.py` | Sanity tests on known states: GHZ, $W$, two disjoint Bell pairs, 4-qubit cluster. Confirms the expected topological signatures. |
| `atlas_ttbar_signature.py` | Applies the pipeline to the ATLAS tt̄ measurement. |
| `hzz_signature.py` | Main analysis: scans $m_{Z^*} \in [10, 60]$ GeV and computes $\chi(K_W)$ at each. |
| `generate_figures.py` | Produces the three paper figures. |
| `rwf_hzz_methodology.tex` | Full paper draft, Overleaf-ready. |

## Installation

```bash
pip install numpy scipy matplotlib gudhi
```

Tested against numpy 2.4, scipy 1.17, gudhi 3.12.

## Running

```bash
# Validate the pipeline end-to-end against ATLAS measurement
python rwf_pipeline.py

# Run sanity tests on known multipartite states
python test_multipartite.py

# Apply the pipeline to ATLAS tt̄
python atlas_ttbar_signature.py

# Main analysis: H -> ZZ* tripartite signature
python hzz_signature.py

# Regenerate the paper figures
python generate_figures.py
```

Expected runtime: each script completes in under 30 seconds on a laptop.

## Known limitations

1. **The $l=0$ Clebsch-Gordan coefficients in `hvv_state.py` are inferred from the amplitude-decomposition pattern rather than read directly from Aguilar-Saavedra eq. (7).** The $l=2$ block is verified correct; the $l=0$ contribution is small and does not affect the qualitative topological conclusions, but a physics reader should cross-check the $(1/\sqrt{3})$ ansatz against the published paper.
2. No error propagation on the measured $C_{ij}$ → $E_N$ values yet.
3. No NLO corrections to the helicity amplitudes; LO SM only.
4. The flag complex is blind to genuinely multipartite entanglement invisible to pair marginals (GHZ states, cluster states give trivial signatures). This is a known limitation per Remark 8 of the foundational RWF paper.

## Citation

If you use this pipeline:

```bibtex
@misc{scott2026rwf_hzz,
  author = {Scott, Trevor},
  title  = {Topological Signatures of Multipartite Entanglement in H->ZZ* Decays at the LHC},
  year   = {2026},
  url    = {https://github.com/[YOUR_USERNAME]/rwf-hzz}
}
```

And cite the foundational framework:

```bibtex
@misc{scott2026rwf,
  author = {Scott, Trevor},
  title  = {The Relational Witnessing Framework: The Emergence of Time and Experience from Entanglement Entropy},
  year   = {2026}
}
```

And the physics input:

```bibtex
@article{aguilarsaavedra2024,
  author  = {Aguilar-Saavedra, J. A.},
  title   = {Tripartite entanglement in $H \to ZZ, WW$ decays},
  journal = {Phys. Rev. D},
  volume  = {109},
  pages   = {113004},
  year    = {2024},
  eprint  = {2403.13942}
}

@article{atlas2024,
  author  = {{ATLAS Collaboration}},
  title   = {Observation of quantum entanglement with top quarks at the ATLAS detector},
  journal = {Nature},
  volume  = {633},
  pages   = {542},
  year    = {2024},
  eprint  = {2311.07288}
}
```

## License

MIT — see `LICENSE`.

## Contact

Trevor Scott - marquan03@gmail.com 
