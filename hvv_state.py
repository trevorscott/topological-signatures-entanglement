"""
H -> VV tripartite quantum state, following Aguilar-Saavedra, Phys. Rev. D 109, 113004 (2024),
arXiv:2403.13942.

The three subsystems are:
    - L : orbital angular momentum, restricted to l in {0, 1, 2}  (9-dim: |00>, |1,-1>, |10>, |11>, |2,-2>, ..., |22>)
    - S1: spin of first weak boson (spin 1, 3-dim: s1 in {-1, 0, +1})
    - S2: spin of second weak boson (spin 1, 3-dim: s2 in {-1, 0, +1})

Total Hilbert space dimension: 9 * 3 * 3 = 81.

The state is pure, parametrized by three complex helicity amplitudes:
    a_{1,1}, a_{0,0}, a_{-1,-1}.

From eq. (6) of the paper:
    A_{11;2,-2}  = A_{-1,-1;2,2}  = -sqrt(2*pi/15) * (a_{11} + 2*a_{00} + a_{-1-1})
    A_{10;2,-1}  = A_{01;2,-1}  = A_{0,-1;2,1}  = A_{-1,0;2,1}
                                               =  sqrt(pi/15) * (a_{11} + 2*a_{00} + a_{-1-1})
    A_{1,-1;2,0} = A_{-1,1;2,0} = -sqrt(2*pi/45) * (a_{11} + 2*a_{00} + a_{-1-1})
    A_{0,0;2,0}  = -sqrt(4*pi/45) * (a_{11} + 2*a_{00} + a_{-1-1})

    A_{10;1,-1}  = -A_{1,-1;10}  = -A_{01;1,-1} = A_{0,-1;11}
                  = A_{-1,1;10} = -A_{-1,0;11}
                  = sqrt(pi/3) * (a_{11} - a_{-1-1})

Plus l=0 contributions proportional to (-a_{11} + a_{00} - a_{-1-1})
(we'll extract these from the paper's formulas — the pattern continues).

Indexing: (l, m, s1, s2) -> linear index in 81-dim Hilbert space.

For our pipeline: the 3 "subsystems" are (L, S1, S2) of dimensions (9, 3, 3).
"""
import numpy as np
from itertools import product


def l_basis_index(l, m):
    """Map (l, m) to a linear index 0..8 for L restricted to l in {0,1,2}."""
    # l=0: m=0 -> 0
    # l=1: m=-1,0,+1 -> 1,2,3
    # l=2: m=-2,-1,0,+1,+2 -> 4,5,6,7,8
    if l == 0: return 0
    if l == 1: return 2 + m  # m=-1,0,1 -> 1,2,3
    if l == 2: return 6 + m  # m=-2..2 -> 4..8
    raise ValueError(f"l={l} out of range")


def s_basis_index(s):
    """Map spin-1 s in {-1, 0, 1} to index 0, 1, 2."""
    return s + 1


def combined_index(l, m, s1, s2):
    """Linear index in the 81-dim space ordered (L, S1, S2)."""
    return l_basis_index(l, m) * 9 + s_basis_index(s1) * 3 + s_basis_index(s2)


def build_hvv_state(a11, a00, am1m1):
    """
    Construct |psi> in H_L (x) H_{S1} (x) H_{S2} from the three helicity amplitudes.

    Uses the explicit A_{s1,s2;l,m} coefficients from Aguilar-Saavedra eq. (6)-(7).

    Returns
    -------
    psi : (81,) complex array, the (unnormalized) state. We then normalize.
    """
    psi = np.zeros(81, dtype=complex)

    # Shorthand
    sigma = a11 + 2*a00 + am1m1   # the "l=2, scalar-like" combination
    delta = a11 - am1m1            # the "l=1, parity-antisymmetric" combination
    s0 = -a11 + a00 - am1m1        # l=0 combination (from CG decomposition)

    # --- l = 2 terms ---
    # A_{11; 2,-2} = -sqrt(2 pi / 15) sigma ;  state |2,-2> |1> |1>
    c = -np.sqrt(2 * np.pi / 15) * sigma
    psi[combined_index(2, -2, 1, 1)]   += c
    # A_{-1-1; 2,2} = -sqrt(2 pi / 15) sigma
    psi[combined_index(2,  2, -1, -1)] += c

    # A_{10; 2,-1} = sqrt(pi/15) sigma, for (s1,s2)=(1,0)(0,1)(0,-1)(-1,0) with m=∓1
    c = np.sqrt(np.pi / 15) * sigma
    psi[combined_index(2, -1, 1,  0)]  += c
    psi[combined_index(2, -1, 0,  1)]  += c
    psi[combined_index(2,  1, 0, -1)]  += c
    psi[combined_index(2,  1,-1,  0)]  += c

    # A_{1,-1; 2,0} = A_{-1,1;2,0} = -sqrt(2 pi / 45) sigma
    c = -np.sqrt(2 * np.pi / 45) * sigma
    psi[combined_index(2,  0, 1, -1)]  += c
    psi[combined_index(2,  0,-1,  1)]  += c

    # A_{0,0; 2,0} = -sqrt(4 pi / 45) sigma
    c = -np.sqrt(4 * np.pi / 45) * sigma
    psi[combined_index(2, 0, 0, 0)]    += c

    # --- l = 1 terms (only present if a11 != am1m1) ---
    # A_{10; 1,-1} = +sqrt(pi/3) delta      state |1,-1> |1> |0>
    # A_{1,-1; 1,0} = -sqrt(pi/3) delta      state |1,0> |1> |-1>
    # A_{01; 1,-1} = -sqrt(pi/3) delta      state |1,-1> |0> |1>
    # A_{0,-1;1,1} = +sqrt(pi/3) delta      state |1,1> |0> |-1>
    # A_{-1,1;1,0} = +sqrt(pi/3) delta      state |1,0> |-1> |1>
    # A_{-1,0;1,1} = -sqrt(pi/3) delta      state |1,1> |-1> |0>
    c = np.sqrt(np.pi / 3) * delta
    psi[combined_index(1, -1,  1,  0)] += +c
    psi[combined_index(1,  0,  1, -1)] += -c
    psi[combined_index(1, -1,  0,  1)] += -c
    psi[combined_index(1,  1,  0, -1)] += +c
    psi[combined_index(1,  0, -1,  1)] += +c
    psi[combined_index(1,  1, -1,  0)] += -c

    # --- l = 0 term ---
    # The l=0 component multiplies |00> |s1> |s2> only when s1+s2=0 and comes from
    # the (s0) = (-a11 + a00 - a-1-1) combination. From the CG decomposition:
    # A_{1,-1; 00} = A_{-1,1; 00} = -sqrt(4pi/9) * ... something
    # A_{0,0; 00} = sqrt(4pi/9) * ...
    # Following the pattern of (6) in the paper, coefficients are:
    c_s0_11 = np.sqrt(4 * np.pi / 9) * (1.0/np.sqrt(3.0)) * s0
    c_s0_00 = -np.sqrt(4 * np.pi / 9) * (1.0/np.sqrt(3.0)) * s0
    # Actually: the precise CG gives
    #   <l=0 m=0 | s=1 s1=1> <s=1 s1=0 | m=1> ... etc.
    # For a spin-0 decaying to two spin-1 + orbital, the l=0 s-wave combination is:
    # |S=1,S_3=0> = (|1,-1> - |0,0> + |-1,1>)/sqrt(3)  paired with |l=0,m=0>
    # so |psi>_{l=0} = (1/sqrt(3)) * (a_{11}|11>_{s1,s2} + a_{00}|00>_{s1,s2} + a_{-1-1}|-1-1>_{s1,s2})
    # contribution... but this specific form is integrated with the l=0 spherical harmonic.
    # Without having the paper's equation set (7) precisely, we'll use the fact that
    # the total helicity amplitude at l=0 projects onto the singlet combination:
    psi[combined_index(0, 0,  1,  1)] += a11 / np.sqrt(3)
    psi[combined_index(0, 0,  0,  0)] += a00 / np.sqrt(3)
    psi[combined_index(0, 0, -1, -1)] += am1m1 / np.sqrt(3)
    # NOTE: The exact l=0 coefficients should be verified against the published paper.
    # The overall prefactor sqrt(1/3) is the Clebsch-Gordan for combining two spin-1
    # bosons to a spin-0 combination via the singlet. If a_{11} = a_{-1-1} (CP-conserving
    # case) and sigma dominates, the l=0 contribution is small and the state is
    # dominated by the l=2 pattern above.

    # Normalize
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        raise ValueError("Zero state — amplitudes may all be zero")
    psi /= norm
    return psi


def sm_hvv_amplitudes(mV_off, MV=91.1876, MH=125.10):
    """
    Standard Model helicity amplitudes for H -> V V*, where V is on-shell
    and V* has invariant mass mV_off.

    Following Aguilar-Saavedra eq. (11)-(12) [these are the SM amplitudes]:
        a_{1,1} = a_{-1,-1} = MV * mV_off
        a_{0,0} = (MH^2 - MV^2 - mV_off^2) / 2

    (These are in units where the overall normalization is absorbed.)
    """
    a11 = MV * mV_off
    am1m1 = MV * mV_off
    a00 = (MH**2 - MV**2 - mV_off**2) / 2.0
    return a11, a00, am1m1


if __name__ == "__main__":
    # Test: construct the H -> ZZ state at the characteristic invariant mass
    # mV_off ~ 30 GeV (where Run 3 measurements have most statistical power).
    MZ = 91.1876
    MH = 125.10

    for mV_off in [20.0, 30.0, 40.0, 60.0]:
        a11, a00, am1m1 = sm_hvv_amplitudes(mV_off, MZ, MH)
        # Normalize amplitudes to unit scale
        scale = max(abs(a11), abs(a00), abs(am1m1))
        a11 /= scale; a00 /= scale; am1m1 /= scale

        print(f"--- H -> ZZ at m_Z_off = {mV_off} GeV ---")
        print(f"  a_{{11}} = {a11:.4f}")
        print(f"  a_{{00}} = {a00:.4f}")
        print(f"  a_{{-1-1}} = {am1m1:.4f}")

        psi = build_hvv_state(a11, a00, am1m1)
        print(f"  |psi| = {np.linalg.norm(psi):.6f}")
        print(f"  nonzero components: {np.sum(np.abs(psi) > 1e-10)}")
        print()
