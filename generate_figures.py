"""
Generate figures for the methodology paper:
    Fig 1: Pairwise log-negativity vs m_Z_off (shows the asymmetric L-central topology)
    Fig 2: Persistence diagram for H -> ZZ at m_Z_off = 30 GeV
    Fig 3: Filtration visualization — what the flag complex looks like as epsilon decreases
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude")

from hvv_state import build_hvv_state, sm_hvv_amplitudes
from rwf_pipeline_mixed import (
    topological_signature_mixed,
    pairwise_negativity_matrix_mixed,
)

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 110


# ------------------------------------------------------------------
# Figure 1: pairwise log-negativity vs m_Z_off
# ------------------------------------------------------------------
MZ = 91.1876
MH = 125.10
mV_range = np.linspace(5, 70, 50)
rows = []
for mV in mV_range:
    a11, a00, am1m1 = sm_hvv_amplitudes(mV, MZ, MH)
    scale = max(abs(a11), abs(a00), abs(am1m1))
    a11 /= scale; a00 /= scale; am1m1 /= scale
    psi = build_hvv_state(a11, a00, am1m1)
    rho = np.outer(psi, psi.conj())
    W = pairwise_negativity_matrix_mixed(rho, [9, 3, 3])
    rows.append([mV, W[0, 1], W[0, 2], W[1, 2]])
rows = np.array(rows)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(rows[:, 0], rows[:, 1], "-", lw=2.0, label=r"$E_N(L : S_1)$", color="#2c3e50")
ax.plot(rows[:, 0], rows[:, 2], "--", lw=1.5, label=r"$E_N(L : S_2)$", color="#16a085")
ax.plot(rows[:, 0], rows[:, 3], "-", lw=2.0, label=r"$E_N(S_1 : S_2)$", color="#c0392b")
ax.set_xlabel(r"Off-shell $Z$ invariant mass $m_{Z^*}$ [GeV]")
ax.set_ylabel(r"Pairwise log-negativity [bits]")
ax.set_title(r"$H \to ZZ^*$: edge weights of the RWF flag complex")
ax.grid(alpha=0.3)
ax.legend(loc="center right", framealpha=0.95)
ax.set_ylim(-0.02, 1.15)
plt.tight_layout()
plt.savefig("/home/claude/fig1_hzz_edge_weights.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved Fig 1: /home/claude/fig1_hzz_edge_weights.png")


# ------------------------------------------------------------------
# Figure 2: persistence diagram at m_Z_off = 30 GeV
# ------------------------------------------------------------------
a11, a00, am1m1 = sm_hvv_amplitudes(30.0, MZ, MH)
scale = max(abs(a11), abs(a00), abs(am1m1))
a11 /= scale; a00 /= scale; am1m1 /= scale
psi = build_hvv_state(a11, a00, am1m1)
rho = np.outer(psi, psi.conj())
sig = topological_signature_mixed(rho, [9, 3, 3], max_dim=2)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
colors = ["#c0392b", "#2980b9", "#27ae60"]
markers = ["o", "s", "^"]
for dim in [0, 1, 2]:
    diag = sig["diagrams"].get(dim, [])
    if not diag:
        continue
    births = [b for (b, d) in diag]
    deaths = [d if d != float("-inf") else 0.0 for (b, d) in diag]
    ax.scatter(births, deaths, s=80, c=colors[dim], marker=markers[dim],
               label=f"$H_{dim}$", edgecolors="black", linewidths=0.5, zorder=10)

# Diagonal
mx = max(1.2, max(b for dim_, (b, _) in sig["persistence"]) * 1.1)
ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel(r"Birth $\epsilon$ (log-negativity, bits)")
ax.set_ylabel(r"Death $\epsilon$ (log-negativity, bits)")
ax.set_title(r"Persistence diagram, $H \to ZZ^*$ at $m_{Z^*} = 30$ GeV")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, mx)
ax.set_ylim(-0.05, mx)
plt.tight_layout()
plt.savefig("/home/claude/fig2_hzz_persistence.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved Fig 2: /home/claude/fig2_hzz_persistence.png")


# ------------------------------------------------------------------
# Figure 3: filtration visualization
# Build the flag complex at several thresholds, draw it as a graph
# ------------------------------------------------------------------
W = sig["W"]
labels = [r"$L$", r"$S_1$", r"$S_2$"]
# Positions for the three nodes
positions = {0: (0.0, 1.0), 1: (-0.87, -0.5), 2: (0.87, -0.5)}

# Three threshold values: above all edges, above S1-S2 only, below all
W_max = W.max()
thresholds = [0.5, 0.02, 0.005]  # decreasing

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, eps in zip(axes, thresholds):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Threshold $\\epsilon = {eps:.3f}$ bits", fontsize=12)

    # Draw all edges with weight >= eps
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            if W[i, j] >= eps:
                edges.append((i, j, W[i, j]))

    # Check if the 2-simplex is present (all 3 edges present)
    present_edges = {(i, j) for i, j, _ in edges}
    triangle_present = len(present_edges) == 3

    # Fill triangle if present
    if triangle_present:
        from matplotlib.patches import Polygon
        verts = [positions[0], positions[1], positions[2]]
        tri = Polygon(verts, facecolor="#c0392b", alpha=0.15, edgecolor="none")
        ax.add_patch(tri)

    # Draw edges
    for i, j, w in edges:
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        lw = 1 + 2 * w
        ax.plot([x0, x1], [y0, y1], "-", color="#2c3e50", lw=lw, zorder=1)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        ax.annotate(f"{w:.2f}", (xm, ym), ha="center", va="center",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="white", edgecolor="gray"))

    # Draw nodes
    for i, (x, y) in positions.items():
        ax.scatter([x], [y], s=900, c="#ecf0f1", edgecolors="black",
                   linewidths=1.5, zorder=3)
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=14, zorder=4)

    # Notation of simplices
    note = f"# edges = {len(edges)}"
    if triangle_present:
        note += ", 2-simplex $\\{L, S_1, S_2\\}$ present"
    ax.text(0, -0.9, note, ha="center", fontsize=10, style="italic")

fig.suptitle(r"Flag complex filtration of $H \to ZZ^*$ at $m_{Z^*} = 30$ GeV",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("/home/claude/fig3_hzz_filtration.png", dpi=140, bbox_inches="tight")
plt.close()
print("Saved Fig 3: /home/claude/fig3_hzz_filtration.png")

print()
print("All three figures saved.")
