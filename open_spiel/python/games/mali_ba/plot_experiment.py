#!/usr/bin/env python3
"""Plot heuristic experiment results and fit a quadratic response surface
to find the optimal (mult_add_in, add_add_in) combination.

Usage:
    python plot_experiment.py [results_csv]
    defaults to /tmp/heuristic_experiment_results.csv
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

results_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/heuristic_experiment_results.csv"

mult_vals, add_vals, win_vals = [], [], []

with open(results_file) as f:
    for row in csv.DictReader(f):
        mult_vals.append(float(row["mult_add_in"]))
        add_vals.append(float(row["add_add_in"]))
        win_vals.append(int(row["total_wins"]))

m = np.array(mult_vals)
a = np.array(add_vals)
w = np.array(win_vals)
n = len(w)

# -----------------------------------------------------------------------
# Quadratic response surface fit (no external deps — pure numpy lstsq)
# Model: w = b0 + b1*m + b2*a + b3*m^2 + b4*a^2 + b5*m*a
# -----------------------------------------------------------------------
X = np.column_stack([
    np.ones(n),   # b0  intercept
    m,            # b1  mult
    a,            # b2  add
    m**2,         # b3  mult^2
    a**2,         # b4  add^2
    m * a,        # b5  interaction
])

coeffs, residuals, rank, sv = np.linalg.lstsq(X, w, rcond=None)
b0, b1, b2, b3, b4, b5 = coeffs

w_pred = X @ coeffs
ss_res = np.sum((w - w_pred)**2)
ss_tot = np.sum((w - w.mean())**2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

print(f"\n--- Quadratic Response Surface Fit (n={n}) ---")
print(f"  intercept : {b0:+.3f}")
print(f"  mult      : {b1:+.3f}")
print(f"  add       : {b2:+.3f}")
print(f"  mult^2    : {b3:+.3f}")
print(f"  add^2     : {b4:+.3f}")
print(f"  mult*add  : {b5:+.3f}")
print(f"  R²        : {r2:.3f}")

# -----------------------------------------------------------------------
# Find predicted optimum on a fine grid within the sampled ranges
# (analytical optimum can be outside the feasible range, so grid is safer)
# -----------------------------------------------------------------------
MULT_MIN, MULT_MAX = 0.2, 1.5
ADD_MIN,  ADD_MAX  = 0.0, 2.6
mg = np.linspace(MULT_MIN, MULT_MAX, 200)
ag = np.linspace(ADD_MIN,  ADD_MAX,  200)
MG, AG = np.meshgrid(mg, ag)
WG = (b0
      + b1 * MG
      + b2 * AG
      + b3 * MG**2
      + b4 * AG**2
      + b5 * MG * AG)

best_idx = np.unravel_index(np.argmax(WG), WG.shape)
best_mult = MG[best_idx]
best_add  = AG[best_idx]
best_pred = WG[best_idx]

print(f"\n--- Predicted Optimum ---")
print(f"  mult_add_in = {best_mult:.4f}")
print(f"  add_add_in  = {best_add:.4f}")
print(f"  predicted wins = {best_pred:.1f}")
print(f"  best observed  = {w.max():.0f}  (at mult={m[np.argmax(w)]:.4f}, add={a[np.argmax(w)]:.4f})")

# -----------------------------------------------------------------------
# Plots: scatter + marginals + heatmap
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Heuristic Parameter Experiment  (n={n}, R²={r2:.2f})", fontsize=14)

# -- Top row: scatter plots with fitted marginal curves --
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(m, w, color="steelblue", alpha=0.7, zorder=3)
ms = np.linspace(MULT_MIN, MULT_MAX, 200)
# Marginal curve: hold add at its mean
a_mean = a.mean()
w_marg_m = b0 + b1*ms + b2*a_mean + b3*ms**2 + b4*a_mean**2 + b5*ms*a_mean
ax1.plot(ms, w_marg_m, color="steelblue", linewidth=2, label=f"fit (add={a_mean:.2f})")
ax1.axvline(best_mult, color="red", linestyle="--", linewidth=1, label=f"opt={best_mult:.3f}")
ax1.set_xlabel("mult_add_in")
ax1.set_ylabel("Total Wins")
ax1.set_title("Wins vs mult_add_in")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(a, w, color="darkorange", alpha=0.7, zorder=3)
as_ = np.linspace(ADD_MIN, ADD_MAX, 200)
m_mean = m.mean()
w_marg_a = b0 + b1*m_mean + b2*as_ + b3*m_mean**2 + b4*as_**2 + b5*m_mean*as_
ax2.plot(as_, w_marg_a, color="darkorange", linewidth=2, label=f"fit (mult={m_mean:.2f})")
ax2.axvline(best_add, color="red", linestyle="--", linewidth=1, label=f"opt={best_add:.3f}")
ax2.set_xlabel("add_add_in")
ax2.set_ylabel("Total Wins")
ax2.set_title("Wins vs add_add_in")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# -- Top right: residuals (sanity check) --
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(w_pred, w - w_pred, color="gray", alpha=0.7)
ax3.axhline(0, color="black", linewidth=1)
ax3.set_xlabel("Predicted wins")
ax3.set_ylabel("Residual (actual − predicted)")
ax3.set_title("Residuals")
ax3.grid(True, alpha=0.3)

# -- Bottom: heatmap of fitted surface with data points overlaid --
ax4 = fig.add_subplot(2, 1, 2)
hm = ax4.contourf(MG, AG, WG, levels=30, cmap=cm.viridis)
fig.colorbar(hm, ax=ax4, label="Predicted wins")
sc = ax4.scatter(m, a, c=w, cmap=cm.viridis, edgecolors="white",
                 linewidths=0.8, s=60, zorder=5, label="Observed")
ax4.plot(best_mult, best_add, "r*", markersize=16, zorder=6,
         label=f"Predicted optimum ({best_mult:.3f}, {best_add:.3f})")
ax4.set_xlabel("mult_add_in")
ax4.set_ylabel("add_add_in")
ax4.set_title("Fitted Response Surface  (★ = predicted optimum)")
ax4.legend(fontsize=9)

plt.tight_layout()
plt.show()