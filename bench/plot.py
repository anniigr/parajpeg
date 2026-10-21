"""Plots for the README from bench/results/*.json (matplotlib, no seaborn)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img")
out.mkdir(parents=True, exist_ok=True)
lat = json.loads((res / "latency.json").read_text())

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C_TURBO, C_PJ = "#8c959f", "#0969da"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

def kind_label(k):
    return {"444 rst": "camera 4:4:4\n+restart", "420": "4:2:0", "444": "4:4:4", "420 rst": "4:2:0\n+restart",
            "420 prog": "progressive\n4:2:0"}.get(k, k)

# 1) Speedup over single-threaded libjpeg-turbo, by encoding (median over the NASA images).
groups = {}
for r in lat:
    name = r["file"].split("_", 1)[1].rsplit(".", 1)[0]
    groups.setdefault(name, []).append(r)
order = ["camera", "q75_420", "q90_420", "q90_444", "q85_420_rst", "q85_prog"]
labels = {"camera": "camera\noriginal", "q75_420": "q75 4:2:0", "q90_420": "q90 4:2:0", "q90_444": "q90 4:4:4",
          "q85_420_rst": "q85 4:2:0\nrestart", "q85_prog": "q85\nprogressive"}
def med(v):
    v = sorted(v); return v[len(v) // 2]
xs, one, best = [], [], []
for g in order:
    if g not in groups: continue
    rs = groups[g]
    xs.append(labels[g])
    one.append(med([r["turbo_ms"] / r["cpu_ms"][0] for r in rs]))
    best.append(med([r["turbo_ms"] / min(r["cpu_ms"]) for r in rs]))
fig, ax = plt.subplots(figsize=(7.5, 3.4))
import numpy as np
x = np.arange(len(xs)); w = 0.38
ax.bar(x - w / 2, one, w, color=C_TURBO, label="parajpeg, 1 thread")
ax.bar(x + w / 2, best, w, color=C_PJ, label="parajpeg, best of 1-8 threads")
ax.axhline(1.0, color=INK, lw=0.8)
for i, v in enumerate(best): ax.text(i + w / 2, v + 0.05, f"{v:.1f}x", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, xs); ax.set_ylabel("speedup vs libjpeg-turbo (1 thread)")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_ylim(0, max(best) * 1.28)
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.set_title("Single-image decode latency, 12–21 MP NASA photographs, Apple M3", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(out / "speedup.png", dpi=160); plt.close(fig)

# 2) Thread scaling (throughput in megapixels/s) per encoding, one representative image.
fig, ax = plt.subplots(figsize=(7.5, 3.4))
palette = ["#0969da", "#1a7f37", "#9a6700", "#8250df", "#cf222e", "#57606a"]
rep = sorted(groups.get("q90_420", lat), key=lambda r: -r["width"] * r["height"])[0]["file"].split("_")[0]
for i, g in enumerate(order):
    rs = [r for r in groups.get(g, []) if r["file"].startswith(rep)]
    if not rs: continue
    r = rs[0]; mp = r["width"] * r["height"] / 1e6
    ax.plot(r["threads"], [mp / (t / 1e3) for t in r["cpu_ms"]], marker="o", ms=3.5, lw=1.6, color=palette[i], label=labels[g].replace("\n", " "))
    ax.scatter([1], [mp / (r["turbo_ms"] / 1e3)], marker="x", color=palette[i], s=25)
ax.set_xlabel("threads"); ax.set_ylabel("megapixels / s"); ax.set_xticks(rs[0]["threads"] if rs else [1, 2, 4, 8])
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title(f"Thread scaling, {rep} (x = libjpeg-turbo)", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(out / "scaling.png", dpi=160); plt.close(fig)
print("wrote", out / "speedup.png", out / "scaling.png")
