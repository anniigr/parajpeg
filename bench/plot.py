"""README figures from the benchmark results (matplotlib only).

    python bench/plot.py [results_dir] [out_dir]

Reads latency.json (bench/run_all.sh) and, if present, syncdist.csv (pjsyncdist --csv). Writes every
figure twice, for GitHub's light and dark themes: NAME.png and NAME-dark.png.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img")
out.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-4 and neutrals of a colour-blind-validated palette, stepped per theme.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}


def styled(theme, w=7.6, h=3.6):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax, t


def finish(fig, ax, t, title, subtitle, name, theme):
    ax.set_title(title, loc="left", fontsize=11, color=t["ink"], pad=22, fontweight="semibold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=t["ink2"], va="bottom")
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    leg = ax.get_legend()
    if leg:
        for txt in leg.get_texts():
            txt.set_color(t["ink"])
    fig.tight_layout()
    fig.savefig(out / f"{name}{'' if theme == 'light' else '-dark'}.png", dpi=160, facecolor=t["surface"])
    plt.close(fig)


lat = json.loads((res / "latency.json").read_text())
MACHINE = "Apple M3 (4P+4E), libjpeg-turbo 3.2.0 single-threaded = 1.0, median of 15 decodes"


def group_of(r):
    enc = r["file"].split("_", 1)[1].rsplit(".", 1)[0]  # e.g. q90_420, camera, lrg_q75_420
    enc = enc.replace("lrg_", "")
    if enc == "camera":
        enc = "camera_prog" if "prog" in r["kind"] else "camera_rst"
    return enc


groups = {}
for r in lat:
    groups.setdefault(group_of(r), []).append(r)
speedup = lambda r, i: r["turbo_ms"] / r["cpu_ms"][i]

# 1) Speedup by encoding at 1/4/8 threads (median over the photographs of each encoding).
ORDER = [("camera_rst", "camera\noriginal\n4:4:4+RST"), ("q85_420_rst", "q85 4:2:0\nRST every\nMCU row"),
         ("q75_420", "q75 4:2:0"), ("q90_420", "q90 4:2:0"), ("q90_444", "q90 4:4:4"),
         ("q85_prog", "q85\nprogressive\n4:2:0"), ("camera_prog", "progressive\noriginal\n4:4:4, 7 scans")]
ORDER = [(k, l) for k, l in ORDER if k in groups]
for theme in THEMES:
    fig, ax, t = styled(theme, 7.8, 3.9)
    x = np.arange(len(ORDER))
    w = 0.26
    thread_idx = {th: lat[0]["threads"].index(th) for th in (1, 4, 8)}
    for j, (th, idx) in enumerate(thread_idx.items()):
        vals = [statistics.median(speedup(r, idx) for r in groups[k]) for k, _ in ORDER]
        ax.bar(x + (j - 1) * w, vals, w - 0.03, color=t["series"][j], label=f"{th} thread{'s' if th > 1 else ''}")
        if th == 8:
            for xi, v in zip(x, vals):
                ax.text(xi + w, v + 0.08, f"{v:.1f}×", ha="center", va="bottom", fontsize=8.5, color=t["ink"])
    ax.axhline(1.0, color=t["ink2"], lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x, [l for _, l in ORDER], fontsize=8.5)
    ax.set_ylabel("speedup over libjpeg-turbo (dashed = 1)")
    ax.set_ylim(0, 6.2)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.5, bbox_to_anchor=(1, 1.0))
    finish(fig, ax, t, "Single-image decode latency: 12 MP and 21 MP NASA photographs", MACHINE, "speedup", theme)

# 2) Scaling with threads, median over the photographs, for four representative encodings.
SERIES = [("q85_420_rst", "restart markers (q85 4:2:0)"), ("q90_444", "no markers, 4:4:4 (q90)"),
          ("q90_420", "no markers, 4:2:0 (q90)"), ("q85_prog", "progressive (q85 4:2:0)")]
for theme in THEMES:
    fig, ax, t = styled(theme, 7.6, 3.7)
    ths = lat[0]["threads"]
    ends = []
    for j, (k, label) in enumerate(SERIES):
        if k not in groups:
            continue
        ys = [statistics.median(speedup(r, i) for r in groups[k]) for i in range(len(ths))]
        ax.plot(ths, ys, color=t["series"][j], lw=2, marker="o", ms=5, label=label)
        ends.append([ys[-1], ys[-1], t["series"][j]])
    ends.sort()
    for a, b in zip(ends, ends[1:]):  # keep end labels at least 0.28 apart
        b[1] = max(b[1], a[1] + 0.28)
    for v, y, c in ends:
        ax.plot([ths[-1] + 0.12, ths[-1] + 0.3], [v, y], color=c, lw=1)
        ax.text(ths[-1] + 0.35, y, f"{v:.1f}×", va="center", fontsize=8.5, color=t["ink"])
    ax.axhline(1.0, color=t["ink2"], lw=1, ls=(0, (4, 3)))
    ax.set_xticks(ths)
    ax.set_xlim(0.6, ths[-1] + 1.1)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("threads (cores 1–4 are performance cores, 5–8 efficiency cores)")
    ax.set_ylabel("speedup over libjpeg-turbo")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    finish(fig, ax, t, "Thread scaling by encoding", MACHINE, "scaling", theme)

# 3) Synchronisation distance of speculatively started decoders.
sync = res / "syncdist.csv"
if sync.exists():
    rows = list(csv.DictReader(sync.open()))
    def sub(r):
        return "444" if "_444" in r["file"] else "420"
    curves = [("full state, 4:2:0", [float(r["state_bytes"]) for r in rows if sub(r) == "420"]),
              ("full state, 4:4:4", [float(r["state_bytes"]) for r in rows if sub(r) == "444"]),
              ("bit position only", [float(r["bit_bytes"]) for r in rows])]
    nimg = len({r["file"] for r in rows})
    for theme in THEMES:
        fig, ax, t = styled(theme, 7.6, 3.7)
        for j, (label, v) in enumerate(curves):
            v = np.sort(np.maximum(np.array(v), 0.5))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, color=t["series"][j], lw=2, label=label)
        for k, lab in ((256, "GPU subsequence\n256 B"), (1024, "CPU checkpoint\n1 KiB")):
            ax.axvline(k, color=t["ink2"], lw=1, ls=(0, (4, 3)))
            ax.text(k * 1.08, 30, lab, fontsize=8, color=t["ink2"], va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(1, 4000)
        ax.set_ylim(0, 101)
        ax.xaxis.grid(True, color=t["grid"], lw=0.8)
        ax.set_xlabel("bytes decoded from the arbitrary start until in step with the true decoder (log scale)")
        ax.set_ylabel("% of start positions")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        finish(fig, ax, t, "How fast a decoder started at a random byte synchronises",
               f"{len(rows):,} random starts in {nimg} baseline photographs without restart markers (pjsyncdist)", "syncdist", theme)

print("wrote figures to", out)
