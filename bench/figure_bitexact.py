"""Bit-exactness figure for the README: real decodes by libjpeg-turbo (djpeg) and parajpeg (pjdec),
and their per-byte difference, for an intact baseline file and a truncated progressive file.

    python bench/figure_bitexact.py SOURCE_IMAGE [out_dir]

SOURCE_IMAGE is any image cjpeg can read (PPM/BMP/...); the Kodak image kodim23 is used in the README.
Needs cjpeg/djpeg (libjpeg-turbo), build/pjdec, matplotlib and numpy. Writes bitexact.png and
bitexact-dark.png. The byte counts printed in the figure are computed here, not typed in.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

src = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img")
PJDEC = os.environ.get("PJDEC", "build/pjdec")
CJPEG = os.environ.get("CJPEG", "cjpeg")
DJPEG = os.environ.get("DJPEG", "djpeg")


def read_ppm(p):
    data = Path(p).read_bytes()
    parts, pos = [], 0
    while len(parts) < 4:  # magic, width, height, maxval
        while data[pos:pos + 1].isspace():
            pos += 1
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        parts.append(data[pos:end])
        pos = end
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(data[pos + 1:], np.uint8).reshape(h, w, 3)


def decode_both(jpg, tmp, tag):
    ref, mine = tmp / f"{tag}_ref.ppm", tmp / f"{tag}_pj.ppm"
    # libjpeg-turbo's C reference paths (JSIMD_FORCENONE): on damaged data its SIMD IDCT saturates
    # where the C code wraps; parajpeg follows the C code (see docs/correctness.md).
    env = dict(os.environ, JSIMD_FORCENONE="1")
    rc = subprocess.run([DJPEG, "-dct", "int", "-outfile", str(ref), str(jpg)], env=env, stderr=subprocess.DEVNULL).returncode
    if rc not in (0, 2):  # 2 = finished with warnings (e.g. premature end of data)
        raise RuntimeError(f"djpeg failed on {jpg}")
    subprocess.run([PJDEC, str(jpg), str(mine), "--threads", "8"], check=True, stderr=subprocess.DEVNULL)
    return read_ppm(ref), read_ppm(mine)


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    base, prog = tmp / "base.jpg", tmp / "prog.jpg"
    subprocess.run([CJPEG, "-quality", "90", "-sample", "2x2", "-optimize", "-outfile", str(base), str(src)], check=True)
    subprocess.run([CJPEG, "-quality", "85", "-progressive", "-outfile", str(prog), str(src)], check=True)
    cut = tmp / "prog_cut.jpg"
    pdata = prog.read_bytes()
    FRAC = 0.17  # inside the second scan (first luma AC band): the lower part keeps only DC + smoothing
    cut.write_bytes(pdata[: int(len(pdata) * FRAC)])
    rows = [("intact baseline JPEG (q90, 4:2:0)", *decode_both(base, tmp, "base")),
            (f"truncated progressive JPEG: first {FRAC:.0%} of {len(pdata):,} bytes, cut inside the second scan",
             *decode_both(cut, tmp, "cut"))]

THEMES = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e"),
          "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7")}
for theme, t in THEMES.items():
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
    fig.patch.set_facecolor(t["surface"])
    for r, (label, ref, mine) in enumerate(rows):
        diff = np.abs(ref.astype(int) - mine.astype(int))
        ndiff = int(np.count_nonzero(diff))
        panels = [(ref, "libjpeg-turbo 3.2.0 (djpeg -dct int)"), (mine, "parajpeg, 8 threads"),
                  (np.clip(diff * 64, 0, 255).astype(np.uint8), f"|difference| × 64: {ndiff:,} of {diff.size:,} bytes differ")]
        for c, (img, title) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(t["ink2"])
                s.set_linewidth(0.6)
            ax.set_title(title, fontsize=8.5, color=t["ink"] if c < 2 else t["ink2"], pad=4)
        axes[r][0].text(0, 1.13, label, transform=axes[r][0].transAxes, fontsize=9, color=t["ink"], fontweight="bold")
        print(f"{label}: {ndiff} differing bytes of {diff.size}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, hspace=0.32, wspace=0.04)
    fig.savefig(out / f"bitexact{'' if theme == 'light' else '-dark'}.png", dpi=110, facecolor=t["surface"])
    plt.close(fig)
