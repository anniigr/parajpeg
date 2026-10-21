#!/usr/bin/env bash
# Runs the benchmark matrix and writes JSON + markdown into bench/results/.
set -euo pipefail
BIN=${BIN:-build/pjbench}
DATA=${1:-bench/data/jpg}
OUT=${2:-bench/results}
RUNS=${RUNS:-9}
T=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
mkdir -p "$OUT"
{
  echo "# Machine"; echo
  (sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep 'Model name') | sed 's/^/- CPU: /'
  echo "- Cores: $T"; echo "- OS: $(uname -sr)"; echo "- Compiler: $(c++ --version | head -1)"
  echo "- libjpeg-turbo: $(ls -d /opt/homebrew/Cellar/jpeg-turbo/* 2>/dev/null | xargs -n1 basename || echo system)"
  echo; echo "# Latency (single image, median of $RUNS)"; echo
  "$BIN" --runs "$RUNS" --threads "1,2,4,$T" --json "$OUT/latency.json" $(ls "$DATA"/iss*.jpg)
  echo; echo "# Batch (24 Kodak images, one image per worker)"; echo
  "$BIN" --runs "$RUNS" --batch "$T" --json "$OUT/batch.json" $(ls "$DATA"/kodim*.jpg)
} | tee "$OUT/results.md"
