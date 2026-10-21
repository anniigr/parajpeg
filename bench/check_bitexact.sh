#!/usr/bin/env bash
# Byte-for-byte comparison of pjdec against libjpeg-turbo's djpeg (islow IDCT, fancy upsampling) for
# every JPEG given, e.g. the benchmark corpus:  bench/check_bitexact.sh bench/data/jpg/*.jpg
set -euo pipefail
PJDEC=${PJDEC:-build/pjdec}
DJPEG=${DJPEG:-$(command -v djpeg || echo /opt/homebrew/opt/jpeg-turbo/bin/djpeg)}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
for f in "$@"; do
  "$DJPEG" -dct int -outfile "$TMP/ref.ppm" "$f"
  for t in 1 8; do
    "$PJDEC" "$f" "$TMP/out.ppm" --threads "$t" 2>/dev/null
    if cmp -s "$TMP/ref.ppm" "$TMP/out.ppm"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "DIFFERS: $f (threads=$t)"; fi
  done
done
echo "$pass identical, $fail different ($# files x 2 thread counts, reference: $("$DJPEG" -version 2>&1 | head -1))"
[ "$fail" -eq 0 ]
