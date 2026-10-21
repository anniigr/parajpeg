#!/usr/bin/env bash
# Builds the benchmark corpus from public-domain NASA photographs (ISS crew Earth observations) and
# the Kodak Lossless True Color Image Suite. Downloads go to $DATA/src (untouched), re-encodings to
# $DATA/jpg. Requires curl and libjpeg-turbo's cjpeg/djpeg.
set -euo pipefail
DATA=${1:-bench/data}
CJPEG=${CJPEG:-$(command -v cjpeg || echo /opt/homebrew/opt/jpeg-turbo/bin/cjpeg)}
DJPEG=${DJPEG:-$(command -v djpeg || echo /opt/homebrew/opt/jpeg-turbo/bin/djpeg)}
mkdir -p "$DATA/src" "$DATA/ppm" "$DATA/jpg"

NASA_IDS="iss062e001866 iss063e062018 iss069-e-089946_lrg iss074e0458251"
for id in $NASA_IDS; do
  f="$DATA/src/$id.jpg"
  [ -s "$f" ] || curl -sL --fail -m 300 -o "$f" "https://images-assets.nasa.gov/image/$id/$id~orig.jpg"
  cp "$f" "$DATA/jpg/${id}_camera.jpg"           # the camera original, as delivered
  "$DJPEG" -outfile "$DATA/ppm/$id.ppm" "$f"
done
for i in $(seq -w 1 24); do
  f="$DATA/src/kodim$i.png"
  [ -s "$f" ] || curl -sL --fail -m 60 -o "$f" "https://r0k.us/graphics/kodak/kodak/kodim$i.png"
done

enc() { "$CJPEG" "$@"; }
for id in $NASA_IDS; do
  p="$DATA/ppm/$id.ppm"
  enc -quality 90 -sample 2x2 -optimize          -outfile "$DATA/jpg/${id}_q90_420.jpg" "$p"
  enc -quality 90 -sample 1x1 -optimize          -outfile "$DATA/jpg/${id}_q90_444.jpg" "$p"
  enc -quality 75 -sample 2x2 -optimize          -outfile "$DATA/jpg/${id}_q75_420.jpg" "$p"
  enc -quality 85 -sample 2x2 -restart 1         -outfile "$DATA/jpg/${id}_q85_420_rst.jpg" "$p"
  enc -quality 85 -progressive                   -outfile "$DATA/jpg/${id}_q85_prog.jpg" "$p"
done
# Kodak: small images (768x512). cjpeg reads PNG only in some builds; otherwise convert with ImageMagick.
if "$CJPEG" -help 2>&1 | grep -q "PNG"; then
  for i in $(seq -w 1 24); do enc -quality 90 -sample 2x2 -optimize -outfile "$DATA/jpg/kodim${i}_q90_420.jpg" "$DATA/src/kodim$i.png"; done
elif command -v magick >/dev/null; then
  for i in $(seq -w 1 24); do magick "$DATA/src/kodim$i.png" "$DATA/ppm/kodim$i.ppm"; enc -quality 90 -sample 2x2 -optimize -outfile "$DATA/jpg/kodim${i}_q90_420.jpg" "$DATA/ppm/kodim$i.ppm"; done
fi
ls -la "$DATA/jpg" | awk '{s+=$5} END {print NR-3 " files, " s/1e6 " MB"}'
