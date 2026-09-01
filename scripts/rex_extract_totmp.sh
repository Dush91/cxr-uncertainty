#!/usr/bin/env bash
# Phase 6: re-extract ALL ReXGradient frontal PNGs to /tmp (outside the 200GB
# ~/content quota), then -- once verified -- free ~/content by deleting the
# 145GB tars + the old partial extract, and move the complete set back into
# ~/content (persistent) for the later L4 training phase.
#
# WHY /tmp: ~/content has a 200GB policy quota. tars(145G)+partial extract(48G)
# already ~= 193G, so the in-place extract ENOSPCs ~7GB before finishing.
# /tmp is a SEPARATE overlay mount (not under the ~/content quota), bounded only
# by the 148G physical free on /dev/vda1 -> fits the full 76G frontal set.
# The old 82k partial extract is KEPT until the /tmp extract is verified, as a
# fallback (tars survive too), so a mid-run failure is recoverable.
set -uo pipefail
cd ~/content/rex_phase6

FRONTAL="frontal_paths.txt"
TARG=129113

echo "[rextract] start $(date) cwd=$(pwd)"

# locate frontal_paths.txt
if [[ ! -f "$FRONTAL" ]]; then
  echo "[rextract] frontal_paths.txt not in cwd; searching..."
  FRONTAL="$(find . data -maxdepth 3 -name frontal_paths.txt 2>/dev/null | head -1)"
  [[ -z "$FRONTAL" ]] && { echo "[rextract] FAIL no frontal_paths.txt"; exit 1; }
fi
echo "[rextract] using frontal_paths=$FRONTAL ($(wc -l < "$FRONTAL") lines)"

# 1. stop the in-place extract that is about to ENOSPC
echo "[rextract] stopping in-place extract ..."
pkill -f 'tar -I zstd' 2>/dev/null
pkill -f 'cat data/rexgradient/deid_png.part' 2>/dev/null
sleep 3

# 2. re-extract ALL frontal PNGs to /tmp/rex (tars + old partial stay in ~/content)
rm -rf /tmp/rex
mkdir -p /tmp/rex
echo "[rextract] extracting all frontal to /tmp/rex (this reads the 145G stream, ~1.5-2h) ..."
cat data/rexgradient/deid_png.part* | tar -I zstd -xf - -C /tmp/rex -T "$FRONTAL"
rc=$?
echo "[rextract] tar rc=$rc"
if [[ $rc -ne 0 ]]; then
  echo "[rextract] FAIL tar rc=$rc -- tars + old partial preserved for retry"
  exit 1
fi

# 3. verify /tmp count
n=$(find /tmp/rex/deid_png -name '*.png' 2>/dev/null | wc -l)
echo "[rextract] /tmp count=$n (target $TARG)"
if [[ "$n" -lt "$TARG" ]]; then
  echo "[rextract] FAIL count $n < $TARG -- keeping tars for retry; /tmp left for inspection"
  exit 1
fi

# 4. /tmp verified complete -> safe to free ~/content: delete tars + old partial
echo "[rextract] deleting 145G tars + old partial extract from ~/content ..."
rm -f data/rexgradient/deid_png.part*
rm -rf data/rexgradient/deid_png

# 5. move the complete set from /tmp into ~/content (cross-device copy, ~76G)
echo "[rextract] copying /tmp/rex/deid_png -> ~/content/data/rexgradient/deid_png ..."
mkdir -p data/rexgradient
cp -a /tmp/rex/deid_png data/rexgradient/deid_png
rc=$?
if [[ $rc -ne 0 ]]; then echo "[rextract] FAIL cp rc=$rc (tars gone! /tmp copy intact at /tmp/rex)"; exit 1; fi
rm -rf /tmp/rex

# 6. final verify
n=$(find data/rexgradient/deid_png -name '*.png' 2>/dev/null | wc -l)
echo "[rextract] final ~/content count=$n"
if [[ "$n" -lt "$TARG" ]]; then echo "[rextract] FAIL final count $n < $TARG"; exit 1; fi

echo "[rextract] disk after:"; df -h ~/content /tmp 2>/dev/null | tail -2
echo "REXTRACT_DONE count=$n"
date