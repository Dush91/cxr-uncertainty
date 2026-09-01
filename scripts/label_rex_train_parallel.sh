#!/usr/bin/env bash
# Phase 6: label the ReXGradient TRAIN split (140k reports) with the CheXpert
# rule-based labeler in PARALLEL across N cores -- the long pole of the
# from-scratch D-Ens control, but CPU-only (run with the GPU OFF).
#
# The CheXpert labeler is single-core (NegBio + bllipparser + GENIA NLP), so
# the 140k train split is ~24h on one core. Sharding the input CSV across N
# cores and running one Docker container per shard cuts that to ~24/N hours
# (~3h on 8 cores). The shards are quote-safe (scripts/shard_csv.py uses the
# csv module, NOT `split -l`, which would corrupt multi-line quoted reports).
#
# Flow:
#   1. label_rexgradient.py --splits train --input-only
#        -> rex_reports_train.csv (headerless QUOTE_ALL) + rex_index_train.parquet
#   2. shard_csv.py --nshards $N
#        -> shard_000.csv .. shard_<N-1>.csv
#   3. N parallel `docker run chexpert-labeler label.py` (one per shard)
#        -> shard_000_labeled.csv .. shard_<N-1>_labeled.csv
#   4. cat shards in order -> rex_reports_train_labeled.csv
#   5. label_rexgradient.py --splits train --parse-only
#        -> rex_labels.parquet (feed to build_rex_manifest.py --split-train)
#
# Usage:
#   bash scripts/label_rex_train_parallel.sh [--nshards N] [--out DIR] \
#       [--metadata-dir DIR] [--docker-image IMG]
#
# Defaults: N = min(8, nproc); out = runs/rex_labeling_train;
#           metadata-dir = data/rexgradient/metadata; image = chexpert-labeler:latest
#
# Environment: PYTHONPATH=. (so the scripts import cxr_uncertainty). All python
# calls on Lightning MUST use `bash -lc` to activate the cloudspace conda env;
# this script itself should be invoked as `bash -lc scripts/label_rex_train_parallel.sh`.
set -euo pipefail

NSHARDS=""
OUT=""
METADATA_DIR=""
DOCKER_IMAGE=""
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nshards)        NSHARDS="$2"; shift 2;;
    --out)            OUT="$2"; shift 2;;
    --metadata-dir)   METADATA_DIR="$2"; shift 2;;
    --docker-image)   DOCKER_IMAGE="$2"; shift 2;;
    --python)         PYTHON="$2"; shift 2;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# defaults
[[ -z "$NSHARDS" ]] && NSHARDS="$(nproc 2>/dev/null || echo 8)"
NSHARDS="$(( NSHARDS > 8 ? 8 : NSHARDS ))"      # cap at 8 (diminishing returns + RAM)
[[ -z "$OUT" ]] && OUT="runs/rex_labeling_train"
[[ -z "$METADATA_DIR" ]] && METADATA_DIR="data/rexgradient/metadata"
[[ -z "$DOCKER_IMAGE" ]] && DOCKER_IMAGE="chexpert-labeler:latest"

SHARDS_DIR="$OUT/shards"
IN_CSV="$OUT/rex_reports_train.csv"
LABELED_CSV="$OUT/rex_reports_train_labeled.csv"

echo "[parallel-labeler] nshards=$NSHARDS out=$OUT metadata=$METADATA_DIR image=$DOCKER_IMAGE"

# 1. extract reports + write labeler input CSV + index sidecar (no labeler run)
echo "=== [1/5] extract train reports (--input-only) ==="
PYTHONPATH=. "$PYTHON" scripts/label_rexgradient.py \
    --splits train --input-only \
    --metadata-dir "$METADATA_DIR" --out "$OUT"

# 2. shard the input CSV (quote-safe)
echo "=== [2/5] shard input into $NSHARDS quote-safe shards ==="
PYTHONPATH=. "$PYTHON" scripts/shard_csv.py \
    --input "$IN_CSV" --out-dir "$SHARDS_DIR" --nshards "$NSHARDS"

# docker bind-mounts (-v) require ABSOLUTE paths; a relative path is rejected by
# dockerd. SHARDS_DIR now exists (shard_csv.py just created it), so resolve it.
SHARDS_DIR="$(cd "$SHARDS_DIR" && pwd)"

# 3. run one labeler container per shard, in parallel
echo "=== [3/5] run $NSHARDS parallel labeler containers ==="
shard_files=( "$SHARDS_DIR"/shard_*.csv )
n=${#shard_files[@]}
echo "[parallel-labeler] launching $n labeler containers ..."
# Measured per-JVM steady-state RSS is ~0.95GB (300-report sample). The earlier
# 4-shard studio crash was a CONCURRENT JVM STARTUP SPIKE (all bllipparser/CoreNLP
# models loading at once), not steady-state memory. Two safeguards:
#   --memory=3500m        caps any one JVM so a spike can't balloon (4x3.5=14GB
#                         only if ALL spike at once, which the stagger prevents);
#                         steady-state 4x0.95=3.8GB leaves ~10GB host headroom.
#   sleep 20 stagger      so the 4 JVM startup spikes don't all hit simultaneously.
pids=()
shard_i=0
for sf in "${shard_files[@]}"; do
  base="$(basename "$sf" .csv)"
  out_shard="$SHARDS_DIR/${base}_labeled.csv"
  [[ $shard_i -gt 0 ]] && sleep 20
  # mount the shards dir to /data; label.py reads/writes relative paths there.
  docker run --rm --memory=3500m --memory-swap=3500m \
      -v "$SHARDS_DIR":/data "$DOCKER_IMAGE" \
      python label.py \
      --reports_path "/data/$(basename "$sf")" \
      --output_path "/data/$(basename "$out_shard")" \
      --verbose > "$SHARDS_DIR/${base}.log" 2>&1 &
  pids+=( "$!" )
  echo "  [shard $base] pid $! -> $(basename "$out_shard")"
  shard_i=$((shard_i+1))
done

# wait for all; fail fast if any shard errors
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "  [FAIL] shard ${i} (pid ${pids[$i]}) -- see $SHARDS_DIR/shard_$(printf '%03d' "$i").log" >&2
    fail=1
  fi
done
[[ $fail -ne 0 ]] && { echo "[parallel-labeler] one or more shards failed" >&2; exit 1; }
echo "[parallel-labeler] all $n shards labeled"

# 4. concat labeled shards IN ORDER -> rex_reports_train_labeled.csv
#    Each shard's labeled CSV has a header row (["Reports"]+CATEGORIES); keep
#    only the FIRST header and strip it from the rest, so the concatenated
#    file is one valid labeled CSV (else the interleaved headers would break
#    label_rexgradient.py --parse-only's row-count assertion).
echo "=== [4/5] concat labeled shards in order ==="
: > "$LABELED_CSV"
first=1
for sf in "${shard_files[@]}"; do
  base="$(basename "$sf" .csv)"
  if [[ $first -eq 1 ]]; then
    cat "$SHARDS_DIR/${base}_labeled.csv" >> "$LABELED_CSV"
    first=0
  else
    tail -n +2 "$SHARDS_DIR/${base}_labeled.csv" >> "$LABELED_CSV"
  fi
done
echo "[parallel-labeler] wrote $LABELED_CSV ($(wc -l < "$LABELED_CSV") lines)"

# 5. parse the concatenated labeled CSV -> rex_labels.parquet
echo "=== [5/5] parse labeled CSV -> rex_labels.parquet (--parse-only) ==="
PYTHONPATH=. "$PYTHON" scripts/label_rexgradient.py \
    --splits train --parse-only \
    --metadata-dir "$METADATA_DIR" --out "$OUT"

echo "[parallel-labeler] DONE. train labels -> $OUT/rex_labels.parquet"
echo "  next: build_rex_manifest.py --split-train train --labels <valid+test parquet>,$OUT/rex_labels.parquet"