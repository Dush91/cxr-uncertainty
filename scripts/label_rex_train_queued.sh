#!/usr/bin/env bash
# Phase 6: label ReXGradient TRAIN reports with a WORKER-POOL QUEUE.
#
# WHY this exists (not the _parallel.sh all-at-once variant): the CheXpert labeler
# accumulates ~1MB/report in memory (it holds all reports + extracted mentions in a
# melted DataFrame). A 46k-report shard grew past the per-container memory cap and got
# OOM-killed at ~2400 reports (5%). So shards MUST be small (~2k reports) to stay under
# the cap, but small shards means MANY of them -- too many to launch all at once (RAM).
# This script shards into N small shards and runs them through a fixed-size worker pool
# (K concurrent containers), queuing the rest. `wait -n` releases a slot as each finishes.
#
# Flow:
#   1. label_rexgradient.py --splits train --input-only   (rex_reports_train.csv + index)
#   2. shard_csv.py --nshards $N                          (N small shards)
#   3. worker pool: K concurrent `docker run --memory=$MEM ... label.py`, queue the rest,
#      retry each shard once on failure.                  (N labeled shards)
#   4. cat labeled shards in order                        (rex_reports_train_labeled.csv)
#   5. label_rexgradient.py --splits train --parse-only   (rex_labels.parquet)
#
# Usage:
#   bash scripts/label_rex_train_queued.sh --nshards 70 --workers 4 [--memory 3000] \
#       [--out DIR] [--metadata-dir DIR] [--docker-image IMG] [--max-retry 1]
#
# Sizing: with ~1MB/report growth + ~950MB base, a 2000-report shard peaks ~3GB.
# --memory 3000 caps it safely; --nshards 70 -> 2000 reports each (140k/70).
# 4 workers x 3GB = 12GB < 14GB studio RAM. Total time ~ (N/workers) x per-shard-time.
# Environment: invoke via `bash -lc` (Lightning conda env); PYTHONPATH=. set below.
set -uo pipefail

NSHARDS=""
WORKERS=""
MEMORY=""
OUT=""
METADATA_DIR=""
DOCKER_IMAGE=""
MAX_RETRY="1"
PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nshards)      NSHARDS="$2"; shift 2;;
    --workers)      WORKERS="$2"; shift 2;;
    --memory)       MEMORY="$2"; shift 2;;
    --out)          OUT="$2"; shift 2;;
    --metadata-dir) METADATA_DIR="$2"; shift 2;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2;;
    --max-retry)    MAX_RETRY="$2"; shift 2;;
    --python)       PYTHON="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -z "$NSHARDS" ]] && NSHARDS=70
[[ -z "$WORKERS" ]] && WORKERS=4
[[ -z "$MEMORY" ]] && MEMORY=3000
[[ -z "$OUT" ]] && OUT="runs/rex_labeling_train_q"
[[ -z "$METADATA_DIR" ]] && METADATA_DIR="data/rexgradient/metadata"
[[ -z "$DOCKER_IMAGE" ]] && DOCKER_IMAGE="chexpert-labeler:latest"

SHARDS_DIR="$OUT/shards"
IN_CSV="$OUT/rex_reports_train.csv"
LABELED_CSV="$OUT/rex_reports_train_labeled.csv"

echo "[queued-labeler] nshards=$NSHARDS workers=$WORKERS memory=${MEMORY}m out=$OUT image=$DOCKER_IMAGE"

# 1. extract reports + write labeler input CSV + index sidecar (no labeler run)
echo "=== [1/5] extract train reports (--input-only) ==="
PYTHONPATH=. "$PYTHON" scripts/label_rexgradient.py \
    --splits train --input-only \
    --metadata-dir "$METADATA_DIR" --out "$OUT"

# 2. shard the input CSV (quote-safe) into N small shards
echo "=== [2/5] shard input into $NSHARDS quote-safe shards ==="
PYTHONPATH=. "$PYTHON" scripts/shard_csv.py \
    --input "$IN_CSV" --out-dir "$SHARDS_DIR" --nshards "$NSHARDS"

# docker bind-mounts need ABSOLUTE paths; SHARDS_DIR now exists.
SHARDS_DIR="$(cd "$SHARDS_DIR" && pwd)"

# 3. worker-pool queue: run K concurrent containers, release a slot per finish, retry on fail.
echo "=== [3/5] worker pool: $WORKERS concurrent labeler containers, $NSHARDS shards ==="
shard_files=( "$SHARDS_DIR"/shard_*.csv )
n=${#shard_files[@]}
echo "[queued-labeler] $n shards, pool size $WORKERS, per-container cap ${MEMORY}m"

run_one_shard() {
  local sf="$1" attempt="$2"
  local base; base="$(basename "$sf" .csv)"
  local out_shard="$SHARDS_DIR/${base}_labeled.csv"
  local log="$SHARDS_DIR/${base}.log"
  docker run --rm --memory="${MEMORY}m" --memory-swap="${MEMORY}m" \
      -v "$SHARDS_DIR":/data "$DOCKER_IMAGE" \
      python label.py \
      --reports_path "/data/$(basename "$sf")" \
      --output_path "/data/$(basename "$out_shard")" \
      --verbose > "$log" 2>&1
}

failed_shards=()
running=0
for sf in "${shard_files[@]}"; do
  base="$(basename "$sf" .csv)"
  echo "  [start] $base"
  run_one_shard "$sf" 0 &
  ((running++))
  if (( running >= WORKERS )); then
    # wait for any one job to finish; check its exit code
    wait -n; rc=$?
    ((running--))
    if (( rc != 0 )); then
      # identify which shard failed by scanning for missing labeled outputs of running set
      # (simplest: track via per-shard retry below) -- here just note a failure happened
      echo "  [warn] a shard exited non-zero (rc=$rc); will retry missing shards in pass 2"
    fi
  fi
done
# drain remaining
wait
echo "[queued-labeler] pass 1 done; checking for missing/failed shards"

# Retry pass: any shard without a non-empty labeled output gets retried up to MAX_RETRY.
for attempt in $(seq 1 "$MAX_RETRY"); do
  missing=()
  for sf in "${shard_files[@]}"; do
    base="$(basename "$sf" .csv)"
    out_shard="$SHARDS_DIR/${base}_labeled.csv"
    # a successful labeled shard has a header + >=2 lines
    if [[ ! -s "$out_shard" ]] || (( $(wc -l < "$out_shard") < 2 )); then
      missing+=( "$sf" )
    fi
  done
  nm=${#missing[@]}
  [[ $nm -eq 0 ]] && break
  echo "  [retry pass $attempt] $nm missing/failed shards"
  running=0
  for sf in "${missing[@]}"; do
    base="$(basename "$sf" .csv)"
    echo "  [retry $attempt] $base"
    run_one_shard "$sf" "$attempt" &
    ((running++))
    if (( running >= WORKERS )); then wait -n; ((running--)); fi
  done
  wait
done

# final check
fail=0
for sf in "${shard_files[@]}"; do
  base="$(basename "$sf" .csv)"
  out_shard="$SHARDS_DIR/${base}_labeled.csv"
  if [[ ! -s "$out_shard" ]] || (( $(wc -l < "$out_shard") < 2 )); then
    echo "  [FAIL] $base -- no labeled output (see $SHARDS_DIR/${base}.log)" >&2
    fail=1
  fi
done
[[ $fail -ne 0 ]] && { echo "[queued-labeler] one or more shards failed permanently" >&2; exit 1; }
echo "[queued-labeler] all $n shards labeled"

# 4. concat labeled shards IN ORDER -> rex_reports_train_labeled.csv (keep first header only)
echo "=== [4/5] concat labeled shards in order ==="
: > "$LABELED_CSV"
first=1
for sf in "${shard_files[@]}"; do
  base="$(basename "$sf" .csv)"
  if [[ $first -eq 1 ]]; then
    cat "$SHARDS_DIR/${base}_labeled.csv" >> "$LABELED_CSV"; first=0
  else
    tail -n +2 "$SHARDS_DIR/${base}_labeled.csv" >> "$LABELED_CSV"
  fi
done
echo "[queued-labeler] wrote $LABELED_CSV ($(wc -l < "$LABELED_CSV") lines)"

# 5. parse the concatenated labeled CSV -> rex_labels.parquet
echo "=== [5/5] parse labeled CSV -> rex_labels.parquet (--parse-only) ==="
PYTHONPATH=. "$PYTHON" scripts/label_rexgradient.py \
    --splits train --parse-only \
    --metadata-dir "$METADATA_DIR" --out "$OUT"

echo "[queued-labeler] DONE. train labels -> $OUT/rex_labels.parquet"
echo "  next: build_rex_manifest.py --split-train train --labels <valid+test parquet>,$OUT/rex_labels.parquet"