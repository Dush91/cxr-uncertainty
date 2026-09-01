#!/usr/bin/env bash
# Phase 7 Stage 1: GPU orchestrator (run on Lightning from ~/content/rex_phase7).
# Trains the adapted members (convnextv2 x3 seeds), then rebuilds eval arrays +
# retrieval index + baselines under adapted paths. Assumes code + checkpoints +
# 224 fp16 caches + png768 subset + cal/eval PNGs already under data/.
# All python calls go through bash -lc per studio rules.
set -uo pipefail
cd ~/content/rex_phase7
mkdir -p logs

echo "[runner] start $(date); GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "$(nvidia-smi -L 2>/dev/null | wc -l)" -eq 0 ]]; then
  echo "[runner] NO GPU visible -- abort"; exit 1
fi

RUN224="--train-cache data/adapted_cache_train.npy --cal-cache data/adapted_cache_cal.npy"
run224() {                          # skip if this member's checkpoint already exists
  local member="" seed=0 name=""
  local args=("$@")                 # preserve originals -- the parse loop shifts them away
  while [[ $# -gt 0 ]]; do
    case "$1" in --member) member="$2";; --seed) seed="$2";; esac; shift
  done
  name="$member"; [[ "$member" == "convnextv2" ]] && name="convnextv2_s${seed}"
  if [[ -s "checkpoints/rex_adapted/${name}.pt" ]]; then
    echo "[skip] checkpoints/rex_adapted/${name}.pt already trained"; return 0
  fi
  bash -lc "PYTHONPATH=. python scripts/train_rex_adapted.py $RUN224 ${args[*]}" \
        2>&1 | tee -a logs/train.log
}

# ---- per-member training (Route A; arkswin = linear probe @768 on PNG subset) ----
run224 --member convnextv2 --seed 0 || exit 1
run224 --member convnextv2 --seed 1 || exit 1
run224 --member convnextv2 --seed 2 || exit 1
run224 --member xrv_nih   || exit 1
run224 --member raddino   || exit 1
run224 --member arkswin   || exit 1

CK=checkpoints/rex_adapted

arr() { bash -lc "PYTHONPATH=. python scripts/build_eval_arrays.py \
    --manifest data/rex_manifest_fs.parquet --source rex_adapted \
    --members xrv_nih,convnextv2,raddino,arkswin --batch-size 8 \
    --ckpt-overlay xrv_nih=$CK/xrv_nih.pt,convnextv2=$1,raddino=$CK/raddino.pt,arkswin=$CK/arkswin.pt \
    --out runs/eval_arrays 2>&1 | tee -a logs/runner.log" || exit 1; }

# ---- s0 headline arrays: build, then retrieval index reads them IN PLACE ----
arr $CK/convnextv2_s0.pt
bash -lc "PYTHONPATH=. python scripts/build_retrieval_index.py \
    --arr-dir runs/eval_arrays --source rex_adapted \
    --embed raddino --out runs/retrieval/rex_adapted_raddino" \
    2>&1 | tee -a logs/runner.log || exit 1
mv runs/eval_arrays/rex_adapted runs/eval_arrays/rex_adapted_s0

# ---- seed-variance arrays (s1, s2 convnextv2) into their own dirs ----
arr $CK/convnextv2_s1.pt
mv runs/eval_arrays/rex_adapted runs/eval_arrays/rex_adapted_s1
arr $CK/convnextv2_s2.pt
mv runs/eval_arrays/rex_adapted runs/eval_arrays/rex_adapted_s2

# ---- baselines per seed (AUAC/E-AURC, tail std-vs-conf, DeLong) ----
for S in s0 s1 s2; do
  bash -lc "PYTHONPATH=. python scripts/eval_baselines.py \
    --datasets rex_adapted_$S:runs/eval_arrays/rex_adapted_$S/arrays_rex_adaptedcal.npz:runs/eval_arrays/rex_adapted_$S/arrays_rex_adaptedeval.npz \
    --out runs/baselines/rex_adapted_$S" 2>&1 | tee -a logs/runner.log || exit 1
done

echo "STAGE1_DONE $(date)"