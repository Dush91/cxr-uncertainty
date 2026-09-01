#!/usr/bin/env bash
# Phase 6: train 15 from-scratch D-Ens models (3 backbones x 5 seeds) on the L4.
# Baur et al. protocol: 5 independent random-init trainings per backbone.
# Sequential (one GPU); resumable -- skips any checkpoint already present.
#
# Backbones: resnet18 (torchvision, 512-d), vit_tiny + convnext_tiny (timm).
# AMP, bs256, 224px, ~25 epochs w/ early-stop on ReX-valid (split_role="cal").
# Checkpoints -> checkpoints/fs_<backbone>_s<seed>.pt (pulled to Mac after; DUA).
#
# Usage:  bash scripts/train_fs_all.sh [--epochs 25] [--bs 256]
# Env:    EPOCHS=.. BS=..  overrides.  Run via `bash -lc` on the L4 studio.
set -uo pipefail
cd ~/content/rex_phase6 2>/dev/null || cd /teamspace/studios/this_studio/content/rex_phase6

MANIFEST=data/rex_manifest_fs.parquet
CKPT_DIR=checkpoints
mkdir -p "$CKPT_DIR"
EPOCHS="${EPOCHS:-25}"
BS="${BS:-256}"
BACKBONES=(resnet18 vit_tiny convnext_tiny)
SEEDS=(0 1 2 3 4)

echo "[fs-all] start $(date) epochs=$EPOCHS bs=$BS manifest=$MANIFEST"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "[fs-all] WARNING: no GPU detected"

total=0; done=0
for bb in "${BACKBONES[@]}"; do for s in "${SEEDS[@]}"; do total=$((total+1)); done; done

for bb in "${BACKBONES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$CKPT_DIR/fs_${bb}_s${seed}.pt"
    log="$CKPT_DIR/fs_${bb}_s${seed}.log"
    if [ -f "$out" ]; then
      echo "[skip] $out already exists (size $(du -h "$out" | cut -f1))"; done=$((done+1)); continue
    fi
    echo "[train $((done+1))/$total] $bb seed=$seed -> $out  $(date)"
    # use the pre-decoded memmap cache if present (eliminates per-epoch PNG decode;
    # built by scripts/build_train_cache.py). Falls back to decode if absent.
    CACHE_ARGS=""
    if [ -f data/fs_cache_train.npy ] && [ -f data/fs_cache_val.npy ]; then
      CACHE_ARGS="--cache-train data/fs_cache_train.npy --cache-val data/fs_cache_val.npy"
    fi
    PYTHONPATH=. python scripts/train_fromscratch.py \
        --backbone "$bb" --seed "$seed" \
        --manifest "$MANIFEST" --train-role train --val-role cal \
        --epochs "$EPOCHS" --bs "$BS" --amp $CACHE_ARGS --out "$out" > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[FAIL] $bb s$seed rc=$rc -- see $log"; tail -25 "$log"
    else
      done=$((done+1))
      echo "[ok] $bb s$seed -> $(du -h "$out" | cut -f1)  ($(grep -c 'epoch' "$log" 2>/dev/null) lines) $(date)"
    fi
  done
done

echo "[fs-all] DONE $done/$total models trained  $(date)"
ls -la "$CKPT_DIR"/fs_*.pt 2>/dev/null
echo "ALL_TRAIN_DONE"