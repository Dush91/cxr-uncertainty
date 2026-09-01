#!/usr/bin/env bash
# Phase 6: AUTO-CHAIN the labeler to the /tmp extract.
# Polls rex_extract_totmp.log; the MOMENT the extract verifies (REXTRACT_DONE),
# cleans the failed prior labeler attempts and launches the queued labeler with
# full 4 workers (serial after extract = full RAM available, no CPU contention
# with the now-finished tar). If the extract FAILs, do NOT launch the labeler.
#
# Labeler sizing (post-extract, full RAM): --nshards 110 (~1273 reports/shard,
# peak ~2.1GB at ~1MB/report growth) --memory 2600m --workers 4.
# 4 x 2.6GB cap = 10.4GB max, actual ~8.4GB; ~3.6GB headroom on the 14GB studio
# (the earlier crash was 4x3.5GB UNCAPPED = 14GB; the caps prevent that).
# 110 shards / 4 workers = 28 batches x ~11.3min = ~5.2h.
set -uo pipefail
cd ~/content/rex_phase6 2>/dev/null || cd /teamspace/studios/this_studio/content/rex_phase6
LOG=rex_extract_totmp.log

echo "[chain] waiting for REXTRACT_DONE in $LOG ..."
for i in $(seq 1 240); do
  if grep -q "REXTRACT_DONE" "$LOG" 2>/dev/null; then
    echo "[chain] extract verified: $(grep REXTRACT_DONE "$LOG")"
    # 1. clean failed prior labeler attempts (free ~/content, reduce clutter)
    echo "[chain] cleaning failed prior runs ..."
    rm -rf runs/rex_labeling_train runs/rex_labeling_train2 runs/rex_labeling_train_3s 2>/dev/null
    # 2. verify labeler image present (not wiped by a restart)
    if ! docker images chexpert-labeler:latest --format '{{.Repository}}' 2>/dev/null | grep -q chexpert; then
      echo "[chain] labeler image missing -- reloading from ~/content/rex_phase6/chexpert_labeler.tgz"
      docker load < ~/content/rex_phase6/chexpert_labeler.tgz || { echo "[chain] FAIL no image; cannot label"; exit 1; }
    fi
    # 3. launch the queued labeler (background, detached)
    echo "[chain] launching queued labeler: nshards=110 workers=4 memory=2600"
    nohup bash -lc "cd ~/content/rex_phase6 && PYTHONPATH=. bash scripts/label_rex_train_queued.sh \
        --nshards 110 --workers 4 --memory 2600 --out runs/rex_labeling_train_q" \
        > runs/labeler_chain.log 2>&1 < /dev/null &
    echo "[chain] labeler launched, pid $! ; log=runs/labeler_chain.log"
    echo "CHAIN_LAUNCHED_LABELER pid=$!"
    exit 0
  fi
  # abort if the extract itself failed
  if grep -qi "FAIL" "$LOG" 2>/dev/null; then
    echo "[chain] extract FAILED -- NOT launching labeler:"
    tail -12 "$LOG"
    echo "CHAIN_ABORTED_EXTRACT_FAILED"
    exit 1
  fi
  sleep 60
done
echo "[chain] TIMEOUT waiting for extract (240 min); NOT launching labeler"
echo "CHAIN_TIMEOUT"
exit 1