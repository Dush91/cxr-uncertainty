#!/bin/zsh
# Parallel chunked upload of the remaining train-cache bytes to Lightning.
# Resumes from whatever the remote partial already holds; verifies end to end.
# One-off infrastructure script (DUA cleanup follows separately).
set -e
SRC="/Users/dushyanttyagi/Desktop/MSc AI- CourseWork/data/adapted_cache_train.npy"
KEY="/Users/dushyanttyagi/.ssh/lightning_rsa"
H="s_01kzvzcj31a4jjy78ey4bx00dr@ssh.lightning.ai"
R="/teamspace/studios/this_studio/cxr_repo/data"
REMOTE="$R/adapted_cache_train.npy"

# 0. stop the single-stream append upload
pkill -f "rsync --append" 2>/dev/null || true
sleep 2

# 1. how many bytes are already on the host?
S=$(ssh -i $KEY $H "bash -lc 'stat -c%s $REMOTE'" 2>/dev/null | tail -1)
echo "byte frontier: $S"

# 2. remaining bytes -> 512MB parts, locally (tail -c works for any S)
rm -rf /tmp/cxrparts /tmp/cxrparts_local
mkdir -p /tmp/cxrparts
tail -c +$((S+1)) "$SRC" > /tmp/cxrparts_local
split -b 536870912 /tmp/cxrparts_local /tmp/cxrparts/part_
rm /tmp/cxrparts_local
N=$(ls /tmp/cxrparts | wc -l)
echo "split into $N parts ($(du -sh /tmp/cxrparts | cut -f1))"

# 3. parallel upload, 6 streams, 5 retries per part
ssh -i $KEY $H "bash -lc 'mkdir -p $R/parts'"
ls /tmp/cxrparts/part_* | xargs -P 6 -I{} sh -c '
  f="{}"; b=$(basename "$f")
  for i in 1 2 3 4 5; do
    rsync -e "ssh -i '"$KEY"'" "$f" "'"$H:$R/parts/"'"/ && { echo "UP $b"; exit 0; }
    sleep 5
  done
  echo "FAILED $b"; exit 1'
echo "PARTS_UPLOAD_DONE"

# 4. remote concat + cleanup (+ disk headroom check)
ssh -i $KEY $H "bash -lc 'cd $R && df -h . | tail -1 && cat adapted_cache_train.npy $(ls parts/* | sort) > full_new.npy && rm -rf parts && mv full_new.npy adapted_cache_train.npy && stat -c%s adapted_cache_train.npy'"
echo "CONCAT_DONE"

# 5. verify: local sha256 vs remote
L=$(shasum -a 256 "$SRC" | cut -d" " -f1)
Rm=$(ssh -i $KEY $H "bash -lc 'sha256sum $REMOTE'" 2>/dev/null | cut -d" " -f1)
echo "sha256 local : $L"
echo "sha256 remote: $Rm"
[ "$L" = "$Rm" ] && echo TRAIN_CACHE_UPLOAD_DONE || echo TRAIN_CACHE_UPLOAD_CHECKSUM_MISMATCH