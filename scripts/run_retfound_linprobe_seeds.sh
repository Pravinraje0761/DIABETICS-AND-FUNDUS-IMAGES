#!/usr/bin/env bash
# Background driver: RETFound linear-probe seed sweep (seeds 43/44/45).
# Frozen ViT-L backbone, only the head trains -> cheap enough for the laptop.
# Checkpoints go to C:\dr_ckpts_seeds (outside OneDrive) to avoid sync stalls;
# small logs (_history.csv, _summary.json, _internal_test_*) land in
# outputs/phase5_retfound/ with a seedNN tag. Primary linprobe_best.pt untouched.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
CKPTS='C:\dr_ckpts_seeds'
LOGDIR=logs/seeds
mkdir -p "$LOGDIR"
export DR_DATA_ROOT='C:\dr_run'

for s in 43 44 45; do
  echo "===== RETFound linprobe seed $s  $(date) =====" | tee -a "$LOGDIR/_driver.log"
  "$PY" -u scripts/train_retfound.py --mode linprobe \
      --weights weights/RETFound_mae_natureCFP.pth \
      --batch-size 16 --num-workers 2 --seed "$s" \
      --ckpt-dir "$CKPTS" --tag "seed$s" \
      > "$LOGDIR/rf_linprobe_seed$s.log" 2>&1
  echo "  seed $s exit $?" | tee -a "$LOGDIR/_driver.log"
done
echo "ALL LINPROBE SEEDS DONE $(date)" | tee -a "$LOGDIR/_driver.log"
