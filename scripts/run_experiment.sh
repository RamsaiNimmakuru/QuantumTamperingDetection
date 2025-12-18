#!/usr/bin/env bash
EXP_DIR=results/experiments/$(date +%Y%m%d_%H%M%S)
mkdir -p ${EXP_DIR}
cp config_* ${EXP_DIR}/ 2>/dev/null || true
echo "Saving run metadata to ${EXP_DIR}"
python3 scripts/train_multistream_prototype.py --data-dir data --batch-size 8 --epochs 6 --lr 1e-4 --unfreeze-epoch 3 | tee ${EXP_DIR}/train_log.txt
cp models/multistream_best.pt ${EXP_DIR}/ || true
cp results/*.png ${EXP_DIR}/ 2>/dev/null || true
cp results/*.json ${EXP_DIR}/ 2>/dev/null || true
echo "Run finished. Results saved to ${EXP_DIR}"
