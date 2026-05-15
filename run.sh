#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- exp/hour-window-finetune: 2-stage target-distribution finetune ----
#
# Motivation:
#   EDA on the eval set (HANDOFF §1.6, raw evallog.txt) confirmed the
#   test time window is Sun 23:40:35 -> Mon 01:13:44 (93 minutes,
#   cross-midnight). sod (seconds-of-day, computed as ts % 86400)
#   range: [85235, 86399] U [0, 4424].
#
#   Earlier hour01-finetune used "0:01-1:30" (sod [60, 5400]) which:
#     - dropped the entire Sun 23:40-23:59 segment (20.9% of test)
#     - covered Mon 01:13-01:30 where test has 0 rows.
#   This branch fixes the filter to match the EDA-confirmed window
#   exactly via spec "23:40-1:13", which dataset.PCVRParquetDataset
#   interprets as cross-midnight: keep rows with sod in [85235, 86399]
#   or sod in [0, 4424].
#
# Stage 1 (full-data pretrain, baseline config):
#   - Train on every row (no time filter), val_ratio=0.1, EarlyStopping.
#   - Produces a ``*.best_model`` ckpt under TRAIN_CKPT_PATH.
#
# Stage 2 (target-distribution finetune):
#   - Load Stage 1's best ckpt via --finetune_from.
#   - Filter both train and val to sod in [85235, 86399] U [0, 4424]
#     via --hour_minute_filter "23:40-1:13".
#   - 10x smaller dense+sparse LR to avoid blowing up pretrained features.
#   - --reinit_sparse_after_epoch 999 to keep the loaded sparse
#     Embeddings intact (baseline trainer resets them per epoch).
#   - --save_every_epoch: each finetune epoch's weights persist under
#     epoch{N}.layer=X.head=Y.hidden=Z/ regardless of val outcome.
#     val on a tiny filtered subset is noisy and may mis-rank epochs
#     versus their true platform score, so we keep every candidate.
#
# Stage 1: baseline full-train. All five rankmixer flags preserved
# verbatim from main run.sh (any drift here would silently change the
# architecture and confound the finetune outcome).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --num_epochs 999 \
    --valid_ratio 0.1 \
    --patience 5 \
    --seed 42 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# Locate Stage 1's *.best_model ckpt before Stage 2.
STAGE1_BEST_DIR=$(ls -dt "${TRAIN_CKPT_PATH}"/global_step*.best_model 2>/dev/null | head -1)
if [ -z "${STAGE1_BEST_DIR}" ] || [ ! -f "${STAGE1_BEST_DIR}/model.pt" ]; then
    echo "[Stage 2] ERROR: no *.best_model/model.pt under ${TRAIN_CKPT_PATH} — aborting finetune."
    exit 1
fi
echo "[Stage 2] Finetune starting from: ${STAGE1_BEST_DIR}/model.pt"

# Stage 2: finetune on the EDA-confirmed test window only. Same five
# rankmixer flags as Stage 1 so the ckpt loads strict=True without
# shape mismatch.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --num_epochs 6 \
    --valid_ratio 0.1 \
    --patience 99 \
    --seed 42 \
    --lr 1e-5 \
    --sparse_lr 0.005 \
    --reinit_sparse_after_epoch 999 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --hour_minute_filter "23:40-1:13" \
    --finetune_from "${STAGE1_BEST_DIR}/model.pt" \
    --save_every_epoch
