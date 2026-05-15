#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- exp/hour01-finetune: two-stage train + target-distribution finetune ----
#
# Motivation:
#   Test set is restricted to Mon 00:01-01:30 (~90 minutes), but train spans
#   ~4.6 days dominated by hour 22-23. Every sample-level absolute-time
#   feature we have tried lifts val but drops platform AUC — classic
#   train/test distribution shift.
#
#   Standard fix from the domain-adaptation literature: pretrain on the
#   broad source distribution, then in-domain finetune on the subset whose
#   time-of-day matches the target distribution.
#
# Stage 1 (full-data pretrain, baseline config):
#   - Train on every row (no time filter), val_ratio=0.1, EarlyStopping
#   - Produces a ``*.best_model`` ckpt under TRAIN_CKPT_PATH
#
# Stage 2 (target-distribution finetune):
#   - Load Stage 1's best ckpt via --finetune_from
#   - Filter both train and val to seconds-of-day in [60, 5400]
#     (= 00:01 to 01:30 inclusive), matching the test window
#   - 10x smaller dense+sparse LR to avoid blowing up the pretrained features
#   - --save_every_epoch: every finetune epoch's weights are persisted under
#     epoch{N}.layer=X.head=Y.hidden=Z/, so we never lose a candidate ckpt
#     even if its val score underperforms an earlier finetune epoch.
#     (val on a tiny 00:01-01:30 valid slice is noisy and may mis-rank
#     finetune epochs vs. their true platform score.)
#
# Manual ckpt picking:
#   The platform UI auto-picks ``*.best_model``. If val rejects all
#   finetune epochs, the *.best_model from Stage 1 is preserved as the
#   submission default. To submit a specific finetune epoch instead,
#   rename ``epoch{N}.layer=...`` -> ``epoch{N}.layer=....best_model``
#   before clicking submit.
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

# ---- Locate Stage 1's best ckpt before launching Stage 2 ----
STAGE1_BEST_DIR=$(ls -dt "${TRAIN_CKPT_PATH}"/global_step*.best_model 2>/dev/null | head -1)
if [ -z "${STAGE1_BEST_DIR}" ] || [ ! -f "${STAGE1_BEST_DIR}/model.pt" ]; then
    echo "[Stage 2] ERROR: no *.best_model/model.pt under ${TRAIN_CKPT_PATH} — aborting finetune."
    exit 1
fi
echo "[Stage 2] Finetune starting from: ${STAGE1_BEST_DIR}/model.pt"

# ---- Stage 2: finetune on 00:01-01:30 only ----
# --reinit_sparse_after_epoch 999: trainer baseline resets high-cardinality
# Embeddings at the end of every epoch (cold-restart trick); for finetune
# this would wipe the pretrained sparse weights we just loaded — disable.
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
    --hour_minute_filter "0:01-1:30" \
    --finetune_from "${STAGE1_BEST_DIR}/model.pt" \
    --save_every_epoch
