#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/full-train-4ep ----
# Baseline model (no architecture change) trained on 100% of the data
# for exactly 4 epochs, with no validation split and the final ckpt
# selected by convention (per feedback_kdd_fixed_4_epoch).
#
# Motivation:
#   Platform line-eval AUC has historically not been well predicted by
#   the training-time val AUC. The hour-shuffle-val run -- 0.8672 on
#   val vs 0.7946 on the platform -- was the breaking-point example.
#   Multiple runs have shown that picking the ckpt at the end of
#   epoch 4 (independent of val signal) is a reliable rule for this
#   competition's narrow test window (Mon 00:01-01:30 immediately
#   after the training data ends).
#
# Concretely:
#   --num_epochs 4   : stop after the 4th epoch regardless of val.
#   --valid_ratio 0  : disable validation entirely; train on every
#                      Row Group (100% of the data). dataset.py now
#                      returns valid_loader=None when valid_ratio<=0,
#                      and trainer.py skips evaluate()/EarlyStopping
#                      and saves one ckpt per epoch.
#
# Checkpoint layout produced by this run:
#   $TRAIN_CKPT_PATH/global_step{S1}.layer=2.head=4.hidden=64/
#   $TRAIN_CKPT_PATH/global_step{S2}.layer=2.head=4.hidden=64/
#   $TRAIN_CKPT_PATH/global_step{S3}.layer=2.head=4.hidden=64/
#   $TRAIN_CKPT_PATH/global_step{S4}.layer=2.head=4.hidden=64.best_model/
# Pick the .best_model directory on the Model Management page for eval.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --num_epochs 4 \
    --valid_ratio 0 \
    --seed 42 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
