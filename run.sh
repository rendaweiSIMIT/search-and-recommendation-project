#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/event-hour-seq ----
# Per-event hour-of-day enrichment on seq tokens (Path A from the time-
# feature discussion). The key insight: previous absolute-hour attempts
# (exp/hour-of-day -0.014, exp/hour-shuffle-val -0.017) failed because
# the sample-side hour-of-day has heavy train/test distribution shift
# (test = Mon 00-01, train mostly 22-23). The event-side hour-of-day
# avoids that trap entirely because the user's past events were
# collected in the SAME time range as the training observation window,
# so the per-event hour distribution looks identical in train and test.
# Models the user's lifestyle / activity-time pattern, which is a stable
# user-level trait independent of when the candidate impression happens.
#
# Concretely:
#   dataset.py adds ``{domain}_event_hour`` per seq token (0=pad,
#     1-24 = hour 0-23 of the event's own timestamp).
#   model.py adds a 25-class Embedding (padding_idx=0) added to each
#     seq token alongside the existing time_bucket embedding inside
#     _embed_seq_domain. No NS token added -> T constraint unchanged.
#
# Trained on 100% of the data for 4 epochs, no validation, final ckpt
# auto-marked .best_model (per feedback_kdd_fixed_4_epoch). Earlier
# epochs are retained for rollback.
#
# Checkpoint layout:
#   $TRAIN_CKPT_PATH/global_step{S1}.layer=2.head=4.hidden=64/
#   ...
#   $TRAIN_CKPT_PATH/global_step{S4}.layer=2.head=4.hidden=64.best_model/
# Pick the .best_model directory on the Model Management page for eval.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_event_hour \
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
