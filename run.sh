#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/delay-aux ----
# Conversion-delay auxiliary task (multi-task learning).
#
# Motivation (EDA-driven):
#   The `label_time` column carries when the final binary label was
#   determined. For positive samples (label_type == 2), label_time -
#   timestamp = the actual click-to-conversion latency; EDA on the full
#   training set showed:
#     - 1-D AUC of label_time alone vs binary label = 0.595
#       (signal = 0.190 -- STRONGEST single-feature signal in the whole
#       dataset, surpassing every NS feature, every seq fid, and every
#       pretrained dense column we've leveraged so far)
#     - Mean conversion delay for positives = 7456s (~2h), heavy tail
#       up to >86400s (~1 day), and ~99% of >86400s samples are positive
#   Negative samples have very short label_time deltas, positives have
#   long ones. So predicting log(label_time - timestamp) is a strong
#   side-task that forces the encoder to internalize "what features
#   correlate with quick vs slow vs no conversion".
#
# Why this avoids the time-feature trap we hit with hour-of-day:
#   - We never use `label_time` as a model INPUT; it is hidden in test
#     anyway (eval log shows label_time = 0 for every test row).
#   - The aux target is a DELTA (label_time - timestamp) not an absolute
#     clock value, so it carries no day-of-week / hour-of-day signal
#     subject to train -> test distribution shift.
#   - Loss is gated to positive samples only (~9.6% of batches); the
#     branch handles zero-positive batches gracefully via a mask + skip.
#
# Architecture (Alibaba ESMM family idea, scaled down):
#   pooled (B, D) ---> delay_head: Linear D -> 2D, SiLU, Dropout,
#                                  Linear 2D -> 1  ===> log_delay_pred
#   total_loss = BCE(logits, label)
#              + delay_aux_weight * MSE(log_delay_pred[pos], log_delay[pos])
#   delay_aux_weight defaults to 0.1; turn down/up via flag.
#
# valid_ratio = 0.03 follows the project convention
# (see feedback_kdd_tiny_recency_val): val = last ~3h of training,
# directly adjacent to the platform's Mon 00:01-01:30 test window,
# so ckpt selection optimises for the actual test distribution rather
# than the baseline's 10% positional val that bleeds into Sun afternoon.
#
# Orthogonal to mixed / hash / paired-pool / din-real / cross-arch /
# onetrans-suffix because the change touches only:
#   - dataset.py    : emits log_delay tensor (no schema change)
#   - model.py      : adds delay_head submodule + return_aux=True path
#   - trainer.py    : adds positive-masked MSE term to the BCE loss
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_delay_aux \
    --delay_aux_weight 0.1 \
    --delay_aux_hidden_mult 2 \
    --valid_ratio 0.03 \
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
