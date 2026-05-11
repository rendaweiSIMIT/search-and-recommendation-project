#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: exp/time-features ----
# Adds 1 NS token from user-level RELATIVE time features:
#   * recency_{a,b,c,d}: bucket of "time since most recent action" per
#     domain. 65 buckets (BUCKET_BOUNDARIES). 0 = no data.
#   * vel_1h_{a,b,c,d}: log-bucketed count of actions in last 1h. 8 buckets.
#   * vel_24h_{a,b,c,d}: same but 24h window.
# 12 categorical features summed into a single d_model NS token.
#
# user_ns_tokens 5 -> 4 to keep T = 16 (num_ns 4+1+2+1 = 8, T = 2*4+8 = 16,
# 64 % 16 = 0).
#
# --valid_gap_ratio 0.2 is THIS branch's honest-validation knob: train
# becomes the first 70% of RGs, the middle 20% is skipped, val is the last
# 10%. This forces val to generalize across a ~0.77-day gap, simulating
# "test is days into the future" -- a feature that only works because val
# is calendar-adjacent to train (like hour-of-day was) will now expose
# itself.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_recency_velocity \
    --valid_gap_ratio 0.2 \
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
