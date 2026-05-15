#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- exp/user-time-encoding: sample-level (hour, weekday, sin/cos) as USER feature ----
# Adds an additive residual derived from the impression `timestamp` to every
# user NS token. No change to seq-side tokens. The sin/cos cyclical projection
# is the key piece: dow=0 (Mon, train-unseen) lies adjacent to dow=6 (Sun,
# train-dominant) on the 7-period unit circle, so the trained Linear(4, D)
# generalizes from Sun's heptagon-vertex to Mon's adjacent vertex — something
# a discrete dow embedding cannot do. See model.py PCVRHyFormer `__init__`
# and `_make_sample_time_residual` for the recipe.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_user_time_encoding \
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
