#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/paired-pool-62-66 ----
# Same softmax-weighted (int, dense) pooling as exp/paired-pool but applied
# to fids 62-66 ONLY; fids 89-91 fall back to baseline mean pool.
#
# Why drop 89-91:
#   89-91 carry already-normalized scores in [-0.92, 0.92]. After the
#   tokenizer's signed_log1p compression the per-position weights are
#   essentially equal-length, so softmax collapses to a near-uniform
#   distribution -- the softmax pool degenerates into mean pool with
#   extra parameters. Skipping them removes that no-op overhead and
#   isolates the lift to the 5 raw-counter columns (max ~132M) where
#   the weight range is wide enough for softmax to actually steer the
#   user representation.
#
# The fid restriction is set via --paired_pool_fids (default in train.py
# is now '62,63,64,65,66'); no other config changes vs paired-pool.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --paired_pool_fids 62,63,64,65,66 \
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
