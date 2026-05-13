#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/paired-pool-89-91 ----
# Same softmax-weighted (int, dense) pooling as exp/paired-pool but applied
# to fids 89-91 ONLY; fids 62-66 fall back to baseline mean pool.
#
# Background -- what previous paired-pool runs taught us:
#   exp/paired-pool       (8 fids: 62-66 + 89-91): +0.0010 vs baseline.
#   exp/paired-pool-62-66 (5 fids: 62-66 only):     BELOW baseline.
#   => the 89-91 subset is responsible for the positive lift; the 62-66
#      subset is a net-negative contribution that drags the 8-fid winner
#      down to +0.0010. Subtracting the two:
#        only-89-91 lift  =  +0.0010 -  (-Y from 62-66)  >  +0.0010.
#
# Why 62-66 hurts when 89-91 helps (both math + empirical):
#   62-66 dense values are raw counters with max ~1.3e9. After signed_log1p
#   the range collapses to [0, ~21], over which softmax exp(21) / exp(0)
#   is ~1.3e9 -- effectively argmax. The pooled vector is dominated by
#   one category, the rest are zeroed out. That throws away information
#   the baseline mean pool would have kept.
#   89-91 dense values are already in [-0.92, 0.92]. signed_log1p maps
#   them to [-0.65, 0.65]. softmax over that range has ~3.7x weight
#   ratio -- a healthy soft-attention spread, which is what the original
#   paired-pool concept ("by score weighted pool") was after.
#
# Other config: --valid_ratio 0.03 follows the project-wide convention
# (see feedback_kdd_tiny_recency_val): val = last ~3h of training =
# directly adjacent to the platform's Mon 00:01-01:30 test window. The
# baseline 0.10 val (~11h) bleeds into Sun afternoon traffic that does
# not look like Mon early-morning test, so a smaller late-only val is
# a strictly better ckpt-selection signal for this competition.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --paired_pool_fids 89,90,91 \
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
