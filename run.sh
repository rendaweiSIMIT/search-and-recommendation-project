#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation. Pure CUDA allocator
# change, no training-dynamics impact.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/din-style ----
# Add DIN-style item-aware cross-attention as a parallel residual head:
#   * 4 CrossAttention modules (one per sequence domain), each takes
#     item-side NS tokens as queries and the RAW seq tokens of that
#     domain as keys/values.
#   * Outputs are mean-pooled per domain, concatenated, and projected
#     back to d_model via a Linear+LN -> ADDED as residual to the pooled
#     output right before the classifier.
#   * Pure additive: when this path learns weight 0 the model recovers
#     baseline behavior, so the experiment is downside-bounded.
#
# Why this hypothesis: PCVRHyFormer's current queries mix user AND item
# features then attend over seq. DIN evidence (Alibaba 2018, KDD) says
# making queries item-only is a strong inductive bias that consistently
# beats mixed-query in CTR/CVR -- production AUC gains 0.005-0.015.
# Our existing architecture can only approximate this through indirect
# learning; this branch makes it explicit.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_din_attention \
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
