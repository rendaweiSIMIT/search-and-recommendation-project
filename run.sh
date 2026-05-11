#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation. Pure CUDA allocator
# change, no training-dynamics impact.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/senet-gating ----
# Add FibiNET-style SENet per-sample feature gating to the
# RankMixerNSTokenizer (both user and item sides). The gate is learned
# per-(sample, fid) from a 2-layer MLP over per-fid embedding summaries,
# so the model can dynamically decide "feature X matters more than Y
# FOR THIS SPECIFIC USER" instead of weighting every fid equally.
#
# Why this rather than switching to GroupNSTokenizer:
#   * GroupNSTokenizer forces num_queries=1 which breaks integration
#     with our 4 winning branches (all use num_queries=2). SENet keeps
#     num_queries=2, leaves the rankmixer chunking intact, and is fully
#     orthogonal to mixed / hash / paired / din.
#   * FibiNET production evidence (+0.005~+0.01 AUC) is stronger than
#     the "semantic vs random tokenization" debate.
#
# Param cost: ~1.3K dense params (negligible vs baseline 2.5M dense).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_senet_gating \
    --senet_reduction 4 \
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
