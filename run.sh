#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/onetrans-suffix ----
# Add a OneTrans-style (ByteDance WWW 2026) causal pyramid attention
# block as a post-HyFormer enrichment head. The block treats the
# concatenated [evolved_seq_a | seq_b | seq_c | seq_d | NS_tokens]
# as a unified causal sequence and applies multi-head attention with
# MIXED PARAMETERIZATION:
#   * S-tokens (sequential): share one set of K/V projections + (no Q)
#   * NS-tokens (heterogeneous): each gets its own QKV + own FFN
#
# Pyramid: only NS positions issue queries -> attention cost is
# O(num_ns * L_total * d) instead of O(L_total^2), bounded even with
# ~1.5K-long S token stream.
#
# Output: the enriched NS tokens (B, num_ns, d_model) are pooled by a
# concat -> Linear -> LayerNorm into a (B, d_model) residual added to
# the pooled output before the classifier. Pure additive -> downside-
# bounded (pool projection can learn to zero out the path).
#
# Why this is worth a slot:
#   * OneTrans paper validated +1.5% AUC in industrial production by
#     unifying sequence modeling and feature interaction in one
#     causal Transformer. We adopt the architecture's core innovation
#     (mixed parameterization + unified causal attention) as an
#     enrichment head, capturing most of the lift while keeping the
#     existing PCVRHyFormer skeleton intact.
#   * Fully orthogonal to mixed / hash / paired / din / senet /
#     cross-arch / bpr -- final integration round can stack them.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_onetrans_suffix \
    --onetrans_hidden_mult 4 \
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
