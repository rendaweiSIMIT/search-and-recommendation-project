#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/inter-cross-arch ----
# InterFormer-style sequence -> non-sequence (s2n) feedback.
#
# Motivation (from Meta CIKM 2025 paper):
#   - Baseline architecture has unidirectional info flow: NS guides Q
#     which attends seq (n2s direction). The reverse path (seq summary
#     informs NS) is completely missing.
#   - InterFormer ablation (sole < sep < n2s ≈ s2n < int) shows that
#     bidirectional info flow consistently beats unidirectional.
#
# Implementation:
#   - After all HyFormer blocks run, take the EVOLVED seq tokens per
#     domain and produce two summary vectors per domain:
#       * CLS-pool: a learnable query attends over evolved seq tokens
#       * Recent-K mean: mean of the K most-recent valid positions
#         (sequences are descending-sorted, K=8 by default)
#   - Concat the 2*num_domains summaries -> 2-layer MLP -> self-gating
#     -> LayerNorm -> (B, d_model) residual.
#   - Add the residual to the pooled output BEFORE the classifier.
#
# Orthogonality:
#   - Pure additive residual: model can learn the gate to 0 if useless
#     (downside-bounded; recovers baseline behavior).
#   - Does not touch T, num_queries, NS layout, embedding tables.
#   - Stacks cleanly with mixed / hash / paired / din / senet for final
#     integration round (din is the n2s direction, this is s2n -> the
#     two together = full InterFormer "int" mode).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_cross_arch \
    --cross_arch_recent_k 8 \
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
