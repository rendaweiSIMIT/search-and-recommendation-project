#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/din-real ----
# Real DIN + MLP head (Alibaba KDD 2018 architecture). Re-attempt of
# exp/din-style with three substantive fixes:
#
#   1. MLP-scored attention (not softmax dot-product). Per the DIN paper,
#      the scoring function is an MLP that consumes [q, k, q-k, q*k] for
#      each (query, key) pair. The 4-way concat captures non-linear
#      query/key interactions that a single dot product cannot, which is
#      exactly the inductive bias DIN's authors argued for.
#
#   2. SINGLE candidate query projected from all item NS tokens, instead
#      of multiple item NS tokens each used as a separate query with
#      mean-pool merging. A single candidate is closer to DIN's "one ad
#      vs all your history" formulation; the projection step lets the
#      head learn what to emphasise among item NS features.
#
#   3. Fusion via 2-layer MLP over [hyformer_pooled, din_pooled] that
#      REPLACES the pooled output (not additive residual). The MLP IS
#      the "MLP" in "DIN + MLP" from the paper; it lets the model learn
#      how to weight target-aware history aggregation against the
#      HyFormer pooled mixed signal, rather than forcing the
#      contribution to be additive (the exp/din-style constraint that
#      likely killed it).
#
# Orthogonal to mixed / hash / paired / senet / cross-arch / bpr because
# the change happens AFTER output_proj and only touches the pooled
# vector + classifier path. Eval container needs the matching
# `use_din_real=True` model architecture; train_config.json carries the
# flag so infer.py picks it up automatically.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_din_real \
    --din_hidden_mult 2 \
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
