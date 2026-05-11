#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation (Taiji shares physical GPU
# across tenants; mixed-extended hit OOM here on first run before this
# fix). Pure CUDA allocator change, no training-dynamics impact.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/bpr-loss ----
# Replace BCE with all-pairs BPR ranking loss inside each batch.
# For each (positive, negative) pair in the batch (B=256 -> ~24 pos x ~232
# neg -> ~5500 pairs) the loss is -log(sigmoid(score_pos - score_neg)),
# a smooth surrogate for P(score_pos > score_neg) = AUC. Directly aligned
# with the evaluation metric, no model / dataset change required.
#
# Why not focal: focal reweights "hard examples" assuming hard = informative,
# but in PCVR data hard often = label noise. focal hurt -0.x in previous test.
# BPR avoids that trap entirely by working on RELATIVE scores instead of
# absolute calibration.
#
# Edge case: if a batch happens to have 0 positives or 0 negatives the loss
# falls back to BCE so the step still produces a gradient.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --loss_type bpr \
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
