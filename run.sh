#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/feat-item-int-13 ----
# Explicit adapter for item_int_feats_13 (EDA Top 1-D AUC 0.5616,
# signal 0.123 -- the strongest scalar int feature in the dataset).
#
# Motivation:
#   The baseline's RankMixerNSTokenizer concatenates all 14 item_int
#   fid embeddings (14 * 64 = 896 dims) into one long vector, then
#   splits it into 2 chunks of 448 dims each, then projects each
#   chunk to d_model=64. Any single fid contributes only ~64/896 ~ 7%
#   of the input dimension to one chunk -- the signal is diluted.
#   exp/pretrained-mixed already proved (and won +0.0039) that giving
#   a high-signal feature its own dedicated d_model-wide path beats
#   leaving it buried. This branch applies the same idea to the
#   strongest scalar int feature instead of a pretrained dense.
#
# Architecture:
#   item_int_feats[:, item_int_13_offset]  (scalar 0-8, 9 unique)
#       -> dedicated nn.Embedding(10, 64, padding_idx=0)
#       -> Linear(64, 64) + LayerNorm
#       -> additive residual to pooled output (B, 64)
#       -> classifier
#
# Trained for 4 epochs on 100% of data, no validation, seed 42
# (per feedback_kdd_fixed_4_epoch). Final ckpt auto-marked .best_model.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --explicit_item_int_fids 13 \
    --num_epochs 4 \
    --valid_ratio 0 \
    --seed 42 \
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
