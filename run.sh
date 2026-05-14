#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/feat-user-int-1 ----
# Explicit adapter for user_int_feats_1 (EDA 1-D AUC 0.5410, signal
# 0.082 -- the strongest scalar user_int feature; second only to
# item_int_13 in the scalar-int leaderboard).
#
# Motivation:
#   Same dilution argument as exp/feat-item-int-13, applied on the user
#   side. The baseline RankMixerNSTokenizer concatenates all 46
#   user_int fid embeddings (46 * 64 = 2944 dim) into one long vector,
#   then splits it into 5 chunks of 589 dim each, then projects each
#   chunk to d_model=64. Any single fid contributes only ~64/2944 ~
#   2% of the input dimension to one chunk -- the dilution is even
#   more severe on the user side than the item side (because there
#   are 46 user fids vs 14 item fids).
#
# Architecture:
#   user_int_feats[:, user_int_1_offset]  (scalar 0-5, 6 unique)
#       -> dedicated nn.Embedding(7, 64, padding_idx=0)
#       -> Linear(64, 64) + LayerNorm
#       -> additive residual to pooled output (B, 64)
#       -> classifier
#
# This is a sibling experiment to exp/feat-item-int-13: same infra,
# different fid. Running both lets us see whether the explicit-adapter
# recipe transfers across user_int / item_int and across signal
# magnitudes (0.082 vs 0.123).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --explicit_user_int_fids 1 \
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
