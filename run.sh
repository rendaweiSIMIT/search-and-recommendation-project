#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/feat-user-int-1-solo: explicit adapter for user_int_feats_1 ----
# EDA (HANDOFF §4.6.5) ranks user_int_feats_1 as the top scalar user_int
# by 1-D AUC: 0.5410 (signal 0.082). Baseline RankMixerNSTokenizer
# concatenates all 46 user_int fid embeddings (46 * 64 = 2944 dim) into
# one long vector, splits into 5 chunks of 589 dim each, then projects
# each chunk to d_model=64 -- fid 1's signal gets diluted to ~64/2944
# = 2% of one chunk's input (even more severe than the item side).
# This solo branch adds a dedicated 64-d embedding + adapter whose
# output is added as a residual to the pooled output (same recipe as
# exp/pretrained-mixed +0.0039 winner).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --explicit_user_int_fids 1 \
    "$@"
