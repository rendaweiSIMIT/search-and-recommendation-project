#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/feat-item-int-13-solo: explicit adapter for item_int_feats_13 ----
# EDA (HANDOFF §4.6.5) shows item_int_feats_13 has the strongest scalar
# int 1-D AUC: 0.5616 (signal 0.123), only seq timestamp columns and
# label_time score higher. Baseline RankMixerNSTokenizer concatenates
# all 14 item_int fid embeddings into one long vector and splits into 2
# chunks of 448 dim each, diluting fid 13 to ~64/896 = 7% of one chunk's
# input. This solo branch adds a dedicated 64-d embedding + adapter
# whose output is added as a residual to the pooled output (same recipe
# as exp/pretrained-mixed +0.0039 winner). Single-fid scope to isolate
# fid 13's contribution from confounds with other strong scalar ints
# (6 / 10 / 12 each get their own solo branch).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --explicit_item_int_fids 13 \
    "$@"
