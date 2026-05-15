#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/feat-item-int-6-solo: explicit adapter for item_int_feats_6 ----
# EDA (HANDOFF §4.6.5) shows item_int_feats_6 has 1-D AUC 0.4605
# (signal 0.079, negative correlation -- low values associate with
# higher PCVR). Rank #14 overall in scalar 1-D AUC, top non-13 scalar
# item_int. Range [0, 911]. Same dilution problem as fid 13: shares
# the 896-dim concat then 448-dim chunks; effective contribution to
# any single chunk ~7% of input.
# Recipe identical to 13-solo: dedicated 64-d Embedding + 1-layer
# adapter -> additive residual to pooled output.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --explicit_item_int_fids 6 \
    "$@"
