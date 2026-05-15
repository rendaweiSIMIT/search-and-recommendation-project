#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/feat-item-int-10-solo: explicit adapter for item_int_feats_10 ----
# EDA (HANDOFF §4.6.5) shows item_int_feats_10 has 1-D AUC 0.4683
# (signal 0.063, negative correlation). Rank #20 overall by signal,
# the 4th-strongest scalar item_int. Range [0, 299]. Same dilution
# argument as the other item_int solos. Recipe identical: dedicated
# 64-d Embedding + 1-layer adapter -> additive residual to pooled
# output before the classifier.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --explicit_item_int_fids 10 \
    "$@"
