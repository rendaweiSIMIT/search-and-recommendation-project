#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/feat-item-int-12-solo: explicit adapter for item_int_feats_12 ----
# EDA (HANDOFF §4.6.5) shows item_int_feats_12 has 1-D AUC 0.4679
# (signal 0.064, negative correlation). Rank #18 by signal among
# scalar features, 3rd-strongest scalar item_int. Range [-1, 2442]
# (the -1 maps to 0 padding in the dataset pipeline). Same dilution
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
    --explicit_item_int_fids 12 \
    "$@"
