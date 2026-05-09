#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: exp/seq-hash ----
# Hash-trick rescue for the 4 sequence features that the platform's
# emb_skip_threshold=1M dropped (b_seq_69 max=64M, c_seq_29 max=5.76M,
# c_seq_34 max=1.03M, c_seq_47 max=86.34M -- almost certainly the user
# item-id history). Hash into 100K buckets so the model still sees
# *something* for these columns at the cost of bucket-collision noise.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --seq_hash_size 100000 \
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
