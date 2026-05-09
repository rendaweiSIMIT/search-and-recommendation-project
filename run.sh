#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: exp/seq-budget ----
# Sequence length reallocation only (zero model.py / dataset.py changes).
#
# Platform data_stats showed mean sequence lengths a=750, b=722, c=514,
# d=2456 vs caps 256/256/512/512 in baseline -> domain D loses ~80% of
# its history. Reallocate caps to 512/512/1024/1536 to recover D's
# signal, and switch all four domains to the LongerEncoder so memory
# stays bounded by ``top_k * L`` (top_k=64) instead of ``L^2``.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --seq_max_lens "seq_a:512,seq_b:512,seq_c:1024,seq_d:1536" \
    --seq_encoder_type longer \
    --seq_top_k 64 \
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
