#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: feature-eng-bundle ----
# Changes vs baseline:
#  * seq_max_lens reallocated: domain D mean=2456 was hard-capped at 512 (~80%
#    of the sequence truncated); raise d to 1536 and c to 1024 to recover that
#    signal. Use longer encoder (top-k compressed) so memory stays bounded.
#  * user_ns_tokens 5 -> 4 to keep T = num_queries*num_sequences + num_ns
#    divisible by d_model=64 once context_token adds +1 NS token.
#    T = 2*4 + (4 user + 1 user_dense + 2 item + 1 ctx) = 8 + 8 = 16, 64 % 16 = 0.
#  * seq_hash_size=100000 rescues the 4 high-card seq features (b_seq_69,
#    c_seq_29/34/47) currently lost to emb_skip_threshold.
#  * use_context_features=True turns on ContextTokenizer that consumes
#    hour-of-day, item_in_c47 and null_pattern_99_103 -> 1 extra NS token.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --seq_max_lens "seq_a:512,seq_b:512,seq_c:1024,seq_d:1536" \
    --seq_encoder_type longer \
    --seq_top_k 64 \
    --seq_hash_size 100000 \
    --use_context_features \
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
