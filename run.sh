#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: exp/pretrained-mixed-extended ----
# Combines three pretrained-embedding adapters + paired-pool restricted
# to raw-counter columns:
#
#   Output side: output' = output * gate_87 + residual_61 + residual_89_91
#     * fid 61 (Meta SUM, 256-d)   -> additive adapter (residual)
#     * fid 87 (Tencent LFM4Ads)   -> gating adapter (multiplicative)
#     * fid 89-91 (3 x 10-d affinity scores) -> additive adapter (NEW)
#
#   Input side: NS tokenizer paired pool on fid 62,63,64,65,66 only.
#     (fid 89-91 dropped from paired-pool since their [-1,1] range made
#      signed_log1p + softmax a no-op; they now flow through the
#      output-side adapter above instead.)
#
# All four CLI knobs default to the right values in train.py, so the
# only thing run.sh needs to set is the existing rankmixer config.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
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
