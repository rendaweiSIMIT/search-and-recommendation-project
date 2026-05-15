#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation (bigger hash table -> larger
# embedding allocations).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- exp/seq-c47-id-hash: 1M-bucket hash rescue for high-cardinality seq fids ----
# EDA (HANDOFF §1.4 / §4.6) flagged c_seq_47 (max ~86M) as almost
# certainly the user's item-id history -- a sequence with potentially
# the strongest behavioral signal but currently silenced by the
# baseline's emb_skip_threshold=1M (zero-vector lookup).
#
# exp/seq-hash (a9e9534) was the +0.0019 winner that rescued 4
# high-cardinality fids (b_seq_69, c_seq_29, c_seq_34, c_seq_47) with
# a 100K hash table -- a ~860:1 average collision rate for c_seq_47.
# This branch bumps the bucket to 1M (~86:1 collision), trading
# parameters for cleaner item-id buckets. The other 3 high-card fids
# benefit too: c_seq_34 (1.03M unique) at 1M is essentially full
# embedding; c_seq_29 (5.76M) at 1M is 5.76:1; b_seq_69 (64M) at 1M
# is 64:1.
#
# Memory: 4 hashed fids x 1M buckets x 64 emb_dim x 4 bytes ~= 1 GB
# embedding params + Adagrad state. Fits the vGPU envelope.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_hash_size 1000000 \
    "$@"
