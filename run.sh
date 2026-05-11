#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/cross-seq-pool ----
# HyFormer paper §4.2.2 ablation: removing the cross-sequence pooling
# tokens from the full query loses 0.05% AUC. Our baseline implementation
# only feeds Seq_i's own pool into the i-th sequence's Query Generation
# FFNs (see baseline MultiSeqQueryGenerator). The paper's full HyFormer
# feeds ALL sequences' pools as a shared inter-sequence context.
#
# With this flag the FFN input grows from (M+1)*D to (M+S)*D = (8+4)*64
# = 768 dim per query (vs baseline 9*64 = 576 dim). Adds ~50k dense
# params total in MultiSeqQueryGenerator. Pure additive context;
# orthogonal to mixed / hash / paired / din / senet / cross-arch /
# bpr / onetrans-suffix because it does not touch the NS tokenizer,
# the per-sequence encoder, the RankMixer block, or the output head.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_cross_seq_pool \
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
