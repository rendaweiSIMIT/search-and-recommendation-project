#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/compile-amp (training speedup, baseline model) ----
# Pure infra change: bf16 autocast + torch.compile on the training forward.
# Model architecture, state_dict, and outputs are unchanged, so the eval
# container needs no matching adaptation.
#   --use_amp:      torch.amp.autocast(dtype=bf16) wraps forward + loss.
#                   bf16 shares fp32's exponent range, so no GradScaler is
#                   required. Sidesteps GradScaler-vs-sparse-Adagrad
#                   interaction issues (sparse grads on Embedding weights
#                   are not first-class GradScaler citizens).
#   --use_compile:  torch.compile(model, dynamic=True). First batch pays
#                   the compile cost; subsequent training batches typically
#                   run 1.2-2x faster on Ampere+. predict()/eval path stays
#                   eager and still benefits from autocast on its own.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_amp \
    --use_compile \
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
