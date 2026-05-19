#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/v9-mixed-dpe-fulldata-frontval ----
# = the 12-epoch fulldata config, but instead of NO validation it keeps a
#   small 5% MONITOR validation split taken from the OLDEST Row Groups:
#   --valid_ratio 0.05 --valid_from_head
#       val = the oldest 5% of Row Groups. This keeps the most recent
#       (most test-relevant) data IN training, and gives the val set
#       fully-matured (un-censored) labels -> a cleaner monitoring curve.
#   --patience 999  -> EarlyStopping never fires; all 12 epochs are trained.
#       The 5% val is a health monitor only, NOT an epoch selector -- pick
#       the epoch by the cumulative-LR analysis (cosine tail, ~10-12).
#   --num_epochs 12 / --seed 3407 / cosine: unchanged from the fulldata run.
# Everything else is unchanged from exp/v9-mixed (v9 full-stack + our
# pretrained-mixed: --additive_dense_fids 61 / --gating_dense_fids 87).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 3 \
    --item_ns_tokens 4 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --hash_bucket_size 100000 \
    --use_target_attention \
    --num_workers 8 \
    --num_cross_layers 2 \
    --cross_low_rank 64 \
    --use_se_net \
    --use_ns_self_attn \
    --use_ns_output_fusion \
    --use_temporal_bias \
    --use_time_gap \
    --precision bf16 \
    --valid_ratio 0.05 \
    --valid_from_head \
    --patience 999 \
    --num_epochs 12 \
    --seed 3407 \
    --lr_schedule cosine \
    --warmup_steps 500 \
    --ema_decay 0.999 \
    --label_smoothing 0.01 \
    --weight_decay 0.02 \
    --loss_type bce_pairwise \
    --pairwise_lambda 0.05 \
    --additive_dense_fids 61 \
    --gating_dense_fids 87 \
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
