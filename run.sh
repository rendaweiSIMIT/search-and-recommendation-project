#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/v9-mixed-dpe-recency-tprofile-snapshot ----
# = exp/v9-mixed-dpe-recency-tprofile (test-best 0.828098 @ epoch 6), trained
#   for a FIXED 8 epochs, then epochs 6/7/8 bundled into one snapshot ensemble.
#
# The LR is byte-identical to the tprofile baseline -- setting num_epochs does
# NOT change the learning rate at any step:
#   --num_epochs 8        the training loop runs exactly 8 epochs
#   --lr_total_epochs 999 the cosine schedule stays sized for 999 epochs
#                         (tprofile's default), so the LR at every step equals
#                         the un-truncated baseline run
#   --patience 999        EarlyStopping never fires; all 8 epochs are trained
# Everything else is unchanged from exp/v9-mixed-dpe-recency-tprofile.
TRAIN_CKPT_DIR="${TRAIN_CKPT_PATH:-./checkpoints}"
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
    --lr_schedule cosine \
    --warmup_steps 500 \
    --ema_decay 0.999 \
    --label_smoothing 0.01 \
    --weight_decay 0.02 \
    --loss_type bce_pairwise \
    --pairwise_lambda 0.05 \
    --additive_dense_fids 61 \
    --gating_dense_fids 87 \
    --num_epochs 8 \
    --lr_total_epochs 999 \
    --patience 999 \
    "$@"

# ---- Snapshot ensemble: bundle epochs 6/7/8 into one model.pt ----
# build_snapshot_ensemble.py reads the per-epoch checkpoints the trainer wrote
# (global_step*.epoch6/7/8), packs their state_dicts into a single ensemble
# model.pt, and names the bundle dir
# global_step*.layer=2.head=4.hidden=64.epoch9 so the platform's checkpoint
# list shows it -- the list only registers global_step*-pattern dirs, a custom
# name like ensemble_top3 is silently dropped. infer.py auto-detects the
# ensemble marker and averages the 3 members' probabilities.
echo "===== Snapshot ensemble: bundling epochs 6/7/8 ====="
python3 -u "${SCRIPT_DIR}/build_snapshot_ensemble.py" \
    --ckpt_dir "${TRAIN_CKPT_DIR}" \
    --epochs 6 7 8

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
