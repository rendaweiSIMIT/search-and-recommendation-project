#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/v9-mixed-dpe-recency-tprofile-cos10-snapshot ----
# = exp/v9-mixed-dpe-recency-tprofile (test-best 0.828098), trained for a
#   FIXED 10 epochs with a NORMAL cosine LR decay, then the top-3 epochs by
#   5% validation AUC are bundled into one snapshot-ensemble checkpoint.
#
#   --num_epochs 10    train exactly 10 epochs; the cosine schedule is sized
#                      for 10 epochs too (NO --lr_total_epochs), so the LR
#                      decays normally from peak to the 0.05 floor across the
#                      run -- unlike -tprofile-snapshot, which pinned a
#                      999-epoch (near-constant) curve.
#   --valid_ratio 0.05 hold out the last 5% of Row Groups as validation.
#   --patience 999     EarlyStopping never fires; all 10 epochs are trained.
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
    --num_epochs 10 \
    --valid_ratio 0.05 \
    --patience 999 \
    "$@"

# ---- Snapshot ensemble: bundle the top-3 val-AUC epochs into one model.pt ----
# build_snapshot_ensemble.py reads val_history.json (per-epoch val metrics the
# trainer wrote), picks the 3 epochs with the highest 5%-val AUC, packs their
# state_dicts into one ensemble model.pt, and names the bundle dir
# global_step*.layer=2.head=4.hidden=64.epoch11 so the platform's checkpoint
# list shows it -- the list only registers global_step*-pattern dirs, a custom
# name is silently dropped. infer.py auto-detects the ensemble marker and
# averages the 3 members' probabilities.
echo "===== Snapshot ensemble: bundling top-3 val-AUC epochs ====="
python3 -u "${SCRIPT_DIR}/build_snapshot_ensemble.py" \
    --ckpt_dir "${TRAIN_CKPT_DIR}" \
    --top_k 3

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
