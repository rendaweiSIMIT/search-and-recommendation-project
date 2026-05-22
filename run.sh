#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/v9-mixed-dpe-recency-tprofile-cos10-multiseed ----
# = exp/v9-mixed-dpe-recency-tprofile (test-best 0.828098), trained 3 times
#   with seeds 42 / 3407 / 1210, each a FIXED 10 epochs with a NORMAL cosine
#   LR decay; then each seed's best-val epoch is bundled into one multi-seed
#   ensemble checkpoint.
#
#   --num_epochs 10    train exactly 10 epochs; the cosine schedule is sized
#                      for 10 epochs too (NO --lr_total_epochs), so the LR
#                      decays normally from peak to the 0.05 floor across the
#                      run -- unlike -tprofile-multiseed, which pinned a
#                      999-epoch (near-constant) curve.
#   --valid_ratio 0.05 hold out the last 5% of Row Groups as validation.
#   --patience 999     EarlyStopping never fires; all 10 epochs are trained.
# Only --seed differs between the three runs; everything else is unchanged
# from exp/v9-mixed-dpe-recency-tprofile.
#
# Layout under ${TRAIN_CKPT_PATH}:
#   seed42/ seed3407/ seed1210/   per-seed training output (per-epoch ckpts +
#                                 one *.best_model dir each)
#   global_step*.epoch11/         the multi-seed ensemble bundle -- this is
#                                 the submission target
TRAIN_CKPT_DIR="${TRAIN_CKPT_PATH:-./checkpoints}"
TRAIN_LOG_DIR="${TRAIN_LOG_PATH:-./logs}"
TRAIN_TB_DIR="${TRAIN_TF_EVENTS_PATH:-./tb}"

run_one_seed() {
    local SEED="$1"
    shift
    local TAG="seed${SEED}"
    mkdir -p "${TRAIN_CKPT_DIR}/${TAG}" "${TRAIN_LOG_DIR}/${TAG}" "${TRAIN_TB_DIR}/${TAG}"
    echo "===== Multi-seed: training ${TAG} ====="
    TRAIN_CKPT_PATH="${TRAIN_CKPT_DIR}/${TAG}" \
    TRAIN_LOG_PATH="${TRAIN_LOG_DIR}/${TAG}" \
    TRAIN_TF_EVENTS_PATH="${TRAIN_TB_DIR}/${TAG}" \
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
        --seed "${SEED}" \
        "$@"
}

run_one_seed 42 "$@"
run_one_seed 3407 "$@"
run_one_seed 1210 "$@"

# ---- Multi-seed ensemble: bundle each seed's best-val epoch into one model.pt
# build_multiseed_ensemble.py reads ${TRAIN_CKPT_DIR}/{seed42,seed3407,seed1210},
# takes each seed's *.best_model checkpoint, packs the 3 state_dicts into one
# ensemble model.pt, and names the bundle dir
# global_step*.layer=2.head=4.hidden=64.epoch11 so the platform's checkpoint
# list shows it -- the list only registers global_step*-pattern dirs, a custom
# name like multiseed_best2 is silently dropped. infer.py auto-detects the
# ensemble marker and averages the 3 members' probabilities.
echo "===== Multi-seed: bundling ensemble ====="
python3 -u "${SCRIPT_DIR}/build_multiseed_ensemble.py" \
    --base_dir "${TRAIN_CKPT_DIR}" \
    --seed_dirs seed42 seed3407 seed1210
