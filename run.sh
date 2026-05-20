#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/v9-mixed-dpe-recency-multiseed: train v9-mixed-dpe-recency twice with
# different random seeds (42 and 3407), pick the best-val-AUC epoch from each
# seed, and bundle the two state_dicts into a single ensemble model.pt for
# submission. Same model/feature config as exp/v9-mixed-dpe-recency.
# ----
# Layout under ${TRAIN_CKPT_PATH}:
#   seed42/      seed=42 training output  (per-epoch ckpts, val_history.json, ...)
#   seed3407/    seed=3407 training output
#   multiseed_best2/   the ensemble bundle -- this is the submission target
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
        --num_epochs 12 \
        --patience 12 \
        --seed "${SEED}" \
        "$@"
}

run_one_seed 42 "$@"
run_one_seed 3407 "$@"

# ---- Multi-seed ensemble: bundle each seed's best-val-AUC epoch into one
# model.pt. Reads ${TRAIN_CKPT_DIR}/{seed42,seed3407}/val_history.json, picks
# the single best epoch per seed, and writes
# ${TRAIN_CKPT_DIR}/multiseed_best2/model.pt + sidecars. infer.py auto-detects
# the ensemble marker and averages probabilities across the 2 members.
echo "===== Multi-seed: bundling ensemble ====="
python3 -u "${SCRIPT_DIR}/build_multiseed_ensemble.py" \
    --base_dir "${TRAIN_CKPT_DIR}" \
    --seed_dirs seed42 seed3407 \
    --out_subdir multiseed_best2
