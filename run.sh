#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- exp/v9-mixed: v9 full-stack + our exp/pretrained-mixed +0.0039 winner ----
# Everything from v9's run.sh, unchanged, PLUS the two pretrained-embedding
# integration paths from our exp/pretrained-mixed branch:
#   --additive_dense_fids 61   user_dense_61 (Meta SUM)   -> additive residual
#   --gating_dense_fids   87   user_dense_87 (Tencent LFM) -> 2*sigmoid gate
# These operate on the pooled output right before the classifier and are
# orthogonal to v9 (v9's UserSparseDensePair only covers fids 62-66, never
# 61/87).
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
    --patience 10 \
    "$@"

# ---- Snapshot ensemble: bundle top-3 val-AUC epochs into one model.pt ----
# Reads ${TRAIN_CKPT_PATH}/val_history.json (written per-epoch by trainer),
# picks the 3 epochs with the highest val_auc, loads each epoch's saved
# state_dict, and writes them into a single ensemble bundle at
# ${TRAIN_CKPT_PATH}/ensemble_top3/model.pt + sidecars. The submission target
# is that bundle directory; infer.py auto-detects the ensemble marker and
# averages probabilities across the 3 members.
TRAIN_CKPT_DIR="${TRAIN_CKPT_PATH:-./checkpoints}"
python3 -u "${SCRIPT_DIR}/build_snapshot_ensemble.py" \
    --ckpt_dir "${TRAIN_CKPT_DIR}" \
    --top_k 3 \
    --out_subdir ensemble_top3

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
