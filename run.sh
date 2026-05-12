#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation (not strictly needed for EDA
# but doesn't hurt and keeps the env consistent with training branches).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/eda ----
# Pure exploratory data analysis branch. No model is built and no checkpoint
# is produced. The training-side entry script streams every parquet under
# $TRAIN_DATA_PATH, accumulates per-feature statistics (null rate, value
# range, unique cardinality, 1-D AUC against the binary label) and
# sample-level distributions (hour-of-day, day-of-week, conversion delay
# label_time - timestamp), then dumps:
#
#   - $TRAIN_CKPT_PATH/eda_report_train.json   # full machine-readable dump
#   - $TRAIN_LOG_PATH/eda.log                  # human-readable summary
#   - stdout (captured by platform Logs panel)
#
# Goal: surface insights that inform follow-up feature-engineering or
# architectural decisions. Examples we are looking for:
#   * features with >95% zero rate -> embedding budget wasted on them
#   * features with single-feature AUC > 0.55 -> strong individual signal,
#     consider explicit adapter (like pretrained-mixed for fid 61/87)
#   * label_time - timestamp distribution -> bimodal? long tail? feedback?
#   * hour-of-day positive rate variability -> how much does diurnal really
#     matter for PCVR on this dataset
#
# Eval-side companion: evaluation/infer.py on this branch does the same on
# the held-out test parquets (no labels so no 1-D AUC), then writes a
# placeholder predictions.json so the platform's eval pipeline does not
# crash even though we don't care about the AUC of this branch.
python3 -u "${SCRIPT_DIR}/eda.py" --mode train "$@"
