#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Defensive against vGPU memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Active config: exp/hour-shuffle-val ----
# Re-attempt of hour-of-day as an NS token, but with a critical fix to
# the local val signal so the platform's hidden test AUC is not surprised.
#
# Why our previous exp/hour-of-day attempt failed (-0.014):
#   - baseline get_pcvr_data does positional split: val = last 10% RGs
#     = last ~9 contiguous hours of the 3.86-day training window.
#   - On the platform's training logs we see val AUC under that narrow
#     9-hour window. The model can learn "20-23 hr CVR pattern" and that
#     pattern matches the val window perfectly -> train log val AUC up.
#   - Platform hidden test is a different time window. Its hour mix
#     (e.g. 8-11 hr) was never observed in val. The "hour -> CVR"
#     function the model learned on train transfers to test, but our
#     train log val AUC overstated how confident we should be in it.
#   - Result: val log looked OK, platform AUC dropped 0.014.
#
# This branch fixes the LOCAL signal by switching to RG-level shuffle
# val (shuffle_val_seed=42). Val is now scattered across the time
# window, so val itself sees all 24 hours of day, all observed days
# of week, etc. If hour features genuinely generalize, the new val
# AUC will move. If they only "fit the calendar phase of train's tail
# 9 hours" (the previous trap), the shuffled val will catch that and
# refuse to move.
#
# NS token budget: T = num_queries * S + num_ns must satisfy
#   d_model % T == 0, here 64 % T == 0.
# Default baseline: user_ns=5, user_dense=1, item_ns=2, item_dense=0
#   -> num_ns=8, T=2*4+8=16. 64 % 16 == 0.
# With +1 hour token, num_ns becomes 9 -> T=17 (not divisible).
# We reduce user_ns_tokens 5 -> 4 to compensate:
#   num_ns = 4 (user_ns) + 1 (user_dense) + 2 (item_ns) + 1 (hour) = 8
#   T = 2*4 + 8 = 16. 64 % 16 == 0.  OK.
#
# Eval container needs the matching --use_hour shape because the model
# state_dict gains hour_emb + hour_proj. shuffle_val_seed is a TRAINING-
# ONLY param; the eval container does not care about it.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_hour \
    --shuffle_val_seed 42 \
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
