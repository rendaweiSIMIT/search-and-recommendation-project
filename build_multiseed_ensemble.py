"""Pack the best-val-AUC epoch from each seed-training run into one ensemble model.pt.

Run AFTER training every seed has finished. For each per-seed save_dir
(e.g. ``${TRAIN_CKPT_PATH}/seed42`` and ``${TRAIN_CKPT_PATH}/seed3407``),
this script reads ``val_history.json``, picks the single epoch with the
highest val_auc, locates that epoch's checkpoint dir, and bundles all
seeds' state_dicts into one ``model.pt`` with the ``ensemble`` marker that
infer.py recognises. Sidecars (schema.json, train_config.json,
ns_groups.json, id_stats.npz) are copied so the bundled directory is a
self-contained submission target.

Usage:
    python3 build_multiseed_ensemble.py \\
        --base_dir "$TRAIN_CKPT_PATH" \\
        --seed_dirs seed42 seed3407 \\
        --out_subdir multiseed_best2
"""
import os
import json
import shutil
import argparse
import logging
from glob import glob

import torch


SIDECARS = ('schema.json', 'train_config.json', 'ns_groups.json', 'id_stats.npz')


def _pick_best_epoch(seed_dir: str) -> dict:
    """Return the val_history entry with the highest val_auc."""
    history_path = os.path.join(seed_dir, 'val_history.json')
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"val_history.json not found at {history_path}. "
            f"Trainer must run with a validation loader so val metrics are recorded.")
    with open(history_path, 'r') as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"val_history.json at {history_path} is empty.")
    return max(history, key=lambda h: h['val_auc'])


def _load_best_state_dict(seed_dir: str, best: dict):
    pattern = os.path.join(
        seed_dir,
        f"global_step{best['global_step']}.*.epoch{best['epoch']}")
    dirs = glob(pattern)
    if not dirs:
        raise FileNotFoundError(
            f"No checkpoint dir matched {pattern} -- epoch {best['epoch']}, "
            f"step {best['global_step']} was logged but its ckpt is missing.")
    d = dirs[0]
    model_pt = os.path.join(d, 'model.pt')
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"{model_pt} missing")
    sd = torch.load(model_pt, map_location='cpu')
    # Refuse nested ensembles
    if isinstance(sd, dict) and sd.get('ensemble') is True:
        raise ValueError(
            f"{model_pt} is already an ensemble bundle; refusing to nest.")
    return sd, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', required=True,
                    help='Top-level dir whose subdirs are per-seed save_dirs.')
    ap.add_argument('--seed_dirs', nargs='+', required=True,
                    help='Per-seed subdirectory names under base_dir, '
                         'e.g. seed42 seed3407.')
    ap.add_argument('--out_subdir', default='multiseed_best',
                    help='Subdirectory of base_dir to write the bundle into.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    state_dicts = []
    epochs = []
    val_aucs = []
    seed_labels = []
    sidecar_src = None
    for sd_name in args.seed_dirs:
        seed_dir = os.path.join(args.base_dir, sd_name)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(
                f"Seed dir {seed_dir} not found -- training for this seed "
                f"may have failed.")
        best = _pick_best_epoch(seed_dir)
        logging.info(
            f"{sd_name}: best epoch {best['epoch']:>3}  step {best['global_step']:>7}"
            f"  val_auc {best['val_auc']:.6f}")
        sd, src_dir = _load_best_state_dict(seed_dir, best)
        state_dicts.append(sd)
        epochs.append(int(best['epoch']))
        val_aucs.append(float(best['val_auc']))
        seed_labels.append(sd_name)
        if sidecar_src is None:
            sidecar_src = src_dir

    out_dir = os.path.join(args.base_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        'ensemble': True,
        'state_dicts': state_dicts,
        'n_models': len(state_dicts),
        'epochs': epochs,
        'val_aucs': val_aucs,
        'seeds': seed_labels,
    }
    out_pt = os.path.join(out_dir, 'model.pt')
    torch.save(bundle, out_pt)
    logging.info(
        f"Wrote multi-seed ensemble bundle to {out_pt} "
        f"({len(state_dicts)} state_dicts: {seed_labels})")

    # copy sidecars from one of the picked seed dirs (all share the same schema)
    for s in SIDECARS:
        src = os.path.join(sidecar_src, s)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)
            logging.info(f"  copied sidecar: {s}")

    print(f"\nReady to submit:  MODEL_OUTPUT_PATH={out_dir}")


if __name__ == '__main__':
    main()
