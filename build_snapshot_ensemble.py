"""Pack the top-K best-val-AUC epoch checkpoints into one ensemble model.pt.

Run AFTER training finishes. Reads ``val_history.json`` from the trainer's
save_dir (one entry per epoch with val_auc), picks the top K epochs by
val_auc, locates each epoch's checkpoint directory, and bundles their
state_dicts into a single ``model.pt`` with the ``ensemble`` marker that
infer.py recognises. Sidecars (schema.json, train_config.json, ns_groups.json,
id_stats.npz) are copied so the bundled directory is a self-contained
submission target.

Usage:
    python3 build_snapshot_ensemble.py \\
        --ckpt_dir "$TRAIN_CKPT_PATH" --top_k 3 --out_subdir ensemble_top3
"""
import os
import json
import shutil
import argparse
import logging
from glob import glob

import torch


SIDECARS = ('schema.json', 'train_config.json', 'ns_groups.json', 'id_stats.npz')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True,
                    help='Trainer save_dir (contains val_history.json and '
                         'global_step*.epoch* checkpoint dirs)')
    ap.add_argument('--top_k', type=int, default=3,
                    help='How many top-val-AUC epochs to ensemble.')
    ap.add_argument('--out_subdir', default='ensemble_topK',
                    help='Subdirectory of ckpt_dir to write the bundle into.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    history_path = os.path.join(args.ckpt_dir, 'val_history.json')
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"val_history.json not found at {history_path}. "
            f"Trainer must run with a validation loader so val metrics are recorded.")
    with open(history_path, 'r') as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"val_history.json at {history_path} is empty.")

    # pick top-K by val_auc
    history_sorted = sorted(history, key=lambda h: h['val_auc'], reverse=True)
    top = history_sorted[:args.top_k]
    logging.info(f"Picked top-{len(top)} epochs by val_auc:")
    for h in top:
        logging.info(
            f"  epoch {h['epoch']:>3}  step {h['global_step']:>7}  "
            f"val_auc {h['val_auc']:.6f}")

    # find each epoch's checkpoint directory and load model.pt
    state_dicts = []
    sidecar_src = None
    for h in top:
        pattern = os.path.join(
            args.ckpt_dir,
            f"global_step{h['global_step']}.*.epoch{h['epoch']}")
        dirs = glob(pattern)
        if not dirs:
            raise FileNotFoundError(
                f"No checkpoint dir matched {pattern} -- epoch {h['epoch']}, "
                f"step {h['global_step']} was logged but its ckpt is missing.")
        d = dirs[0]
        model_pt = os.path.join(d, 'model.pt')
        if not os.path.exists(model_pt):
            raise FileNotFoundError(f"{model_pt} missing")
        sd = torch.load(model_pt, map_location='cpu')
        # In case some other branch saves bundles already, refuse nested ensembles
        if isinstance(sd, dict) and sd.get('ensemble') is True:
            raise ValueError(
                f"{model_pt} is already an ensemble bundle; refusing to nest.")
        state_dicts.append(sd)
        if sidecar_src is None:
            sidecar_src = d

    out_dir = os.path.join(args.ckpt_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        'ensemble': True,
        'state_dicts': state_dicts,
        'n_models': len(state_dicts),
        'epochs': [int(h['epoch']) for h in top],
        'val_aucs': [float(h['val_auc']) for h in top],
    }
    out_pt = os.path.join(out_dir, 'model.pt')
    torch.save(bundle, out_pt)
    logging.info(f"Wrote ensemble bundle to {out_pt} ({len(state_dicts)} state_dicts)")

    # copy sidecars from one of the picked epoch dirs (all share the same schema)
    for s in SIDECARS:
        src = os.path.join(sidecar_src, s)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)
            logging.info(f"  copied sidecar: {s}")

    print(f"\nReady to submit:  MODEL_OUTPUT_PATH={out_dir}")


if __name__ == '__main__':
    main()
