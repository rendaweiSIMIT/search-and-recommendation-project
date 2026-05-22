"""Pack chosen-epoch checkpoints into one snapshot-ensemble model.pt.

Run AFTER training finishes. Given a list of epochs (--epochs, default
6 7 8), this locates each epoch's per-epoch checkpoint directory under
--ckpt_dir, loads its model.pt, and bundles the state_dicts into a single
model.pt carrying the ``ensemble`` marker that infer.py recognises.

The bundle directory is named to look like one more epoch checkpoint:
``global_step{S}.layer=L.head=H.hidden=D.epoch{maxN+1}``. The training
platform's checkpoint list only registers directories matching that
pattern -- a custom name (e.g. ``ensemble_top3``) is silently dropped.
epoch = max trained epoch + 1, so the bundle sorts right after the real
epochs and never name-collides with them.

Sidecars (schema.json, train_config.json, ns_groups.json, id_stats.npz)
are copied so the bundle directory is a self-contained submission target.

Usage:
    python3 build_snapshot_ensemble.py --ckpt_dir "$TRAIN_CKPT_PATH" \\
        --epochs 6 7 8
"""
import os
import re
import shutil
import argparse
import logging
from glob import glob

import torch


SIDECARS = ('schema.json', 'train_config.json', 'ns_groups.json', 'id_stats.npz')

# global_step28992.layer=2.head=4.hidden=64.epoch8
_CKPT_RE = re.compile(
    r'global_step(\d+)\.(layer=\d+\.head=\d+\.hidden=\d+)\.epoch(\d+)$')


def _scan_epoch_ckpts(ckpt_dir):
    """Return {epoch: (dir_path, global_step, lhd_str)} for every
    global_step*.epoch* directory directly under ckpt_dir."""
    found = {}
    for d in glob(os.path.join(ckpt_dir, 'global_step*.epoch*')):
        m = _CKPT_RE.match(os.path.basename(d.rstrip('/')))
        if m:
            found[int(m.group(3))] = (d, int(m.group(1)), m.group(2))
    return found


def _bundle_dirname(epoch_ckpts):
    """Name the bundle global_step{S}.layer..head..hidden..epoch{maxN+1} so
    the platform's checkpoint list (which only registers dirs matching that
    pattern) shows it. The step number continues the per-epoch series."""
    max_ep = max(epoch_ckpts)
    _, max_step, lhd = epoch_ckpts[max_ep]
    steps_per_epoch = max(1, max_step // max_ep)
    return f"global_step{max_step + steps_per_epoch}.{lhd}.epoch{max_ep + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True,
                    help='Trainer save_dir (contains global_step*.epoch* dirs)')
    ap.add_argument('--epochs', type=int, nargs='+', default=[6, 7, 8],
                    help='Which epochs to ensemble (default: 6 7 8).')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    epoch_ckpts = _scan_epoch_ckpts(args.ckpt_dir)
    if not epoch_ckpts:
        raise FileNotFoundError(
            f"No global_step*.epoch* checkpoint dirs found under "
            f"{args.ckpt_dir}. Trainer must save one checkpoint per epoch.")

    state_dicts = []
    sidecar_src = None
    for ep in args.epochs:
        if ep not in epoch_ckpts:
            raise FileNotFoundError(
                f"epoch {ep} requested but no checkpoint for it under "
                f"{args.ckpt_dir} (found epochs: {sorted(epoch_ckpts)})")
        d, step, _ = epoch_ckpts[ep]
        model_pt = os.path.join(d, 'model.pt')
        if not os.path.exists(model_pt):
            raise FileNotFoundError(f"{model_pt} missing")
        sd = torch.load(model_pt, map_location='cpu')
        # Refuse to nest: a member must be a plain state_dict, not a bundle.
        if isinstance(sd, dict) and sd.get('ensemble') is True:
            raise ValueError(
                f"{model_pt} is already an ensemble bundle; refusing to nest.")
        state_dicts.append(sd)
        logging.info(f"  epoch {ep:>3}  step {step:>7}  <- {os.path.basename(d)}")
        if sidecar_src is None:
            sidecar_src = d

    out_subdir = _bundle_dirname(epoch_ckpts)
    out_dir = os.path.join(args.ckpt_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        'ensemble': True,
        'state_dicts': state_dicts,
        'n_models': len(state_dicts),
        'epochs': list(args.epochs),
    }
    out_pt = os.path.join(out_dir, 'model.pt')
    torch.save(bundle, out_pt)
    logging.info(
        f"Wrote snapshot ensemble bundle ({len(state_dicts)} members, "
        f"epochs {list(args.epochs)}) to {out_pt}")

    # copy sidecars from one of the picked epoch dirs (all share the schema)
    for s in SIDECARS:
        src = os.path.join(sidecar_src, s)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)
            logging.info(f"  copied sidecar: {s}")

    print(f"\nReady to submit:  MODEL_OUTPUT_PATH={out_dir}")
    print(f"Bundle dir (platform-visible, sorts after the real epochs): "
          f"{out_subdir}")


if __name__ == '__main__':
    main()
