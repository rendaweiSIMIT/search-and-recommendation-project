"""Pack each seed's best-val checkpoint into one multi-seed ensemble model.pt.

Run AFTER every seed has finished training. For each per-seed save_dir
(e.g. base_dir/seed42, base_dir/seed3407, base_dir/seed1210) this finds
that seed's ``*.best_model`` directory -- the epoch with the highest
validation AUC, kept on disk by the trainer -- loads its model.pt, and
bundles all seeds' state_dicts into one model.pt carrying the ``ensemble``
marker that infer.py recognises.

The bundle directory is named to look like one more epoch checkpoint:
``global_step{S}.layer=L.head=H.hidden=D.epoch{maxN+1}``. The training
platform's checkpoint list only registers directories matching that
pattern -- a custom name (e.g. ``multiseed_best2``) is silently dropped.
With 8 trained epochs the bundle becomes ...epoch9, sorting right after
the real per-epoch checkpoints and never name-colliding with them.

Sidecars (schema.json, train_config.json, ns_groups.json, id_stats.npz)
are copied so the bundle directory is a self-contained submission target.

Usage:
    python3 build_multiseed_ensemble.py --base_dir "$TRAIN_CKPT_PATH" \\
        --seed_dirs seed42 seed3407 seed1210
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
_EPOCH_RE = re.compile(
    r'global_step(\d+)\.(layer=\d+\.head=\d+\.hidden=\d+)\.epoch(\d+)$')


def _find_best_model(seed_dir: str):
    """Return (best_model_dir, model_pt_path) for a seed's *.best_model dir.

    The trainer keeps exactly one ``*.best_model`` dir per save_dir (older
    ones are pruned), holding the highest-val-AUC epoch's weights."""
    matches = sorted(glob(os.path.join(seed_dir, 'global_step*.best_model')))
    if not matches:
        raise FileNotFoundError(
            f"No *.best_model dir under {seed_dir} -- training for this seed "
            f"may have failed, or ran with no validation split.")
    d = matches[-1]
    model_pt = os.path.join(d, 'model.pt')
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"{model_pt} missing")
    return d, model_pt


def _bundle_dirname(seed_dir: str) -> str:
    """Name the bundle global_step{S}.layer..head..hidden..epoch{maxN+1},
    sized from one seed's per-epoch checkpoints, so the platform's checkpoint
    list (which only registers dirs matching that pattern) shows it."""
    best = None  # (epoch, global_step, lhd)
    for d in glob(os.path.join(seed_dir, 'global_step*.epoch*')):
        m = _EPOCH_RE.match(os.path.basename(d.rstrip('/')))
        if m:
            ep = int(m.group(3))
            if best is None or ep > best[0]:
                best = (ep, int(m.group(1)), m.group(2))
    if best is None:
        raise FileNotFoundError(
            f"No global_step*.epoch* dirs under {seed_dir} to size the "
            f"bundle name from.")
    max_ep, max_step, lhd = best
    steps_per_epoch = max(1, max_step // max_ep)
    return f"global_step{max_step + steps_per_epoch}.{lhd}.epoch{max_ep + 1}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', required=True,
                    help='Top-level dir whose subdirs are per-seed save_dirs.')
    ap.add_argument('--seed_dirs', nargs='+', required=True,
                    help='Per-seed subdirectory names under base_dir, '
                         'e.g. seed42 seed3407 seed1210.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    state_dicts = []
    seed_labels = []
    sidecar_src = None
    name_src = None
    for sd_name in args.seed_dirs:
        seed_dir = os.path.join(args.base_dir, sd_name)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(
                f"Seed dir {seed_dir} not found -- training for this seed "
                f"may have failed.")
        best_dir, model_pt = _find_best_model(seed_dir)
        sd = torch.load(model_pt, map_location='cpu')
        # Refuse to nest: a member must be a plain state_dict, not a bundle.
        if isinstance(sd, dict) and sd.get('ensemble') is True:
            raise ValueError(
                f"{model_pt} is already an ensemble bundle; refusing to nest.")
        state_dicts.append(sd)
        seed_labels.append(sd_name)
        logging.info(f"  {sd_name}: best_model <- {os.path.basename(best_dir)}")
        if sidecar_src is None:
            sidecar_src = best_dir
        if name_src is None:
            name_src = seed_dir

    out_subdir = _bundle_dirname(name_src)
    out_dir = os.path.join(args.base_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        'ensemble': True,
        'state_dicts': state_dicts,
        'n_models': len(state_dicts),
        'seeds': seed_labels,
    }
    out_pt = os.path.join(out_dir, 'model.pt')
    torch.save(bundle, out_pt)
    logging.info(
        f"Wrote multi-seed ensemble bundle ({len(state_dicts)} members: "
        f"{seed_labels}) to {out_pt}")

    # copy sidecars from one seed's best_model dir (all seeds share the schema)
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
