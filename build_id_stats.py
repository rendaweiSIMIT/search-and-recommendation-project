"""Offline pre-compute of item_id / user_id statistics for ID statistical
encoding (exp/v9-mixed-dpe-id-stats).

Scans the *training* Row Groups only (the last ``valid_ratio`` fraction of
Row Groups is the validation split and is excluded, so validation rows stay
leakage-free), and writes per-id (count, positive_count) tables plus the
global CVR and the smoothing constant to an ``id_stats.npz`` file.

Counts use only item_id / user_id / label_type, never the model -- the file
is a pure data artifact. It is shipped next to each checkpoint so that
infer.py can attach the same features at test time.

Usage (run.sh runs this automatically before train.py):
    python3 build_id_stats.py --valid_ratio 0.1 --alpha 100 --out id_stats.npz
Environment:
    TRAIN_DATA_PATH  training data directory (*.parquet); --data_dir overrides.
"""
import os
import glob
import argparse
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-compute item/user id stats")
    ap.add_argument('--data_dir', default=os.environ.get('TRAIN_DATA_PATH'),
                    help='Training data dir (env TRAIN_DATA_PATH)')
    ap.add_argument('--valid_ratio', type=float, default=0.1,
                    help='Tail fraction of Row Groups used as validation; '
                         'excluded from the stats so val rows stay leak-free. '
                         'MUST match train.py --valid_ratio.')
    ap.add_argument('--alpha', type=float, default=100.0,
                    help='Smoothing constant: smoothed_cvr = '
                         '(pos + alpha*global) / (count + alpha)')
    ap.add_argument('--out', default=None,
                    help='Output .npz path (default <script_dir>/id_stats.npz)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    if not args.data_dir:
        raise ValueError("data_dir not set (pass --data_dir or TRAIN_DATA_PATH)")
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'id_stats.npz')

    # ---- replicate get_pcvr_data's Row Group ordering / split ----
    files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {args.data_dir}")
    rg_info = []  # (file, rg_idx)
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i))
    total_rgs = len(rg_info)
    if args.valid_ratio <= 0:
        n_train_rgs = total_rgs
    else:
        n_valid_rgs = max(1, int(total_rgs * args.valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs
    train_rgs = rg_info[:n_train_rgs]
    logging.info(f"build_id_stats: {total_rgs} Row Groups total, "
                 f"using first {n_train_rgs} for stats "
                 f"(valid_ratio={args.valid_ratio})")

    # ---- read item_id / user_id / label_type from the training Row Groups ----
    by_file: dict = {}
    for f, i in train_rgs:
        by_file.setdefault(f, []).append(i)
    items, users, labels = [], [], []
    for f, idxs in by_file.items():
        t = pq.ParquetFile(f).read_row_groups(
            idxs, columns=['item_id', 'user_id', 'label_type'])
        items.append(t.column('item_id').fill_null(0)
                     .to_numpy(zero_copy_only=False).astype(np.int64))
        users.append(t.column('user_id').fill_null(0)
                     .to_numpy(zero_copy_only=False).astype(np.int64))
        labels.append(t.column('label_type').fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64))
    item_id = np.concatenate(items)
    user_id = np.concatenate(users)
    label = (np.concatenate(labels) == 2).astype(np.int64)
    n_rows = len(label)
    global_cvr = float(label.mean())
    logging.info(f"build_id_stats: {n_rows} training rows, "
                 f"global_cvr={global_cvr:.6f}")

    # ---- group-by aggregation ----
    df = pd.DataFrame({'item': item_id, 'user': user_id, 'label': label})
    ig = df.groupby('item')['label']
    item_ids = ig.size().index.to_numpy().astype(np.int64)
    item_count = ig.size().to_numpy().astype(np.int64)
    item_pos = ig.sum().to_numpy().astype(np.int64)
    ug = df.groupby('user')['label']
    user_ids = ug.size().index.to_numpy().astype(np.int64)
    user_count = ug.size().to_numpy().astype(np.int64)

    np.savez(
        out,
        item_ids=item_ids, item_count=item_count, item_pos=item_pos,
        user_ids=user_ids, user_count=user_count,
        global_cvr=np.array(global_cvr, dtype=np.float64),
        alpha=np.array(args.alpha, dtype=np.float64),
    )
    logging.info(
        f"build_id_stats: wrote {out} -- {len(item_ids)} distinct items, "
        f"{len(user_ids)} distinct users, global_cvr={global_cvr:.6f}, "
        f"alpha={args.alpha}")


if __name__ == '__main__':
    main()
