"""Data-stats logging.

Run once at the very start of training to print a compact, scan-friendly
summary of the input parquet files. The intent is to surface differences
between the local 1k-row demo and the platform's full dataset (vocab
sizes, sequence lengths, null rates, value scales, time-range, sequence
sortedness, presence of pretrained dense embeddings, ...) by reading
back the platform's training log.

Designed to be fast (samples a small head of the parquet) and crash-free
(per-column try/except so one bad column does not abort logging).
"""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────── Sampling ────────────────────────────────────────


def _sample_head(data_dir: str, max_rows: int) -> Tuple[pa.Table, int, int]:
    """Read up to ``max_rows`` rows starting from the first parquet file.

    Returns (table, num_files_total, num_row_groups_read).
    """
    if os.path.isdir(data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    else:
        files = [data_dir]
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    pieces: List[pa.Table] = []
    rows = 0
    rg_read = 0
    for f in files:
        if rows >= max_rows:
            break
        pf = pq.ParquetFile(f)
        for rg_idx in range(pf.metadata.num_row_groups):
            if rows >= max_rows:
                break
            t = pf.read_row_group(rg_idx)
            pieces.append(t)
            rows += t.num_rows
            rg_read += 1
    table = pa.concat_tables(pieces) if len(pieces) > 1 else pieces[0]
    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    return table, len(files), rg_read


# ─────────────────────────── Per-column stats ────────────────────────────────


def _fmt_num(x: Any) -> str:
    """Compact number formatter."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1e9:
        return f"{x:.2e}"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    if ax >= 1e4:
        return f"{x / 1e3:.1f}K"
    if ax >= 1:
        return f"{x:.0f}" if x == int(x) else f"{x:.3g}"
    return f"{x:.3g}"


def _as_array(col: Any) -> pa.Array:
    """Materialize ChunkedArray (from concat_tables) into a single Array."""
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks()
    return col


def _scalar_line(name: str, col: pa.Array) -> str:
    """One-line scalar-column summary."""
    col = _as_array(col)
    n = len(col)
    null = col.null_count
    null_rate = null / max(n, 1)
    try:
        v = col.fill_null(0).to_numpy(zero_copy_only=False)
        if v.dtype.kind == 'f':
            real = v[~np.isnan(v)]
        else:
            real = v
        if real.size == 0:
            return f"{name}: type=scalar n={n} null={null} (ALL_NULL)"
        vmin = real.min(); vmax = real.max()
        # cap unique-counting to avoid pathological cost on huge arrays
        sample = real if real.size <= 200_000 else real[:200_000]
        n_unique = int(np.unique(sample).size)
        return (f"{name}: scalar n={n} null={null}({null_rate:.0%}) "
                f"range=[{_fmt_num(vmin)},{_fmt_num(vmax)}] "
                f"unique~{_fmt_num(n_unique)} mean={_fmt_num(real.mean())}")
    except Exception as e:  # pragma: no cover - defensive
        return f"{name}: scalar n={n} ERROR={e}"


def _array_line(name: str, col: pa.ListArray) -> str:
    """One-line list-column summary (works for list<int64> and list<float>)."""
    col = _as_array(col)
    n = len(col)
    null = col.null_count
    null_rate = null / max(n, 1)
    try:
        offsets = col.offsets.to_numpy()
        lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)
        # Empty (zero-length) rows are conceptually missing too
        empty = int((lengths == 0).sum())
        len_min = int(lengths.min()) if lengths.size > 0 else 0
        len_max = int(lengths.max()) if lengths.size > 0 else 0
        len_mean = float(lengths.mean()) if lengths.size > 0 else 0.0

        # Element-value stats from the flat values buffer
        values = col.values
        if len(values) == 0:
            elem_str = "elem=EMPTY"
        else:
            try:
                vraw = values.to_numpy()
                if vraw.dtype.kind == 'f':
                    valid = vraw[~np.isnan(vraw)]
                    if valid.size == 0:
                        elem_str = "elem=ALL_NAN"
                    else:
                        elem_str = (f"elem_range=[{_fmt_num(valid.min())},"
                                    f"{_fmt_num(valid.max())}] "
                                    f"mean={_fmt_num(valid.mean())}")
                else:
                    sample = vraw if vraw.size <= 500_000 else vraw[:500_000]
                    n_unique = int(np.unique(sample).size)
                    elem_str = (f"elem_range=[{_fmt_num(vraw.min())},"
                                f"{_fmt_num(vraw.max())}] "
                                f"unique~{_fmt_num(n_unique)}")
            except Exception as e:
                elem_str = f"elem_err={e}"

        return (f"{name}: array<{values.type}> n={n} null={null}({null_rate:.0%}) "
                f"empty={empty} len=[{len_min},{len_max}] mean={len_mean:.1f} "
                f"{elem_str}")
    except Exception as e:  # pragma: no cover
        return f"{name}: array n={n} ERROR={e}"


def _line_for(name: str, col: pa.Array) -> str:
    """Dispatch to scalar / array based on type."""
    t = col.type
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _array_line(name, col)
    return _scalar_line(name, col)


# ─────────────────────────── Cross-checks ────────────────────────────────────


def _label_distribution(col: pa.Array) -> Dict[int, int]:
    try:
        v = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        unique, counts = np.unique(v, return_counts=True)
        return {int(k): int(c) for k, c in zip(unique.tolist(), counts.tolist())}
    except Exception:
        return {}


def _time_gap_stats(ts: np.ndarray, lt: np.ndarray) -> Dict[str, Any]:
    gap = (lt - ts).astype(np.int64)
    return {
        "min": int(gap.min()),
        "p50": int(np.median(gap)),
        "p95": int(np.percentile(gap, 95)),
        "max": int(gap.max()),
        "negative_count": int((gap < 0).sum()),
    }


def _check_seq_sorted(col: pa.ListArray, sample_rows: int = 1000) -> Tuple[int, int, int]:
    """Return (asc_count, desc_count, unsorted_count) over a sample."""
    col = _as_array(col)
    asc = desc = uns = 0
    n = min(len(col), sample_rows)
    for i in range(n):
        sub = col[i]
        if sub is None or not sub.is_valid:
            continue
        try:
            py = sub.as_py()
            if not py:
                continue
            # Drop None and non-positive (treated as padding/missing)
            arr = np.fromiter((x for x in py if x is not None and x > 0),
                              dtype=np.int64)
        except Exception:
            continue
        if len(arr) < 2:
            continue
        if np.all(arr[:-1] <= arr[1:]):
            asc += 1
        elif np.all(arr[:-1] >= arr[1:]):
            desc += 1
        else:
            uns += 1
    return asc, desc, uns


# ─────────────────────────── Top-level entry ─────────────────────────────────


def log_data_stats(data_dir: str, max_rows: int = 100_000) -> None:
    """Sample a head of the parquet files in ``data_dir`` and emit a
    compact per-column summary via ``logging.info``.

    Safe to call before/after dataset construction. Wraps every column
    in a try/except so one bad column does not abort the rest.
    """
    t0 = time.time()
    logging.info("=" * 78)
    logging.info(f"Data stats: sampling up to {max_rows} rows from {data_dir}")
    try:
        table, n_files, n_rg = _sample_head(data_dir, max_rows)
    except Exception as e:
        logging.warning(f"Data stats: sampling failed ({e}), skipping")
        return

    n = table.num_rows
    cols = table.column_names

    # ---- Categorize columns ----
    id_label = [c for c in ['user_id', 'item_id', 'label_type', 'label_time', 'timestamp']
                if c in cols]
    user_int = sorted([c for c in cols if c.startswith('user_int_feats_')],
                      key=lambda x: int(x.rsplit('_', 1)[-1]))
    item_int = sorted([c for c in cols if c.startswith('item_int_feats_')],
                      key=lambda x: int(x.rsplit('_', 1)[-1]))
    user_dense = sorted([c for c in cols if c.startswith('user_dense_feats_')],
                        key=lambda x: int(x.rsplit('_', 1)[-1]))
    seq_a = sorted([c for c in cols if c.startswith('domain_a_seq_')],
                   key=lambda x: int(x.rsplit('_', 1)[-1]))
    seq_b = sorted([c for c in cols if c.startswith('domain_b_seq_')],
                   key=lambda x: int(x.rsplit('_', 1)[-1]))
    seq_c = sorted([c for c in cols if c.startswith('domain_c_seq_')],
                   key=lambda x: int(x.rsplit('_', 1)[-1]))
    seq_d = sorted([c for c in cols if c.startswith('domain_d_seq_')],
                   key=lambda x: int(x.rsplit('_', 1)[-1]))

    logging.info(
        f"Sampled {n} rows from {n_files} parquet file(s), {n_rg} row group(s); "
        f"{len(cols)} columns total")
    logging.info(
        f"Column counts: id_label={len(id_label)} user_int={len(user_int)} "
        f"item_int={len(item_int)} user_dense={len(user_dense)} "
        f"seq_a={len(seq_a)} seq_b={len(seq_b)} seq_c={len(seq_c)} seq_d={len(seq_d)}")

    # ---- ID & Label ----
    logging.info("--- ID & Label ---")
    for c in id_label:
        try:
            logging.info("  " + _line_for(c, table.column(c)))
        except Exception as e:
            logging.warning(f"  {c}: ERROR={e}")

    # Label distribution
    if 'label_type' in cols:
        try:
            dist = _label_distribution(table.column('label_type'))
            total = sum(dist.values()) or 1
            pct = {k: f"{v}({v / total:.1%})" for k, v in dist.items()}
            logging.info(f"  label_type distribution: {pct}")
        except Exception as e:
            logging.warning(f"  label_type dist ERROR={e}")

    # Time-gap stats
    if 'timestamp' in cols and 'label_time' in cols:
        try:
            ts = table.column('timestamp').to_numpy().astype(np.int64)
            lt = (table.column('label_time').fill_null(0)
                  .to_numpy(zero_copy_only=False).astype(np.int64))
            tspan_days = (ts.max() - ts.min()) / 86400.0
            logging.info(
                f"  timestamp span: {tspan_days:.2f} days "
                f"(min={ts.min()} max={ts.max()})")
            gap = _time_gap_stats(ts, lt)
            logging.info(f"  label_time - timestamp (sec): {gap}")
        except Exception as e:
            logging.warning(f"  time-gap ERROR={e}")

    # ---- User Int ----
    if user_int:
        logging.info(f"--- User Int ({len(user_int)} cols) ---")
        for c in user_int:
            try:
                logging.info("  " + _line_for(c, table.column(c)))
            except Exception as e:
                logging.warning(f"  {c}: ERROR={e}")

    # ---- Item Int ----
    if item_int:
        logging.info(f"--- Item Int ({len(item_int)} cols) ---")
        for c in item_int:
            try:
                logging.info("  " + _line_for(c, table.column(c)))
            except Exception as e:
                logging.warning(f"  {c}: ERROR={e}")

    # ---- User Dense ----
    if user_dense:
        logging.info(f"--- User Dense ({len(user_dense)} cols) ---")
        for c in user_dense:
            try:
                logging.info("  " + _line_for(c, table.column(c)))
            except Exception as e:
                logging.warning(f"  {c}: ERROR={e}")

    # ---- Domain Sequences ----
    for d_name, d_cols in [('A', seq_a), ('B', seq_b), ('C', seq_c), ('D', seq_d)]:
        if not d_cols:
            continue
        logging.info(f"--- Domain {d_name} sequences ({len(d_cols)} cols) ---")
        for c in d_cols:
            try:
                logging.info("  " + _line_for(c, table.column(c)))
            except Exception as e:
                logging.warning(f"  {c}: ERROR={e}")

    # ---- Cross-checks ----
    logging.info("--- Cross-checks ---")

    # 1. Pretrained user-dense presence (fid 61 = SUM, fid 87 = LFM4Ads)
    for fid, label in [(61, 'SUM (Meta)'), (87, 'LFM4Ads (Tencent)')]:
        col_name = f'user_dense_feats_{fid}'
        if col_name in cols:
            try:
                col = _as_array(table.column(col_name))
                offsets = col.offsets.to_numpy()
                lengths = (offsets[1:] - offsets[:-1])
                logging.info(
                    f"  Pretrained {label} (fid={fid}): present, "
                    f"len_max={int(lengths.max()) if lengths.size else 0}, "
                    f"null={col.null_count}/{len(col)}")
            except Exception as e:
                logging.warning(f"  Pretrained fid={fid} check ERROR={e}")
        else:
            logging.warning(f"  Pretrained {label} (fid={fid}): MISSING")

    # 2. Sequence sortedness for the timestamp fid in each domain
    seq_ts_fids = [('A', 'domain_a_seq_39'), ('B', 'domain_b_seq_67'),
                   ('C', 'domain_c_seq_27'), ('D', 'domain_d_seq_26')]
    for d_name, ts_col in seq_ts_fids:
        if ts_col not in cols:
            continue
        try:
            asc, desc, uns = _check_seq_sorted(table.column(ts_col))
            total = asc + desc + uns
            if total == 0:
                logging.info(f"  Domain {d_name} ts ({ts_col}): no usable sample")
            else:
                logging.info(
                    f"  Domain {d_name} ts ({ts_col}) sortedness "
                    f"(of {total}): asc={asc} desc={desc} unsorted={uns}")
        except Exception as e:
            logging.warning(f"  Domain {d_name} sortedness ERROR={e}")

    # 3. High-cardinality columns (warn if exceeds emb_skip_threshold default)
    high_card_thresh = 1_000_000
    for c in cols:
        try:
            col = _as_array(table.column(c))
            if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                vals = col.values
                if len(vals) == 0:
                    continue
                v = vals.to_numpy()
                if v.dtype.kind in 'iu':
                    mx = int(v.max())
                    if mx > high_card_thresh:
                        logging.info(
                            f"  HIGH-CARD: {c} max value={_fmt_num(mx)} "
                            f"(> {_fmt_num(high_card_thresh)} default emb_skip_threshold)")
            else:
                v = col.fill_null(0).to_numpy(zero_copy_only=False)
                if v.dtype.kind in 'iuf':
                    if v.dtype.kind == 'f':
                        v = v[~np.isnan(v)]
                    if v.size == 0:
                        continue
                    mx = int(v.max())
                    if mx > high_card_thresh:
                        logging.info(
                            f"  HIGH-CARD: {c} max value={_fmt_num(mx)}")
        except Exception:
            pass

    elapsed = time.time() - t0
    logging.info(f"Data stats done in {elapsed:.1f}s")
    logging.info("=" * 78)
