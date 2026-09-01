#!/usr/bin/env python
"""Quote-safe sharder for the CheXpert labeler input CSV (Phase 6).

The labeler input written by ``label_rexgradient.py --input-only`` is a
headerless, single-column, QUOTE_ALL CSV -- one report per row, where a report
can contain embedded commas / newlines / quotes (properly csv-quoted). The
naive ``split -l`` would split MID-record on a multi-line report and corrupt
it. This script reads with the ``csv`` module (which honors the quoting) and
writes N contiguous shards preserving row order, so the parallel labeler
wrapper can run one Docker container per shard and ``cat`` the labeled shards
back in order to reconstruct the full labeled CSV (row order is what
``label_rexgradient.py --parse-only`` rejoins on).

Each shard is itself a valid headerless single-column QUOTE_ALL CSV, so it can
be fed straight to ``label.py --reports_path``.

Usage:
    python scripts/shard_csv.py --input rex_reports_train.csv \
        --out-dir runs/rex_labeling_train/shards --nshards 8
    # -> shard_000.csv .. shard_007.csv under --out-dir
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def shard(input_path: Path, out_dir: Path, nshards: int) -> list:
    """Read the QUOTE_ALL single-col CSV with the csv module (record-aware, so
    embedded newlines/commas inside a quoted report do NOT split a record) and
    write ``nshards`` contiguous shards. Returns the list of shard paths in
    order. Shard sizes differ by at most 1 (earlier shards get the extra)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Count records first to size the shards (csv.reader is a generator; we
    # materialize once -- the train split is ~140k reports, fine in memory as
    # a list of single-element rows).
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    n = len(rows)
    assert n > 0, f"input CSV {input_path} has no records"
    nshards = min(nshards, n)                      # never more shards than rows
    base = n // nshards
    rem = n % nshards
    paths = []
    i = 0
    for s in range(nshards):
        size = base + (1 if s < rem else 0)         # first `rem` shards get +1
        chunk = rows[i:i + size]
        i += size
        p = out_dir / f"shard_{s:03d}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            for row in chunk:
                # labeler input is single-column; row may be len 1. Re-emit as
                # a single QUOTE_ALL field (csv.writer handles the quoting).
                w.writerow(row if len(row) == 1 else [row[0]] if row else [""])
        paths.append(p)
        print(f"[shard {s:03d}] {len(chunk)} rows -> {p}")
    assert i == n, f"shard split lost rows: covered {i} of {n}"
    print(f"[done] {n} rows -> {nshards} shards under {out_dir}")
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="headerless single-column QUOTE_ALL labeler input CSV "
                         "(from label_rexgradient.py --input-only)")
    ap.add_argument("--out-dir", required=True,
                    help="dir to write shard_000.csv .. shard_<N-1>.csv")
    ap.add_argument("--nshards", type=int, required=True,
                    help="number of shards (one per parallel labeler container; "
                         "typically the host core count)")
    args = ap.parse_args(argv)
    shard(Path(args.input), Path(args.out_dir), args.nshards)


if __name__ == "__main__":
    main()