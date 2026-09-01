"""Build the RAD-DINO features sidecar (plan §M.1) — no inference.

Concatenates the ``rad_feats`` + ``ids`` already saved in
``runs/phase4_features/arrays_{openiC,openiD,kermany,covid}.npz`` (from Phase-4
step 6) into ONE npz, ``runs/phase4_features/features_raddino.npz``, with
``rad_feats (N_all, 1536)`` + ``ids (N_all,)`` covering every image across the
four sites. ``reanalyze --features <this>`` then subsets it to the cal/eval
sorted-unique ``image_id`` order so ``--calibrator ts_only_mahalanobis`` runs the
real Mahalanobis flag from the CLI (closing the §K gap: today the CLI path
honestly falls back to ``ts_only`` because the RAD-DINO ``[CLS]`` features are
not in the per_record CSV).

Dedup by id (keep first) so a CSV that spans sites doesn't double-count an image
that appears in two saved arrays (it doesn't today, but defensively). No model
load, no forward — this is a pure array concatenation.

Usage:
    PYTHONPATH=. python scripts/build_features_sidecar.py
    PYTHONPATH=. python scripts/build_features_sidecar.py \
        --arrays-dir runs/phase4_features --out runs/phase4_features/features_raddino.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SITES = ["openiC", "openiD", "kermany", "covid"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="build_features_sidecar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays-dir", default="runs/phase4_features")
    p.add_argument("--out", default="runs/phase4_features/features_raddino.npz")
    args = p.parse_args(argv)

    arrays_dir = Path(args.arrays_dir)
    feats_list, ids_list = [], []
    seen = set()
    for site in SITES:
        f = arrays_dir / f"arrays_{site}.npz"
        if not f.exists():
            print(f"[skip] {site}: {f} not found")
            continue
        z = np.load(f, allow_pickle=True)
        feats = np.asarray(z["rad_feats"], dtype=np.float64)
        ids = [str(s) for s in z["ids"]]
        assert feats.shape[0] == len(ids), (
            f"{site}: rad_feats has {feats.shape[0]} rows but ids has {len(ids)}")
        n_before = len(ids)
        # dedup by id, keep first (defensive against an image in two arrays).
        keep = [i for i, iid in enumerate(ids) if iid not in seen]
        for i in keep:
            seen.add(ids[i])
        feats_list.append(feats[keep])
        ids_list.extend(ids[i] for i in keep)
        print(f"[{site}] {n_before} rows -> {len(keep)} kept "
              f"(feat dim {feats.shape[1] if feats.ndim > 1 else '?'}); "
              f"{n_before - len(keep)} dup dropped")

    if not feats_list:
        raise SystemExit(f"no arrays found in {arrays_dir}")

    rad_feats = np.concatenate(feats_list, axis=0)
    ids = np.array(ids_list, dtype=object)
    assert rad_feats.shape[0] == len(ids) == len(seen), (
        f"counts disagree: feats={rad_feats.shape[0]} ids={len(ids)} unique={len(seen)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rad_feats=rad_feats, ids=ids)
    print(f"\n[done] wrote {out}: rad_feats={rad_feats.shape} unique_ids={len(ids)}")


if __name__ == "__main__":
    main()