"""Phase 4: does the confident-error flag survive distribution shift?

The Phase-3 finding was that TS-only calibration (temperature the raw pooled mean,
keep raw cross-member std for the flag) is the production design: it fixes ECE
~10x while preserving the top-end disagreement signal on in-distribution OpenI.
But in-distribution OpenI is under-powered at the confident end (only a handful of
confident errors), and the ``confident-agree-wrong`` core is provably invisible to
disagreement (Abe NeurIPS 2022). Phase 4 asks the real question on **powered,
leak-free OOD probes** where distribution shift generates many confident errors:

  * Kermany pediatric pneumonia  (5856 imgs, pediatric + scanner shift)
  * COVID-19 radiography          (6432 imgs, COVID lesions no encoder saw)

Both are leak-free (no encoder's pretrain corpus includes them) and reduce to the
NIH **Pneumonia** class (gt=1 for PNEUMONIA/COVID19, 0 for NORMAL; only Pneumonia
is marked valid). Design:

  * Calibrate **TS-only** on the in-distribution OpenI role-C arrays
    (``runs/phase3/arrays.npz``; T is fit there, leak-free).
  * Forward each OOD site through the 3-member ensemble -> (N,M,P).
  * ``build_pipeline(use_beta=False, use_ts=True)`` with ``probs_cal`` = OpenI-C,
    ``probs_eval`` = OOD: disagreement computed on the RAW member probs (the
    signal that discriminates errors, per Phase 3), pooled mean temperature-scaled
    by the OpenI-fit T (monotone, §E.2).
  * Youden thresholds fit on OpenI-C, applied to OOD.
  * Per-site **Pneumonia-class** confident-error AUROC + selectivity + ECE.

This is the acceptance run for plan §J.5 Phase 4: per-site OOD confident-error
AUROC reported; leak-free assertion holds; is the flag catching confident errors
under shift (where there are hundreds, not 2)?

Usage:
    PYTHONPATH=. python scripts/eval_phase4.py --arch-ensemble xrv_nih,convnextv2,raddino \
        --calib-arrays runs/phase3/arrays.npz --ood-manifest data/ood_manifest.parquet \
        --out runs/phase4
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.calibration import build_pipeline
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.models import CXREnsemble
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds
from cxr_uncertainty.utils import load_image_tensor

warnings.filterwarnings("ignore")
PNEUMONIA = "Pneumonia"


@torch.no_grad()
def forward_site(df, ensemble, cfg, device, site):
    """Forward every image in ``df``; return (N,M,P) per-member probs + (N,P) gt/valid + ids."""
    probs, gts, valids, ids = [], [], [], []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            x = load_image_tensor(row.image_path, img_size=cfg.img_size, device=device)
        except Exception as e:
            print(f"  [skip] {row.image_id}: {e}")
            continue
        eo = ensemble.forward_ensemble_full(x)
        probs.append(eo.per_member_probs[:, 0, :].cpu().numpy().astype(np.float64))
        gts.append(np.asarray(row.labels, dtype=np.float64))
        valids.append(np.asarray(row.valid, dtype=bool))
        ids.append(str(row.image_id))
        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            dt = time.time() - t0
            print(f"  [infer {site}] {i+1}/{len(df)}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(probs), np.stack(gts), np.stack(valids), ids


def _monotonic(sel):
    prev = -1.0
    for s in sel:
        if s["auroc_epistemic_std"] < prev - 1e-9:
            return False
        prev = s["auroc_epistemic_std"]
    return True


def eval_pneumonia(df_cal, df_eval, conf_pct, unc_pct, n_imgs, site):
    """Pneumonia-class only: Youden on cal, build_records on eval, evaluate, selectivity."""
    df_eval = df_eval[df_eval.pathology == PNEUMONIA].reset_index(drop=True)
    df_cal = df_cal[df_cal.pathology == PNEUMONIA].reset_index(drop=True)
    if df_eval.empty:
        return None, None
    thr = calibrate_thresholds(df_cal)
    rec, meta = build_records(df_eval, thr, conf_pct, unc_pct)
    rep = evaluate_records(rec, RiskConfig(), n_imgs=n_imgs)
    sel = auroc_at_confidence_levels(rec)
    meta["thresholds"] = {k: round(v, 4) for k, v in thr.items()}
    return rep, sel


def print_site(title, rep, sel):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"records={rep.n_records}  confident={rep.n_confident}  confident_wrong={rep.n_confident_wrong}")
    print(f"confident-error AUROC (epistemic_std) : {rep.confident_error_auroc_std}")
    print(f"confident-error AUROC (mutual_info)   : {rep.confident_error_auroc_mi}")
    print(f"confident-error recall (risk_flag)    : {rep.confident_error_recall}")
    print(f"false-flag rate (confident correct)  : {rep.false_flag_rate}")
    print(f"AURC (risk-coverage)                 : {rep.aurc}")
    print(f"ECE (calibration of p_bar)            : {rep.ece:.4f}")
    print("-" * 70)
    for s in sel:
        print(f"  top {s['conf_top_pct']:2d}%: n={s['n_confident']:5d} "
              f"wrong={s['n_confident_wrong']:4d}  "
              f"AUROC_std={s['auroc_epistemic_std']}  AUROC_mi={s['auroc_mutual_info']}")
    print(f"MONOTONIC non-decreasing 50->5%: {_monotonic(sel)}")
    print("=" * 70)


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_phase4", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib-arrays", default="runs/phase3/arrays.npz")
    p.add_argument("--ood-manifest", default="data/ood_manifest.parquet")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2,raddino")
    p.add_argument("--out", default="runs/phase4")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sites", default="all", help="comma list or 'all'")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device, use_mc_dropout=False)

    # --- load in-distribution OpenI role-C calibration arrays (leak-free) ----
    z = np.load(args.calib_arrays, allow_pickle=True)
    pC, yC, vC, idsC = z["pC"], z["yC"], z["vC"], list(z["idsC"])
    paths = list(z["paths"])
    keys = list(z["keys"])
    # the cal arrays hold ALL 14 classes; keep Pneumonia valid only for OOD, but
    # for the TS fit we use the OpenI valid mask as-is (matches how T was fit in
    # phase 3). The OOD valid mask (only Pneumonia) is set on the eval arrays.
    print(f"[calib] OpenI role-C arrays: {pC.shape}  T re-fit on these (TS-only).")

    df_ood = pd.read_parquet(args.ood_manifest)
    sites = (df_ood.site.unique().tolist() if args.sites == "all"
             else [s.strip() for s in args.sites.split(",")])
    print(f"[ood] sites={sites}  total={len(df_ood)}")

    ensemble = CXREnsemble(args.arch_ensemble.split(","), cfg=cfg)
    print(f"[model] arch ensemble: {ensemble.member_keys} | predictable={len(ensemble.predictable)}")
    member_keys = list(ensemble.member_keys)
    pathologies = list(ensemble.predictable)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for site in sites:
        df_s = df_ood[df_ood.site == site].reset_index(drop=True)
        if args.limit:
            df_s = df_s.iloc[:args.limit]
        print(f"\n[infer] forwarding site={site}  n={len(df_s)} ...")
        pD, yD, vD, idsD = forward_site(df_s, ensemble, cfg, device, site)

        # TS-only pipeline: calibrate on OpenI-C, evaluate on OOD (this site).
        df_cal, df_eval, meta = build_pipeline(
            pC, yC, vC, pD, yD, vD, idsC, idsD, pathologies, member_keys,
            use_beta=False, use_ts=True)
        rep, sel = eval_pneumonia(df_cal, df_eval, args.conf_pct, args.unc_pct,
                                  len(idsD), site)
        if rep is None:
            print(f"[skip] {site}: no Pneumonia rows")
            continue
        print_site(f"{site} — TS-only, Pneumonia class (OOD under shift)", rep, sel)
        summary[site] = {
            "n_images": int(len(idsD)),
            "n_pneumonia_eval": int((df_eval.pathology == PNEUMONIA).sum()),
            "pneumonia_prevalence": float(np.mean(yD[:, pathologies.index(PNEUMONIA)])),
            "report": rep.to_dict(), "selectivity": sel, "meta": meta,
            "monotonic": _monotonic(sel),
            "temperature_T": meta.get("temperature_T"),
        }
        np.savez_compressed(str(out_dir / f"arrays_{site}.npz"),
                            pD=pD, yD=yD, vD=vD, idsD=np.array(idsD, dtype=object))

    with open(out_dir / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- cross-site headline ------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4 HEADLINE — per-site Pneumonia confident-error (TS-only)")
    print("=" * 70)
    print(f"{'site':10s} {'n':>6s} {'prev':>6s} {'cw':>5s} {'AUROC':>7s} "
          f"{'top10':>7s} {'top5':>7s} {'AURC':>8s} {'ECE':>6s} {'monot':>6s}")

    def fmt(v, w, spec="7.3f"):
        return f"{v:{w}{spec}}" if isinstance(v, (int, float)) and v is not None else f"{'--':>{w}s}"

    for s, d in summary.items():
        sel = d["selectivity"]
        def g(pct):
            return next((x["auroc_epistemic_std"] for x in sel if x["conf_top_pct"] == pct), None)
        r = d["report"]
        print(f"{s:10s} {d['n_images']:6d} {d['pneumonia_prevalence']:6.3f} "
              f"{r['n_confident_wrong']:5d} {fmt(r['confident_error_auroc_std'],7)} "
              f"{fmt(g(10),7)} {fmt(g(5),7)} {fmt(r['aurc'],8,'.5f')} "
              f"{fmt(r['ece'],6,'.4f')} {str(d['monotonic']):6s}")
    print("=" * 70)
    print(f"[done] wrote {out_dir}/phase4_summary.json + arrays_<site>.npz")


if __name__ == "__main__":
    main()