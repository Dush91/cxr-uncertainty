"""Production UQ variant end-to-end: TS-only calibration + Mahalanobis flag.

Runs the registered production variant (``cxr_uncertainty/production.py``) on the
leak-free eval sites **from the saved Phase-4-step-6 arrays** — no inference
needed (runs in seconds). This is the demonstration that the production
confident-error design (plan §K) is wired through the standard
``Records`` / ``EvalReport`` / ``auroc_at_confidence_levels`` machinery and that
``confident_error_auroc_mahalanobis`` is a first-class metric.

Design (plan §K / FINDINGS.md):
  * **Calibration = TS-only** on the pooled mean (``build_pipeline(use_beta=False,
    use_ts=True)``); disagreement (epistemic_std / MI) is recorded as the baseline
    signal but is NOT the flag.
  * **Confident-error flag = Mahalanobis** on frozen RAD-DINO ``[CLS]`` features
    (``MahalanobisOOD``, 2 class-conditional Ledoit-Wolf Gaussians fit on OpenI-C
    Pneumonia pos/neg). ``build_records(unc_column="mahalanobis")`` makes the flag
    the feature-density score; ``evaluate_records`` reports
    ``confident_error_auroc_mahalanobis``.

Sites (cal = OpenI role-C in-distribution; eval =):
  * openiD  — in-distribution, all valid classes  +  Pneumonia-only.
  * kermany — OOD under shift, Pneumonia-only.
  * covid   — OOD under shift, Pneumonia-only.

Acceptance: Mahalanobis confident-error AUROC > epistemic_std on Kermany/COVID
(reproduces step 6: ~0.82 / ~0.65 vs ~0.52 / ~0.48) and > disagreement
in-distribution; the re-fit Mahalanobis reproduces the saved step-6 scores.

Usage:
    PYTHONPATH=. python scripts/eval_production.py \
        --arrays-dir runs/phase4_features --out runs/production
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.production import build_ts_only_mahalanobis_pipeline
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds

PNEUMONIA = "Pneumonia"
PNEUMONIA_IDX = NIH_PATHOLOGIES.index(PNEUMONIA)


def _load(arrays_dir: Path, site: str):
    z = np.load(arrays_dir / f"arrays_{site}.npz", allow_pickle=True)
    return {
        "probs": z["probs"], "gt": z["gt"], "valid": z["valid"],
        "feats": z["rad_feats"], "mahalanobis_saved": z["mahalanobis"],
        "ids": [str(s) for s in z["ids"]],
    }


def eval_site(C, site, pathologies, member_keys, conf_pct, unc_pct,
              pneumonia_only=True, return_records=False):
    """Run the production variant on one eval site (cal = OpenI-C). Returns
    (report, selectivity, meta) or, when ``return_records=True``, the 4-tuple
    ``(report, selectivity, meta, records)``. Default keeps the 3-tuple so the
    existing ``test_eval_production_reproduces_step6`` (which unpacks 3) is
    unchanged."""
    df_cal, df_eval, meta = build_ts_only_mahalanobis_pipeline(
        C["probs"], C["gt"], C["valid"], C["ids"], C["feats"],
        site["probs"], site["gt"], site["valid"], site["ids"], site["feats"],
        pathologies, member_keys, PNEUMONIA_IDX)

    if pneumonia_only:
        df_eval = df_eval[df_eval.pathology == PNEUMONIA].reset_index(drop=True)
        df_cal = df_cal[df_cal.pathology == PNEUMONIA].reset_index(drop=True)
    else:
        # drop classes with no eval positives (e.g. Consolidation on OpenI)
        pos_per = df_eval.groupby("pathology")["gt"].sum()
        unmappable = pos_per[pos_per == 0].index.tolist()
        if unmappable:
            df_eval = df_eval[~df_eval.pathology.isin(unmappable)].reset_index(drop=True)
            df_cal = df_cal[~df_cal.pathology.isin(unmappable)].reset_index(drop=True)
    if df_eval.empty:
        return None

    thr = calibrate_thresholds(df_cal)
    rec, build_meta = build_records(df_eval, thr, conf_pct, unc_pct,
                                    unc_column="mahalanobis")
    rep = evaluate_records(rec, RiskConfig(device="cpu"),
                           n_imgs=len(set(rec.image_id)))
    sel = auroc_at_confidence_levels(rec)
    meta.update({"thresholds": {k: round(v, 4) for k, v in thr.items()},
                 **build_meta})
    # cross-check: the re-fit Mahalanobis should reproduce the saved step-6 score.
    id_score = dict(zip(rec.image_id, rec.mahalanobis))
    saved = dict(zip(site["ids"], site["mahalanobis_saved"]))
    diffs = [abs(id_score[i] - saved[i]) for i in id_score if i in saved]
    meta["maha_reproducibility"] = {
        "max_abs_diff_vs_saved": float(max(diffs)) if diffs else None,
        "mean_abs_diff_vs_saved": float(np.mean(diffs)) if diffs else None,
    }
    if return_records:
        return rep, sel, meta, rec
    return rep, sel, meta


def print_site(title, rep, sel):
    print(f"\n{'='*84}\n{title}\n{'='*84}")
    print(f"records={rep.n_records}  confident={rep.n_confident}  "
          f"confident_wrong={rep.n_confident_wrong}")
    print(f"confident-error AUROC (epistemic_std) : {rep.confident_error_auroc_std}")
    print(f"confident-error AUROC (mutual_info)   : {rep.confident_error_auroc_mi}")
    print(f"confident-error AUROC (mahalanobis)   : {rep.confident_error_auroc_mahalanobis}  <- production flag")
    print(f"flag recall (risk_flag=mahalanobis)   : {rep.confident_error_recall}")
    print(f"false-flag rate (confident correct)  : {rep.false_flag_rate}")
    print(f"AURC (risk-coverage, epistemic_std)   : {rep.aurc}")
    print(f"ECE (TS-only calibrated p_bar)        : {rep.ece:.4f}")
    print("-" * 84)
    cols = ["auroc_epistemic_std", "auroc_mutual_info", "auroc_mahalanobis"]
    print(f"{'top%':5s} {'n':>6s} {'wrong':>6s} {'AUROC_std':>10s} {'AUROC_mi':>10s} {'AUROC_maha':>11s}")
    for s in sel:
        cells = []
        for c in cols:
            v = s.get(c)
            cells.append("--" if v is None else f"{v:.4f}")
        print(f"{s['conf_top_pct']:5d} {s['n_confident']:6d} {s['n_confident_wrong']:6d} "
              f"{cells[0]:>10s} {cells[1]:>10s} {cells[2]:>11s}")
    print("=" * 84)


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_production", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays-dir", default="runs/phase4_features")
    p.add_argument("--out", default="runs/production")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2,raddino",
                   help="member keys (must match the order the saved probs were stacked in)")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--write-records", action="store_true",
                   help="also write runs/production/per_record.csv for the openiD_all "
                        "(in-distribution, all classes) site — a real per_record.csv "
                        "with image_ids matching the features sidecar, so "
                        "`reanalyze --calibrator ts_only_mahalanobis --features` "
                        "runs the real Mahalanobis flag from the CLI (plan §M.1). "
                        "No inference; just the records already built.")
    args = p.parse_args(argv)

    arrays_dir = Path(args.arrays_dir)
    member_keys = args.arch_ensemble.split(",")
    pathologies = list(NIH_PATHOLOGIES)

    C = _load(arrays_dir, "openiC")
    assert C["probs"].shape[1] == len(member_keys), (
        f"saved probs have {C['probs'].shape[1]} members; --arch-ensemble has "
        f"{len(member_keys)} ({member_keys})")
    print(f"[data] cal=OpenI-C n={len(C['ids'])}  feats={C['feats'].shape}  "
          f"pneumonia pos={int(C['gt'][:, PNEUMONIA_IDX].sum())}")
    print(f"[model] ensemble={member_keys}  pathologies={len(pathologies)} "
          f"flag=Mahalanobis(fit on Pneumonia idx={PNEUMONIA_IDX})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("openiD_all", "openiD (in-distribution, all valid classes)", "openiD", False),
        ("openiD_pneu", "openiD (in-distribution, Pneumonia only)", "openiD", True),
        ("kermany", "kermany (OOD under shift, Pneumonia only)", "kermany", True),
        ("covid", "covid (OOD under shift, Pneumonia only)", "covid", True),
    ]
    summary = {}
    for key, title, site_name, pn_only in targets:
        site = _load(arrays_dir, site_name)
        # request records only when needed: --write-records writes per_record.csv
        # for the in-distribution all-classes site (openiD_all).
        want_records = args.write_records and key == "openiD_all"
        res = eval_site(C, site, pathologies, member_keys, args.conf_pct,
                        args.unc_pct, pneumonia_only=pn_only,
                        return_records=want_records)
        if res is None:
            print(f"[skip] {title}: no eval rows")
            continue
        if want_records:
            rep, sel, meta, rec = res
        else:
            rep, sel, meta = res
        print_site(title, rep, sel)
        if want_records:
            from cxr_uncertainty.evaluate import save_report
            save_report(rep, str(out_dir), records=rec)
            print(f"[write-records] wrote {out_dir}/per_record.csv "
                  f"({len(rec.image_id)} rows) for the features-sidecar CLI path")
        summary[key] = {
            "title": title, "n_images": int(len(site["ids"])),
            "pneumonia_prevalence": float(np.mean(site["gt"][:, PNEUMONIA_IDX])),
            "report": rep.to_dict(), "selectivity": sel, "meta": meta,
        }
        with open(out_dir / f"evaluation_{key}.json", "w") as f:
            json.dump(summary[key], f, indent=2)

    with open(out_dir / "production_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- headline: production Mahalanobis flag vs disagreement baselines --------
    print("\n" + "=" * 92)
    print("PRODUCTION HEADLINE — confident-error AUROC: Mahalanobis flag vs disagreement (top-10%)")
    print("=" * 92)
    print(f"{'site':16s} {'n':>6s} {'prev':>6s} {'cw':>5s} "
          f"{'epist_std':>10s} {'MI':>10s} {'mahalanobis':>12s} {'ECE':>7s}")
    for key, d in summary.items():
        r = d["report"]
        sel = d["selectivity"]
        row10 = next((s for s in sel if s["conf_top_pct"] == 10), {})
        def g(c):
            v = row10.get(c)
            return f"{v:.4f}" if v is not None else "--"
        print(f"{key:16s} {d['n_images']:6d} {d['pneumonia_prevalence']:6.3f} "
              f"{r['n_confident_wrong']:5d} {g('auroc_epistemic_std'):>10s} "
              f"{g('auroc_mutual_info'):>10s} {g('auroc_mahalanobis'):>12s} "
              f"{r['ece']:7.4f}")
    print("-" * 92)
    print("chance = 0.5000. Under shift (kermany/covid) Mahalanobis should beat chance where")
    print("disagreement (epistemic_std/MI) is at/below chance. In-distribution it should beat")
    print("disagreement too. Reproduces Phase-4 step 6 (~0.82/~0.65 OOD; ~0.84 ID).")
    print("=" * 92)
    print(f"[done] wrote {out_dir}/production_summary.json + evaluation_<site>.json")


if __name__ == "__main__":
    main()