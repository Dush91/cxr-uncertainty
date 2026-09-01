"""Phase 3 (M=5 power-law knee): 5-member vs 3-member AURC + confident-error.

Plan §C/§F.3/§J.5 Phase-3 acceptance: adding the two new representation families
(M4 Ark+ Swin, M5 BiomedCLIP CLIP) to the 3-member ensemble should NOT hurt, and
ideally improves, the selective-prediction metrics — the M=5 power-law knee
(Lobacheva 2020, exponent ~-0.75; 3 underperforms, 10 marginal at 2x cost).

Acceptance (plan §J.5 Phase 3):
  (a) 5-member AURC <= 3-member AURC  (more members -> lower risk-coverage area)
  (b) top-5% confident-error AUROC up (or at least not down) with 5 members
  (c) Mahalanobis still beats chance under shift  -> handled by eval_production.py
      on the same 5-member arrays (it re-fits Mahalanobis on the 5-member OpenI-C
      RAD-DINO features); this script reports the saved per-site Mahalanobis AUROC
      for reference but the authoritative under-shift check is eval_production.py.

Methodological note (the confound this script avoids):
  Adding Ark+ (native_size=768) raises the ensemble's load resolution to 768, so
  EVERY member (including the 224-native RAD-DINO/ConvNeXt/xrv/BiomedCLIP) now
  receives a 768->224 bilinear downsample instead of a native-224 load. That
  changes the 224 members' inputs relative to the saved 3-member (native-224)
  Phase-4 arrays, so comparing the 5-member numbers to the saved 3-member numbers
  would conflate "more members" with "different input path." Instead, this script
  computes BOTH the full 5-member and a 3-member SUBSET (probs[:, :3, :] =
  {xrv_nih, convnextv2, raddino}) from the SAME 5-member arrays — identical 768
  input regime, only the aggregation differs. That isolates the member-count
  effect with zero confound. (The saved 224-regime 3-member numbers are reported
  separately for reference.)

Usage (after collecting the 5-member arrays):
    PYTHONPATH=. python scripts/eval_phase4_features.py \
        --arch-ensemble xrv_nih,convnextv2,raddino,biomedclip,arkswin \
        --out runs/phase4_features_5mem
    PYTHONPATH=. python scripts/eval_phase3_5member.py \
        --arrays-dir runs/phase4_features_5mem --out runs/phase3_5mem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.calibration import build_pipeline
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds

PNEUMONIA = "Pneumonia"
PNEUMONIA_IDX = NIH_PATHOLOGIES.index(PNEUMONIA)


def _load(arrays_dir: Path, site: str):
    z = np.load(arrays_dir / f"arrays_{site}.npz", allow_pickle=True)
    return {
        "probs": z["probs"], "gt": z["gt"], "valid": z["valid"],
        "feats": z["rad_feats"], "mahalanobis": z["mahalanobis"],
        "ids": [str(s) for s in z["ids"]],
    }


def _run_member_set(C, site, pathologies, member_keys, conf_pct, unc_pct,
                    pneumonia_only):
    """TS-only calibration (cal=OpenI-C) + Youden + build_records + evaluate, on
    one member set (a slice of the saved probs). Returns (report, selectivity)."""
    # build_pipeline order: probs_cal, gt_cal, valid_cal, probs_eval, gt_eval,
    # valid_eval, image_ids_cal, image_ids_eval, pathologies, member_keys, ...
    df_cal, df_eval, _meta = build_pipeline(
        C["probs"], C["gt"], C["valid"],
        site["probs"], site["gt"], site["valid"],
        C["ids"], site["ids"],
        pathologies, member_keys, use_beta=False, use_ts=True)
    if pneumonia_only:
        df_eval = df_eval[df_eval.pathology == PNEUMONIA].reset_index(drop=True)
        df_cal = df_cal[df_cal.pathology == PNEUMONIA].reset_index(drop=True)
    else:
        pos_per = df_eval.groupby("pathology")["gt"].sum()
        unmappable = pos_per[pos_per == 0].index.tolist()
        if unmappable:
            df_eval = df_eval[~df_eval.pathology.isin(unmappable)].reset_index(drop=True)
            df_cal = df_cal[~df_cal.pathology.isin(unmappable)].reset_index(drop=True)
    if df_eval.empty:
        return None
    thr = calibrate_thresholds(df_cal)
    rec, _ = build_records(df_eval, thr, conf_pct, unc_pct)  # epistemic_std flag
    rep = evaluate_records(rec, RiskConfig(device="cpu"),
                           n_imgs=len(set(rec.image_id)))
    sel = auroc_at_confidence_levels(rec)
    return rep, sel


def _top5_auroc(sel):
    row = next((s for s in sel if s["conf_top_pct"] == 5), None)
    if row is None:
        return None
    v = row.get("auroc_epistemic_std")
    return float(v) if v is not None else None


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_phase3_5member",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays-dir", default="runs/phase4_features_5mem")
    p.add_argument("--out", default="runs/phase3_5mem")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2,raddino,biomedclip,arkswin",
                   help="member keys in the order the saved probs were stacked")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    args = p.parse_args(argv)

    arrays_dir = Path(args.arrays_dir)
    all_keys = args.arch_ensemble.split(",")
    subset_keys = all_keys[:3]          # {xrv_nih, convnextv2, raddino} — the Phase-4 trio
    pathologies = list(NIH_PATHOLOGIES)

    C = _load(arrays_dir, "openiC")
    M = C["probs"].shape[1]
    assert M == len(all_keys), (
        f"saved probs have {M} members; --arch-ensemble has {len(all_keys)} ({all_keys})")
    print(f"[data] cal=OpenI-C n={len(C['ids'])}  probs={C['probs'].shape}  "
          f"feats={C['feats'].shape}")
    print(f"[model] 5-member={all_keys}  3-member-subset={subset_keys}")
    print(f"[note] both runs use the SAME 768-regime arrays; the 3-member subset is "
          f"probs[:, :3, :] — isolates the member-count effect (no input-path confound).")

    # Slice the cal arrays for the 3-member subset (reuse the same dict shape).
    C3 = {**C, "probs": C["probs"][:, :3, :]}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("openiD_all", "openiD (in-distribution, all valid classes)", "openiD", False),
        ("openiD_pneu", "openiD (in-distribution, Pneumonia only)", "openiD", True),
        ("kermany", "kermany (OOD under shift, Pneumonia only)", "kermany", True),
        ("covid", "covid (OOD under shift, Pneumonia only)", "covid", True),
    ]
    summary = {}
    print("\n" + "=" * 92)
    print("PHASE 3 (M=5 knee) — 5-member vs 3-member-subset (matched 768 regime)")
    print("=" * 92)
    hdr = (f"{'site':16s} {'n':>6s} {'prev':>6s} | "
           f"{'AURC_3':>8s} {'AURC_5':>8s} {'dAURC':>7s} | "
           f"{'top5_3':>8s} {'top5_5':>8s} {'dTop5':>7s}")
    print(hdr)
    print("-" * 92)
    for key, title, site_name, pn_only in targets:
        site = _load(arrays_dir, site_name)
        site3 = {**site, "probs": site["probs"][:, :3, :]}
        r3 = _run_member_set(C3, site3, pathologies, subset_keys, args.conf_pct,
                             args.unc_pct, pneumonia_only=pn_only)
        r5 = _run_member_set(C, site, pathologies, all_keys, args.conf_pct,
                             args.unc_pct, pneumonia_only=pn_only)
        if r3 is None or r5 is None:
            print(f"[skip] {title}: no eval rows")
            continue
        rep3, sel3 = r3
        rep5, sel5 = r5
        aurc3, aurc5 = rep3.aurc, rep5.aurc
        t5_3, t5_5 = _top5_auroc(sel3), _top5_auroc(sel5)
        prev = float(np.mean(site["gt"][:, PNEUMONIA_IDX]))
        daurc = (aurc5 - aurc3) if (aurc3 is not None and aurc5 is not None) else None
        dtop5 = (t5_5 - t5_3) if (t5_3 is not None and t5_5 is not None) else None

        def f(v, spec="8.4f"):
            return f"{v:{spec}}" if isinstance(v, (int, float)) and v is not None else f"{'--':>8s}"

        print(f"{key:16s} {len(site['ids']):6d} {prev:6.3f} | "
              f"{f(aurc3)} {f(aurc5)} {f(daurc, '7.4f')} | "
              f"{f(t5_3)} {f(t5_5)} {f(dtop5, '7.4f')}")
        summary[key] = {
            "title": title, "n_images": int(len(site["ids"])),
            "pneumonia_prevalence": prev,
            "3mem": {"aurc": aurc3, "top5_confident_error_auroc_std": t5_3,
                     "ece": rep3.ece, "report": rep3.to_dict()},
            "5mem": {"aurc": aurc5, "top5_confident_error_auroc_std": t5_5,
                     "ece": rep5.ece, "report": rep5.to_dict()},
            "delta_aurc_5minus3": daurc, "delta_top5_5minus3": dtop5,
        }

    # --- acceptance verdict ----------------------------------------------------
    print("-" * 92)
    id_all = summary.get("openiD_all")
    id_pneu = summary.get("openiD_pneu")
    verdicts = []
    if id_all and id_all["3mem"]["aurc"] is not None and id_all["5mem"]["aurc"] is not None:
        ok = id_all["5mem"]["aurc"] <= id_all["3mem"]["aurc"]
        verdicts.append(f"(a) AURC 5<=3 (OpenI-D all): {ok}  "
                        f"({id_all['5mem']['aurc']:.4f} vs {id_all['3mem']['aurc']:.4f})")
    if id_pneu and id_pneu["3mem"]["top5_confident_error_auroc_std"] is not None \
            and id_pneu["5mem"]["top5_confident_error_auroc_std"] is not None:
        t3, t5 = id_pneu["3mem"]["top5_confident_error_auroc_std"], id_pneu["5mem"]["top5_confident_error_auroc_std"]
        ok = t5 >= t3
        verdicts.append(f"(b) top-5% confident-error AUROC 5>=3 (OpenI-D pneumo): {ok}  "
                        f"({t5:.4f} vs {t3:.4f})")
    # (c) Mahalanobis under shift: report saved per-site, authoritative via eval_production.py
    for sname in ("kermany", "covid"):
        s = summary.get(sname)
        if s:
            print(f"    [ref] {sname}: 5mem top-5% confident-error AUROC (std) = "
                  f"{s['5mem']['top5_confident_error_auroc_std']}  (Mahalanobis under-shift "
                  f"check: run eval_production.py --arrays-dir {arrays_dir})")
    print("\nACCEPTANCE:")
    for v in verdicts:
        print("  " + v)
    print("  (c) Mahalanobis beats chance under shift -> see eval_production.py on the "
          "same 5-member arrays (re-fits Mahalanobis on 5-member OpenI-C RAD-DINO feats).")
    print("=" * 92)

    with open(out_dir / "phase3_5member_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {out_dir}/phase3_5member_summary.json")


if __name__ == "__main__":
    main()