"""Phase 4 step 6: feature-based OOD/UQ vs the disagreement flag under shift.

Phase 4 measured the TS-only disagreement flag on powered leak-free OOD probes
(Kermany pediatric pneumonia, COVID-19 radiography) and found it at/below
chance (Pneumonia AUROC 0.478/0.515) -- two compounding failures: (1) the
OpenI-fit temperature calibration shatters under prevalence/scanner shift
(ECE 0.72-0.74); (2) the confident-agree-wrong core (Abe 2022) is invisible to
cross-member disagreement. This script tests whether FEATURE-based OOD scores
-- which do not rely on cross-member disagreement -- can catch confident errors
under shift where the disagreement flag provably cannot.

Scores (``cxr_uncertainty/feature_uq.py``):
  * epistemic_std  -- the TS-only disagreement baseline (Phase 4).
  * energy14       -- 14-class energy from RAD-DINO logits (Liu 2020).
  * mahalanobis    -- 2 class-conditional Gaussians on RAD-DINO 768-d [CLS]
                      features, fit on OpenI role-C Pneumonia pos/neg (Lee 2018).

Design:
  * Forward OpenI role-C (in-distribution FIT set) + role-D (in-distribution
    baseline) + Kermany + COVID (under shift) through the 3-member ensemble,
    collecting RAD-DINO 14 logits + 768-d [CLS] features + per-member probs
    (re-forward needed -- features are not in the saved Phase-3/4 arrays).
  * Fit Mahalanobis on OpenI-C RAD-DINO features + Pneumonia labels.
  * Apply TS-only calibration (build_pipeline use_beta=False, use_ts=True) with
    probs_cal=OpenI-C for every eval site (matches the Phase-4 production path).
  * Per site, Pneumonia-class confident-error AUROC across selectivity for each
    score; OpenI-D additionally reports the all-valid-class overall number.

Acceptance: does any feature-based score beat chance (0.5) under shift where
disagreement is at chance?

Usage:
    PYTHONPATH=. python scripts/eval_phase4_features.py \
        --arch-ensemble xrv_nih,convnextv2,raddino \
        --manifest data/manifest.parquet --ood-manifest data/ood_manifest.parquet \
        --out runs/phase4_features
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
from sklearn.metrics import roc_auc_score

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.calibration import build_pipeline
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.feature_uq import energy_score, MahalanobisOOD, extract_member
from cxr_uncertainty.models import CXREnsemble
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds
from cxr_uncertainty.utils import load_image_tensor

warnings.filterwarnings("ignore")
PNEUMONIA = "Pneumonia"
PNEUMONIA_IDX = NIH_PATHOLOGIES.index(PNEUMONIA)
SELECTIVITY_PCTS = [50, 25, 15, 10, 5]


@torch.no_grad()
def forward_site(df, ensemble, cfg, device, label):
    """Forward every image in ``df``; collect per-member probs (N,M,P), the
    RAD-DINO member's 14 logits (N,14) + 768-d features (N,D), gt (N,P), valid
    (N,P), and ids. Skips images that fail to load (keeps arrays aligned to ids
    so downstream image_id joins are exact)."""
    raddino_key = "raddino" if "raddino" in ensemble.member_keys else None
    probs, rad_logits, rad_feats, gts, valids, ids = [], [], [], [], [], []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            x = load_image_tensor(row.image_path, img_size=cfg.img_size, device=device)
        except Exception as e:
            print(f"  [skip] {row.image_id}: {e}")
            continue
        eo = ensemble.forward_ensemble_full(x)
        probs.append(eo.per_member_probs[:, 0, :].cpu().numpy().astype(np.float64))   # (M,P)
        if raddino_key is not None:
            lg, ft = extract_member(eo, raddino_key)
            rad_logits.append(lg)
            rad_feats.append(ft)
        gts.append(np.asarray(row.labels, dtype=np.float64))
        valids.append(np.asarray(row.valid, dtype=bool))
        ids.append(str(row.image_id))
        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            dt = time.time() - t0
            print(f"  [infer {label}] {i+1}/{len(df)}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {
        "probs": np.stack(probs),                                   # (N,M,P)
        "gt": np.stack(gts),                                        # (N,P)
        "valid": np.stack(valids),                                  # (N,P)
        "ids": ids,
    }
    if rad_logits:
        out["rad_logits"] = np.stack(rad_logits)                   # (N,14)
        out["rad_feats"] = np.stack(rad_feats)                     # (N,D)
    return out


def selectivity_auroc(confidence, wrong, scores):
    """For each top-pct confidence slice, AUROC of every score vs ``wrong``.
    ``scores`` = {name: array(N,)} all oriented HIGH = likely wrong. Returns a
    list of dicts (one per selectivity level) with n_confident,
    n_confident_wrong, and auroc_<name> per score."""
    out = []
    confidence = np.asarray(confidence, dtype=float)
    wrong = np.asarray(wrong, dtype=int)
    for pct in SELECTIVITY_PCTS:
        cut = float(np.percentile(confidence, 100.0 - pct))
        m = confidence >= cut
        n_w = int((m & (wrong == 1)).sum())
        n_c = int((m & (wrong == 0)).sum())
        if n_w == 0 or n_c == 0:
            out.append({"conf_top_pct": pct, "n_confident": int(m.sum()),
                        "n_confident_wrong": n_w, "n_confident_correct": n_c})
            continue
        row = {"conf_top_pct": pct, "n_confident": int(m.sum()),
               "n_confident_wrong": n_w, "n_confident_correct": n_c}
        for name, s in scores.items():
            s = np.asarray(s, dtype=float)
            try:
                row[f"auroc_{name}"] = round(float(roc_auc_score(wrong[m], s[m])), 4)
            except Exception:
                row[f"auroc_{name}"] = None
        out.append(row)
    return out


def eval_site(pC, yC, vC, idsC, site_arrays, pathologies, member_keys,
              conf_pct, unc_pct, pneumonia_only=True):
    """Run TS-only pipeline (cal=OpenI-C) on one eval site, build Pneumonia-only
    (or all-valid) records, attach image-level energy/mahalanobis scores by
    image_id, and return confident-error AUROC across selectivity for each
    score. Returns (report, selectivity, scores_table, meta) or None."""
    pD = site_arrays["probs"]
    yD, vD, idsD = site_arrays["gt"], site_arrays["valid"], site_arrays["ids"]
    df_cal, df_eval, meta = build_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, pathologies, member_keys,
        use_beta=False, use_ts=True)

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
    rec, build_meta = build_records(df_eval, thr, conf_pct, unc_pct)
    rep = evaluate_records(rec, RiskConfig(device="cpu"), n_imgs=len(set(rec.image_id)))
    a = rec.arrays()
    confidence, wrong = a["confidence"], a["wrong"]
    ep, mi = a["epistemic_std"], a["mutual_info"]

    # image-level feature scores -> align by image_id to the long-form rows.
    id_energy = dict(zip(site_arrays["ids"], site_arrays["energy"]))
    id_maha = dict(zip(site_arrays["ids"], site_arrays["mahalanobis"]))
    img_ids = rec.image_id
    energy = np.array([id_energy.get(i, np.nan) for i in img_ids], dtype=float)
    maha = np.array([id_maha.get(i, np.nan) for i in img_ids], dtype=float)
    # rows missing a feature score (image failed forward) -> drop from AUROC.
    has_feat = np.isfinite(energy) & np.isfinite(maha)

    scores = {
        "epistemic_std": ep,
        "mutual_info": mi,
        "energy14": energy,
        "mahalanobis": maha,
    }
    sel = selectivity_auroc(confidence, wrong, scores)

    # A feature-only AUROC restricted to rows that HAVE feature scores, so the
    # energy/mahalanobis numbers are not diluted by NaN rows (the disagreement
    # baseline is recomputed on the same subset for an apples-to-apples table).
    sub = selectivity_auroc(confidence[has_feat], wrong[has_feat],
                            {"epistemic_std": ep[has_feat],
                             "mutual_info": mi[has_feat],
                             "energy14": energy[has_feat],
                             "mahalanobis": maha[has_feat]})

    meta.update({"thresholds": {k: round(v, 4) for k, v in thr.items()},
                 **build_meta})
    return rep, sel, sub, meta


def print_site(title, rep, sel, sub):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"records={rep.n_records}  confident={rep.n_confident}  "
          f"confident_wrong={rep.n_confident_wrong}")
    score_names = ["epistemic_std", "mutual_info", "energy14", "mahalanobis"]
    hdr = f"{'top%':5s} {'n':>6s} {'wrong':>6s}"
    for n in score_names:
        hdr += f" {n:>13s}"
    print(hdr)
    for s in sel:
        line = f"{s['conf_top_pct']:5d} {s['n_confident']:6d} {s['n_confident_wrong']:6d}"
        for n in score_names:
            v = s.get(f"auroc_{n}")
            line += f" {('  --' if v is None else f'{v:13.4f}'):>13s}"
        print(line)
    print("-" * 78 + "\n  (feature-subset: only rows with feature scores, apples-to-apples)")
    for s in sub:
        line = f"{s['conf_top_pct']:5d} {s['n_confident']:6d} {s['n_confident_wrong']:6d}"
        for n in score_names:
            v = s.get(f"auroc_{n}")
            line += f" {('  --' if v is None else f'{v:13.4f}'):>13s}"
        print(line)
    print("=" * 78)


def fmt(v, w, spec="7.3f"):
    return f"{v:{w}{spec}}" if isinstance(v, (int, float)) and v is not None else f"{'--':>{w}s}"


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_phase4_features", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifest.parquet")
    p.add_argument("--ood-manifest", default="data/ood_manifest.parquet")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2,raddino")
    p.add_argument("--out", default="runs/phase4_features")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--limit", type=int, default=0, help="cap images per site (0=all)")
    p.add_argument("--ood-only", action="store_true",
                   help="skip OpenI re-forward (use when you only want Kermany/COVID; "
                        "Mahalanobis then fits on the saved OpenI-C arrays which lack "
                        "features -- so this flag is NOT supported; kept for future use.")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device, use_mc_dropout=False)

    df_man = pd.read_parquet(args.manifest)
    dfC = df_man[df_man.split_role == "C"].reset_index(drop=True)
    dfD = df_man[df_man.split_role == "D"].reset_index(drop=True)
    df_ood = pd.read_parquet(args.ood_manifest)
    print(f"[data] OpenI role-C={len(dfC)} role-D={len(dfD)}  "
          f"OOD kermany={len(df_ood[df_ood.site=='kermany'])} "
          f"covid={len(df_ood[df_ood.site=='covid'])}")

    ensemble = CXREnsemble(args.arch_ensemble.split(","), cfg=cfg)
    # Load images at the largest native_size so a 768 member (Ark+) gets true
    # resolution; forward_ensemble_full resizes per member. 224-only ensembles
    # keep img_size=224 (byte-identical to the saved Phase-4 arrays).
    cfg.img_size = max(getattr(mb, "native_size", 224) for mb in ensemble.members)
    member_keys = list(ensemble.member_keys)
    pathologies = list(ensemble.predictable)
    assert "raddino" in member_keys, "feature-based UQ needs the RAD-DINO member"
    print(f"[model] ensemble={member_keys} predictable={len(pathologies)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def cap(df):
        return df.iloc[:args.limit].reset_index(drop=True) if args.limit else df

    # --- forward all four sites ------------------------------------------------
    print("\n[infer] OpenI role-C (in-distribution FIT set) ...")
    C = forward_site(cap(dfC), ensemble, cfg, device, "openiC")
    print("[infer] OpenI role-D (in-distribution baseline) ...")
    D = forward_site(cap(dfD), ensemble, cfg, device, "openiD")
    print("\n[infer] Kermany (OOD, pediatric pneumonia shift) ...")
    K = forward_site(cap(df_ood[df_ood.site == "kermany"].reset_index(drop=True)),
                     ensemble, cfg, device, "kermany")
    print("\n[infer] COVID (OOD, COVID lesions unseen) ...")
    CO = forward_site(cap(df_ood[df_ood.site == "covid"].reset_index(drop=True)),
                      ensemble, cfg, device, "covid")

    # --- compute feature scores for every site ---------------------------------
    for arr in (C, D, K, CO):
        arr["energy"] = energy_score(arr["rad_logits"])
    print(f"[scores] energy14 computed for all sites "
          f"(range OpenI-C [{C['energy'].min():.2f},{C['energy'].max():.2f}])")

    # --- fit Mahalanobis on OpenI-C Pneumonia pos/neg RAD-DINO features --------
    yC_pn = C["gt"][:, PNEUMONIA_IDX].astype(int)
    maha = MahalanobisOOD().fit(C["rad_feats"], yC_pn)
    print(f"[maha] fit on OpenI-C: classes={maha.classes_.tolist()} "
          f"n_per_class={maha.n_per_class_}")
    for arr, name in [(C, "openiC"), (D, "openiD"), (K, "kermany"), (CO, "covid")]:
        arr["mahalanobis"] = maha.score(arr["rad_feats"])
        print(f"[maha] {name}: score range [{arr['mahalanobis'].min():.2f},"
              f"{arr['mahalanobis'].max():.2f}]  median={np.median(arr['mahalanobis']):.2f}")

    # persist the forward arrays + scores
    def save(arr, name):
        np.savez_compressed(
            str(out_dir / f"arrays_{name}.npz"),
            probs=arr["probs"], gt=arr["gt"], valid=arr["valid"],
            rad_logits=arr["rad_logits"], rad_feats=arr["rad_feats"],
            energy=arr["energy"], mahalanobis=arr["mahalanobis"],
            ids=np.array(arr["ids"], dtype=object))
    save(C, "openiC"); save(D, "openiD"); save(K, "kermany"); save(CO, "covid")

    # --- evaluate each site ----------------------------------------------------
    pC, yC, vC, idsC = C["probs"], C["gt"], C["valid"], C["ids"]
    summary = {}
    eval_targets = [
        ("openiD (in-distribution, all valid classes)", D, False),
        ("openiD (in-distribution, Pneumonia only)", D, True),
        ("kermany (OOD under shift, Pneumonia only)", K, True),
        ("covid (OOD under shift, Pneumonia only)", CO, True),
    ]
    for title, arr, pn_only in eval_targets:
        res = eval_site(pC, yC, vC, idsC, arr, pathologies, member_keys,
                        args.conf_pct, args.unc_pct, pneumonia_only=pn_only)
        if res is None:
            print(f"[skip] {title}: no eval rows")
            continue
        rep, sel, sub, meta = res
        print_site(title, rep, sel, sub)
        key = title.split(" ")[0] + ("_all" if not pn_only else "_pneu")
        summary[key] = {
            "title": title, "n_images": int(len(arr["ids"])),
            "n_eval_rows": int(rep.n_records),
            "pneumonia_prevalence": float(np.mean(arr["gt"][:, PNEUMONIA_IDX])),
            "report": rep.to_dict(), "selectivity": sel,
            "selectivity_feature_subset": sub, "meta": meta,
        }

    with open(out_dir / "phase4_features_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- cross-site headline: does any feature score beat chance under shift? --
    score_names = ["epistemic_std", "mutual_info", "energy14", "mahalanobis"]
    col_w = {"epistemic_std": 10, "mutual_info": 10, "energy14": 10, "mahalanobis": 12}
    print("\n" + "=" * 92)
    print("PHASE 4 STEP 6 HEADLINE — confident-error AUROC by score (top-10% confidence)")
    print("=" * 92)
    head = f"{'site':22s} {'n':>6s} {'prev':>6s} {'cw':>5s}"
    for n in score_names:
        head += f" {n:>{col_w[n]}s}"
    print(head)
    for key, d in summary.items():
        sel = d["selectivity"]
        row10 = next((s for s in sel if s["conf_top_pct"] == 10), None)
        r = d["report"]
        line = f"{key:22s} {d['n_images']:6d} {d['pneumonia_prevalence']:6.3f} {r['n_confident_wrong']:5d}"
        for n in score_names:
            v = row10.get(f"auroc_{n}") if row10 else None
            line += f" {fmt(v, col_w[n])}"
        print(line)
    print("-" * 92)
    print("chance = 0.5000.  Under shift (kermany/covid) disagreement was at chance "
          "(Phase 4). Does energy14 or mahalanobis exceed 0.5 where disagreement cannot?")
    print("=" * 92)
    print(f"[done] wrote {out_dir}/phase4_features_summary.json + arrays_<site>.npz")


if __name__ == "__main__":
    main()