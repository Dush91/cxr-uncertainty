"""Conformal triage end-to-end: Mondrian per-label LAC coverage (Part 3 step 7).

Runs the registered conformal variant (``cxr_uncertainty/conformal.py``) on the
leak-free eval sites **from the saved Phase-4-step-6 arrays** — no inference
needed (runs in seconds). This is the demonstration that the distribution-free
coverage layer (plan §L) is wired through the standard ``Records`` / ``EvalReport``
machinery and that ``conformal_coverage`` / ``conformal_size`` /
``per_pathology_conformal`` / ``brier`` are first-class metrics.

Design (plan §L / FINDINGS.md):
  * **Calibration = TS-only** on the pooled mean (``build_pipeline(use_ts=True)``);
    the LAC score + threshold sit on this calibrated ``p_bar``.
  * **Coverage = Mondrian per-label LAC** (``fit_mondrian_lac``): one ``τ_p`` per
    pathology (the ``⌈(1−α)(n_p+1)⌉``-th order statistic of the LAC scores on the
    cal split), fit on OpenI role-C. Triage: ``auto = set_size==1``, ``refer =
    set_size!=1``. The conformal-refer flag is the ``risk_flag`` (via
    ``build_records(risk_flag_override=lac_refer)``) so it is apples-to-apples with
    the epistemic_std / Mahalanobis flags on the same confident population.

Sites (cal = OpenI role-C in-distribution; eval =):
  * openiD  — in-distribution, all valid classes  +  Pneumonia-only.
  * kermany — OOD under shift, Pneumonia-only.
  * covid   — OOD under shift, Pneumonia-only.

Acceptance: OpenI-D per-pathology coverage ≥ 1−α (the distribution-free guarantee);
Kermany/COVID coverage < 1−α (honest break under shift — paralleling Phase 4,
in-distribution calibration does not transfer). The conformal-refer flag recall
is reported alongside the epistemic_std + Mahalanobis flags.

Usage:
    PYTHONPATH=. python scripts/eval_conformal.py \
        --arrays-dir runs/phase4_features --out runs/conformal --alpha 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.conformal import (build_conformal_pipeline, coverage_by_label,
                                       save_conformal_sidecar)
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds

PNEUMONIA = "Pneumonia"
PNEUMONIA_IDX = NIH_PATHOLOGIES.index(PNEUMONIA)


def _load(arrays_dir: Path, site: str):
    z = np.load(arrays_dir / f"arrays_{site}.npz", allow_pickle=True)
    return {
        "probs": z["probs"], "gt": z["gt"], "valid": z["valid"],
        "mahalanobis_saved": z["mahalanobis"], "ids": [str(s) for s in z["ids"]],
    }


def _attach_mahalanobis(df, ids, maha_saved):
    """Broadcast the image-level saved Mahalanobis score onto every
    (image, pathology) row so build_records records it (for the apples-to-apples
    flag comparison with the conformal-refer flag)."""
    id_score = {str(i): float(s) for i, s in zip(ids, maha_saved)}
    df = df.copy()
    df["mahalanobis"] = df["image_id"].astype(str).map(id_score).fillna(0.0)
    return df


def flag_metrics(rec, unc_pct):
    """Recall (over confident-wrong) and false-flag rate (over confident-correct)
    for the epistemic_std, Mahalanobis and conformal-refer flags — all on the same
    confident (top conf_pct%) population, so the three flags are comparable."""
    a = rec.arrays()
    is_conf = a["is_confident"]
    wrong = a["wrong"]
    conf_wrong = is_conf & (wrong == 1)
    conf_correct = is_conf & (wrong == 0)
    ep = a["epistemic_std"]
    maha = a.get("mahalanobis")
    lac_ref = a["lac_refer"]

    def rf(score):
        if is_conf.sum() == 0:
            return None, None
        cut = float(np.percentile(score[is_conf], 100.0 - unc_pct))
        flag = is_conf & (score >= cut)
        rec_ = float(flag[conf_wrong].mean()) if conf_wrong.sum() else None
        ffr_ = float(flag[conf_correct].mean()) if conf_correct.sum() else None
        return rec_, ffr_

    out = {"epistemic_std": rf(ep)}
    if maha is not None and np.isfinite(maha).any() and not np.all(maha == 0.0):
        out["mahalanobis"] = rf(maha)
    else:
        out["mahalanobis"] = (None, None)
    # conformal-refer flag: recall/FFR over the confident set.
    rec_c = float(lac_ref[conf_wrong].mean()) if conf_wrong.sum() else None
    ffr_c = float(lac_ref[conf_correct].mean()) if conf_correct.sum() else None
    out["conformal_refer"] = (rec_c, ffr_c)
    return out


def eval_site(C, site, pathologies, member_keys, conf_pct, unc_pct, alpha,
              pneumonia_only=True):
    """Run the conformal variant on one eval site (cal = OpenI-C). Returns
    (report, selectivity, flags, meta) or None."""
    df_cal, df_eval, meta = build_conformal_pipeline(
        C["probs"], C["gt"], C["valid"], C["ids"],
        site["probs"], site["gt"], site["valid"], site["ids"],
        pathologies, member_keys, alpha=alpha)
    # attach the saved Mahalanobis score for the apples-to-apples flag comparison.
    df_eval = _attach_mahalanobis(df_eval, site["ids"], site["mahalanobis_saved"])

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
    rec, build_meta = build_records(
        df_eval, thr, conf_pct, unc_pct,
        risk_flag_override=df_eval["lac_refer"].values)
    rep = evaluate_records(rec, RiskConfig(device="cpu"),
                           n_imgs=len(set(rec.image_id)))
    sel = auroc_at_confidence_levels(rec)
    flags = flag_metrics(rec, unc_pct)
    meta.update({"thresholds": {k: round(v, 4) for k, v in thr.items()},
                 **build_meta})
    return rep, sel, flags, meta


def print_site(title, rep, sel, flags, alpha):
    print(f"\n{'='*84}\n{title}\n{'='*84}")
    print(f"records={rep.n_records}  confident={rep.n_confident}  "
          f"confident_wrong={rep.n_confident_wrong}")
    print(f"conformal coverage (target 1-alpha={1-alpha:.2f}) : {rep.conformal_coverage}")
    print(f"conformal avg set size                    : {rep.conformal_size}")
    print(f"conformal Brier (p_bar vs gt)             : {rep.brier}")
    print(f"confident-error AUROC (epistemic_std)     : {rep.confident_error_auroc_std}")
    print(f"confident-error AUROC (mahalanobis)       : {rep.confident_error_auroc_mahalanobis}")
    print(f"flag recall / FFR (epistemic_std)          : {flags['epistemic_std']}")
    print(f"flag recall / FFR (mahalanobis)            : {flags['mahalanobis']}")
    print(f"flag recall / FFR (conformal_refer)       : {flags['conformal_refer']}")
    print("-" * 84)
    if rep.per_pathology_conformal:
        print(f"{'pathology':22s} {'n':>5s} {'coverage':>9s} {'avg_size':>9s} "
              f"{'refer_rate':>10s}  (target >= {1-alpha:.2f})")
        for p, d in rep.per_pathology_conformal.items():
            mark = "OK" if d["coverage"] >= (1 - alpha) - 0.02 else "BREAK"
            print(f"  {p:22s} {d['n']:5d} {d['coverage']:9.4f} "
                  f"{d['avg_set_size']:9.4f} {d['refer_rate']:10.4f}  {mark}")
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
    p = argparse.ArgumentParser(prog="eval_conformal", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays-dir", default="runs/phase4_features")
    p.add_argument("--out", default="runs/conformal")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2,raddino",
                   help="member keys (must match the order the saved probs were stacked in)")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.1,
                   help="conformal miscoverage level (target coverage = 1-alpha)")
    p.add_argument("--write-sidecar", action="store_true",
                   help="also write runs/conformal/conformal_sidecar.json carrying the "
                        "OpenI-C fit (T + per-pathology tau_p) so the live per-image "
                        "`cli --risk-policy conformal_triage --conformal-sidecar` "
                        "mirror (plan §M.2) can apply the pre-fit thresholds to the "
                        "live p_bar. Reuses the exact OpenI-C fit.")
    args = p.parse_args(argv)

    arrays_dir = Path(args.arrays_dir)
    member_keys = args.arch_ensemble.split(",")
    pathologies = list(NIH_PATHOLOGIES)
    alpha = args.alpha

    C = _load(arrays_dir, "openiC")
    assert C["probs"].shape[1] == len(member_keys), (
        f"saved probs have {C['probs'].shape[1]} members; --arch-ensemble has "
        f"{len(member_keys)} ({member_keys})")
    print(f"[data] cal=OpenI-C n={len(C['ids'])}  "
          f"pneumonia pos={int(C['gt'][:, PNEUMONIA_IDX].sum())}")
    print(f"[model] ensemble={member_keys}  pathologies={len(pathologies)} "
          f"conformal=LAC(Mondrian per-label, alpha={alpha})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("openiD_all", "openiD (in-distribution, all valid classes)", "openiD", False),
        ("openiD_pneu", "openiD (in-distribution, Pneumonia only)", "openiD", True),
        ("kermany", "kermany (OOD under shift, Pneumonia only)", "kermany", True),
        ("covid", "covid (OOD under shift, Pneumonia only)", "covid", True),
    ]
    summary = {}
    openiD_all_meta = None  # captured for the sidecar (full per-pathology OpenI-C fit)
    for key, title, site_name, pn_only in targets:
        site = _load(arrays_dir, site_name)
        res = eval_site(C, site, pathologies, member_keys, args.conf_pct,
                        args.unc_pct, alpha, pneumonia_only=pn_only)
        if res is None:
            print(f"[skip] {title}: no eval rows")
            continue
        rep, sel, flags, meta = res
        print_site(title, rep, sel, flags, alpha)
        summary[key] = {
            "title": title, "n_images": int(len(site["ids"])),
            "pneumonia_prevalence": float(np.mean(site["gt"][:, PNEUMONIA_IDX])),
            "report": rep.to_dict(), "selectivity": sel, "flags": flags, "meta": meta,
        }
        with open(out_dir / f"evaluation_{key}.json", "w") as f:
            json.dump(summary[key], f, indent=2)
        if key == "openiD_all":
            openiD_all_meta = meta

    # Conformal sidecar (plan §M.2): write the OpenI-C fit (T + per-pathology
    # tau_p) so the live per-image --risk-policy conformal_triage mirror can
    # apply the pre-fit thresholds to the live p_bar. The openiD_all run fits on
    # the full OpenI-C (all valid pathologies), so its meta carries every tau_p.
    if args.write_sidecar and openiD_all_meta is not None:
        sidecar_path = save_conformal_sidecar(
            out_dir / "conformal_sidecar.json", openiD_all_meta, pathologies,
            alpha, member_keys=member_keys)
        print(f"[write-sidecar] wrote {sidecar_path} "
              f"(T={openiD_all_meta['temperature_T']}, "
              f"{len(openiD_all_meta['tau_p'])} tau_p thresholds) "
              f"for the live --risk-policy conformal_triage path")
    elif args.write_sidecar:
        print("[write-sidecar] openiD_all site was skipped -> no sidecar written")

    with open(out_dir / "conformal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- headline: coverage vs 1-alpha (the guarantee in-distribution, break OOD) -
    print("\n" + "=" * 92)
    print(f"CONFORMAL HEADLINE — Mondrian per-label LAC coverage (target 1-alpha={1-alpha:.2f})")
    print("=" * 92)
    print(f"{'site':16s} {'n':>6s} {'prev':>6s} {'cw':>5s} "
          f"{'coverage':>9s} {'avg_size':>9s} {'refer_rate':>10s} {'brier':>7s}")
    for key, d in summary.items():
        r = d["report"]
        cov = r.get("conformal_coverage")
        sz = r.get("conformal_size")
        # overall refer rate from per-pathology conformal (weighted); fall back to None.
        ppc = r.get("per_pathology_conformal") or {}
        if ppc:
            tot = sum(v["n"] for v in ppc.values())
            ref = sum(v["refer_rate"] * v["n"] for v in ppc.values()) / tot if tot else None
        else:
            ref = None
        print(f"{key:16s} {d['n_images']:6d} {d['pneumonia_prevalence']:6.3f} "
              f"{r['n_confident_wrong']:5d} "
              f"{(f'{cov:.4f}' if cov is not None else '--'):>9s} "
              f"{(f'{sz:.4f}' if sz is not None else '--'):>9s} "
              f"{(f'{ref:.4f}' if ref is not None else '--'):>10s} "
              f"{(f'{r['brier']:.4f}' if r.get('brier') is not None else '--'):>7s}")
    print("-" * 92)
    print(f"In-distribution (openiD) per-pathology coverage should be >= {1-alpha:.2f} (the")
    print("distribution-free guarantee). Under shift (kermany/covid) coverage is expected to")
    print("BREAK (< target) — in-distribution calibration does not transfer (Phase 4). The")
    print("conformal layer complements the Mahalanobis under-shift flag, not replaces it.")
    print("=" * 92)
    print(f"[done] wrote {out_dir}/conformal_summary.json + evaluation_<site>.json")


if __name__ == "__main__":
    main()