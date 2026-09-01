"""Export Baur UNSURE Task 4 accuracy-coverage curves for every experiment.

``eval_baselines.py`` computes the Task-4 scalar (AUAC) for each ensemble but
only persists the number -- the curve it integrates, which *is* Baur's Task-4
figure, is thrown away. This script regenerates the curve itself for all nine
ensembles and writes them to one JSON for plotting/reporting.

It deliberately re-derives only the Task-4 slice of ``run_dataset()``: load the
arrays, drop -1 rows, per-class Youden thresholds on cal, build records on eval,
then the seven scores (epistemic_std, mutual_info, entropy, -confidence,
mahalanobis, knn, energy) plus each member's own confidence. The hybrid scores
are skipped on purpose -- they need the cal-split weight search and add nothing
to a curve figure.

Faithfulness is *checked*, not assumed: every recomputed AUAC is compared against
the value already published in that run's ``baseline_comparison.json`` and the
script exits non-zero on any mismatch above ``--tol``.

    python scripts/export_task4_curves.py
    python scripts/export_task4_curves.py --out runs/baselines/task4_curves.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_baselines import (KNN_K, _entropy, _load_arrays, _member_arrays,  # noqa: E402
                            _to_long)
from cxr_uncertainty.evaluate import _e_aurc, _risk_coverage_curve, _trapz  # noqa: E402
from cxr_uncertainty.reanalyze import (_knn_distances, build_records,  # noqa: E402
                                       calibrate_thresholds)

# name -> (cal npz, eval npz, provenance for the report).
ARR = Path("runs/eval_arrays")
P4 = Path("runs/phase4_features_4mem")
EXPERIMENTS = [
    ("openi", P4 / "arrays_openiC.npz", P4 / "arrays_openiD.npz",
     dict(dataset="OpenI", regime="pretrained", diversity="arch-diverse",
          label="OpenI 4-member", phase="4.4")),
    ("rex", ARR / "rex/arrays_rexcal.npz", ARR / "rex/arrays_rexeval.npz",
     dict(dataset="ReXGradient-160K", regime="pretrained", diversity="arch-diverse",
          label="ReX 4-member", phase="4.5")),
    ("rex_fs_resnet18", ARR / "rex_fs_resnet18/arrays_rex_fs_resnet18cal.npz",
     ARR / "rex_fs_resnet18/arrays_rex_fs_resnet18eval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="same-arch",
          label="ResNet-18 D-Ens", phase="4.6")),
    ("rex_fs_vit_tiny", ARR / "rex_fs_vit_tiny/arrays_rex_fs_vit_tinycal.npz",
     ARR / "rex_fs_vit_tiny/arrays_rex_fs_vit_tinyeval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="same-arch",
          label="ViT-Tiny D-Ens", phase="4.6")),
    ("rex_fs_convnext_tiny", ARR / "rex_fs_convnext_tiny/arrays_rex_fs_convnext_tinycal.npz",
     ARR / "rex_fs_convnext_tiny/arrays_rex_fs_convnext_tinyeval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="same-arch",
          label="ConvNeXt-Tiny D-Ens", phase="4.6")),
    ("rex_fs_resnet18_m3", ARR / "rex_fs_resnet18_m3/arrays_rex_fs_resnet18_m3cal.npz",
     ARR / "rex_fs_resnet18_m3/arrays_rex_fs_resnet18_m3eval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="same-arch",
          label="ResNet-18 M=3", phase="4.6.1")),
    ("rex_fs_resnet18_m2", ARR / "rex_fs_resnet18_m2/arrays_rex_fs_resnet18_m2cal.npz",
     ARR / "rex_fs_resnet18_m2/arrays_rex_fs_resnet18_m2eval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="same-arch",
          label="ResNet-18 M=2", phase="4.6.1")),
    ("rex_fs_crossarch", ARR / "rex_fs_crossarch/arrays_rex_fs_crossarchcal.npz",
     ARR / "rex_fs_crossarch/arrays_rex_fs_crossarcheval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="arch-diverse",
          label="Cross-arch M=3", phase="4.6.1")),
    ("rex_fs_crossarch_rv", ARR / "rex_fs_crossarch_rv/arrays_rex_fs_crossarch_rvcal.npz",
     ARR / "rex_fs_crossarch_rv/arrays_rex_fs_crossarch_rveval.npz",
     dict(dataset="ReXGradient-160K", regime="from-scratch", diversity="arch-diverse (CNN-TX)",
          label="Cross-arch M=2", phase="4.6.1")),
]

CORE_SCORES = ["confidence", "entropy", "epistemic_std", "mutual_info",
               "knn", "mahalanobis", "energy"]


def _task4_scores(C: dict, D: dict, member_keys: list, conf_pct: float,
                  unc_pct: float):
    """The Task-4 slice of ``eval_baselines.run_dataset``: (scores, wrong, meta).

    Mirrors that function's lines up to the E-AURC block -- same -1 filtering,
    same Youden thresholds, same score construction, HIGH = abstain first."""
    df_cal = _to_long(C, member_keys)
    df_eval = _to_long(D, member_keys)

    knn_img = _knn_distances(np.asarray(D["rad_feats"], dtype=np.float64),
                             np.asarray(C["rad_feats"], dtype=np.float64), k=KNN_K)
    id_to_knn = dict(zip([str(i) for i in D["ids"]], knn_img))
    df_eval["knn_dist"] = [id_to_knn.get(i, 0.0) for i in df_eval["image_id"]]

    n_full = len(df_eval)
    if bool((df_eval["gt"] < 0).any() or (df_cal["gt"] < 0).any()):
        df_cal = df_cal[df_cal["gt"] >= 0].reset_index(drop=True)
        df_eval = df_eval[df_eval["gt"] >= 0].reset_index(drop=True)

    thr = calibrate_thresholds(df_cal)
    rec, _ = build_records(df_eval, thr, conf_pct, unc_pct)
    a = rec.arrays()
    conf, wrong = a["confidence"], a["wrong"]

    id_energy = dict(zip([str(i) for i in D["ids"]], D["energy"]))
    energy = np.array([id_energy.get(i, np.nan) for i in rec.image_id], dtype=float)

    scores = {
        "confidence": -conf,                     # oriented HIGH = uncertain
        "entropy": _entropy(a["p_bar"]),
        "epistemic_std": a["epistemic_std"],
        "mutual_info": a["mutual_info"],
        "knn": a["knn_dist"],
        "mahalanobis": a["mahalanobis"],
        "energy": energy,
    }
    for k in member_keys:
        thr_m = calibrate_thresholds(
            df_cal[["pathology", "gt"]].assign(p_bar=df_cal[f"prob_{k}"]))
        c_m, _ = _member_arrays(df_eval, thr_m, f"prob_{k}")
        scores[f"conf_{k}"] = -c_m

    meta = dict(members=list(member_keys), n_members=len(member_keys),
                n_cal_images=int(C["gt"].shape[0]), n_eval_images=int(D["gt"].shape[0]),
                n_records=int(len(wrong)), n_records_full=int(n_full),
                n_uncertain_rows=int(n_full - len(wrong)),
                error_rate=round(float(wrong.mean()), 4))
    return scores, wrong, meta


def _downsample(cov: np.ndarray, acc: np.ndarray, n: int) -> list:
    """Keep ~n points, denser at low coverage where the curve bends. Always
    keeps both endpoints so the trapezoid integral is preserved."""
    if len(cov) <= n:
        idx = np.arange(len(cov))
    else:
        # geometric spacing in index space, from the low-coverage (index 0) end.
        raw = np.geomspace(1, len(cov), n) - 1
        idx = np.unique(np.clip(raw.round().astype(int), 0, len(cov) - 1))
        idx = np.unique(np.concatenate(([0], idx, [len(cov) - 1])))
    return [[round(float(cov[i]), 6), round(float(acc[i]), 6)] for i in idx]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baselines", default="runs/baselines",
                   help="dir holding <name>/baseline_comparison.json (the AUAC to verify against)")
    p.add_argument("--out", default="runs/baselines/task4_curves.json")
    p.add_argument("--points", type=int, default=200, help="points kept per curve")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--tol", type=float, default=1e-4,
                   help="max |recomputed - published| AUAC before failing")
    p.add_argument("--ds-tol", type=float, default=1e-3,
                   help="max |downsampled - full| AUAC before failing")
    args = p.parse_args()

    out = {"metric": "AUAC", "task": "Baur UNSURE@MICCAI 2025 Task 4",
           "definition": "trapz(1 - risk, coverage); abstain on highest-uncertainty record first",
           "conf_pct": args.conf_pct, "unc_pct": args.unc_pct,
           "points_per_curve": args.points, "experiments": {}}
    failures, checked = [], 0

    for name, cal_path, eval_path, prov in EXPERIMENTS:
        if not cal_path.exists() or not eval_path.exists():
            print(f"[{name}] SKIP -- missing arrays ({cal_path}, {eval_path})")
            failures.append(f"{name}: arrays missing")
            continue
        C, D = _load_arrays(cal_path), _load_arrays(eval_path)
        member_keys = [str(k) for k in (C["member_keys"] if C["member_keys"] is not None
                                        else D["member_keys"])]
        scores, wrong, meta = _task4_scores(C, D, member_keys, args.conf_pct, args.unc_pct)
        meta.update(prov)

        pub_path = Path(args.baselines) / name / "baseline_comparison.json"
        pub = {}
        if pub_path.exists():
            pub = json.load(open(pub_path)).get("aurc_e_aurc_full_eval", {})

        curves, base_err = {}, float(wrong.mean())
        for sname, s in scores.items():
            m = np.isfinite(s)
            curve = _risk_coverage_curve(s[m], wrong[m])
            if curve is None:
                continue
            cov, risk = np.asarray(curve[0]), np.asarray(curve[1])
            acc = 1.0 - risk
            auac_full = float(_trapz(acc, cov))
            aurc = float(_trapz(risk, cov))
            pts = _downsample(cov, acc, args.points)
            pa, pc = np.array([q[1] for q in pts]), np.array([q[0] for q in pts])
            auac_ds = float(np.trapezoid(pa, pc)) if hasattr(np, "trapezoid") \
                else float(np.trapz(pa, pc))

            row = {"auac": round(auac_full, 4), "auac_full": auac_full,
                   "auac_downsampled": round(auac_ds, 4),
                   "aurc": round(aurc, 4),
                   "e_aurc": round(float(_e_aurc(aurc, float(wrong[m].mean()))), 4),
                   "n": int(m.sum()), "points": pts}

            ref = pub.get(sname, {}).get("auac")
            if ref is not None:
                checked += 1
                d = abs(auac_full - float(ref))
                row["published_auac"] = ref
                row["auac_delta_vs_published"] = round(d, 6)
                if d > args.tol:
                    failures.append(f"{name}/{sname}: recomputed {auac_full:.4f} "
                                    f"vs published {ref} (d={d:.2e})")
            if abs(auac_ds - auac_full) > args.ds_tol:
                failures.append(f"{name}/{sname}: downsampled AUAC {auac_ds:.4f} "
                                f"vs full {auac_full:.4f}")
            curves[sname] = row

        best_mem = max(((k, v["auac"]) for k, v in curves.items()
                        if k.startswith("conf_")), key=lambda t: t[1], default=(None, None))
        meta["base_error_rate"] = round(base_err, 4)
        meta["best_member_score"], meta["best_member_auac"] = best_mem
        out["experiments"][name] = {"meta": meta, "scores": curves}

        top = sorted(((k, v["auac"]) for k, v in curves.items() if k in CORE_SCORES),
                     key=lambda t: -t[1])
        print(f"[{name:22s}] M={meta['n_members']} n={meta['n_records']:6d} "
              f"err={base_err:.4f} | " + "  ".join(f"{k}={v:.4f}" for k, v in top[:3]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[out] {args.out}  ({Path(args.out).stat().st_size/1e6:.2f} MB)")

    if failures:
        print(f"\n[FAIL] {len(failures)} check(s) failed:")
        for m in failures:
            print("  -", m)
        return 1
    print(f"[PASS] {checked} recomputed AUAC values match the published "
          f"baseline_comparison.json within {args.tol:g}; all downsampled curves "
          f"within {args.ds_tol:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
