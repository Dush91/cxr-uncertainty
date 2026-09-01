"""Interactive demo: the Phase-6 DEPLOYMENT system on ReXGradient.

Agent 1 (UQ + confident-error flagging) and Agent 2 (similar-case retrieval),
both running on the pretrained 4-member architecture-diverse ensemble
(xrv_nih + convnextv2 + raddino + arkswin). The from-scratch D-Ens models are
NOT here -- they are the UQ benchmark control, not a deployment model
(docs/uncertainty_evaluation.md section 4.6; docs/agent2_retrieval.md v2.0).

Agent 1 implements the two-regions deployment rule
(docs/uncertainty_evaluation.md:808-811):

  * Region A (bulk selective abstention) -> calibrated confidence. Full-set
    AUAC 0.955; predictions below the cal split's top-10% confidence cutoff
    are flagged ABSTAIN.
  * Region B (confident-wrong triage inside the confident stratum) ->
    ensemble disagreement (epistemic_std >= the within-confident-set 50th-pct
    cut). Honest caveat baked into the UI: on this PRETRAINED ensemble
    disagreement is statistically equivalent to confidence at power
    (DeLong p=0.99) -- second-order triage only.
  * Mahalanobis OOD (image level) -> shift watchdog, refit on ReX-cal
    RAD-DINO features; explicitly NOT an error ranker on ReX (AUROC 0.53).

Agent 2 queries the production HNSW index (runs/retrieval/rex_raddino,
8,064 cal-split RAD-DINO embeddings, recall@10 = 1.0000 vs exact) with the
query's own live RAD-DINO embedding; predictions/correctness come from
sidecar.parquet joined at query time. Ground truth never enters the index.

Calibration: TS-only temperature T read from the published ReX baseline
(runs/baselines/rex/baseline_comparison.json); per-pathology Youden thresholds
fit at startup on the leak-free cal split (raw pooled-mean scale, matching the
published eval_baselines records) and cached to runs/app_rex/calibration.json.

Run (repo root):  pip install "gradio>=4.0";  python scripts/demo_app_rex.py
Companion:        scripts/demo_app.py (OpenI benchmark demo, untouched)
                  scripts/query_retrieval.py (Agent-2 CLI)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import torch

import gradio as gr

from cxr_uncertainty.config import RiskConfig, NIH_PATHOLOGIES
from cxr_uncertainty.models import CXREnsemble, Alignment
from cxr_uncertainty.utils import load_image_tensor
from cxr_uncertainty.feature_uq import extract_member, MahalanobisOOD
from cxr_uncertainty.reanalyze import _confidence, _youden_threshold
import cxr_uncertainty.agent3 as agent3
import cxr_uncertainty.agent4 as agent4

MEMBERS = ["xrv_nih", "convnextv2", "raddino", "arkswin"]


def _regime() -> str:
    """--regime {unadapted,adapted}: paths-only swap (Stage 2). Default is the
    ADAPTED arm (Phase 7): a hospital deploys LP-FT-adapted members, so that is
    the production-realistic regime; unadapted remains available explicitly as
    the zero-adaptation comparison lens."""
    for i, a in enumerate(sys.argv):
        if a == "--regime" and i + 1 < len(sys.argv):
            v = sys.argv[i + 1].lower()
            assert v in ("unadapted", "adapted"), f"bad --regime {v!r}"
            return v
    return "adapted"


REGIME = _regime()

if REGIME == "adapted":         # Phase 7 adapted arm (seed s0 headline)
    ARR_DIR = os.path.join(REPO, "runs", "rex_phase7", "eval_arrays", "rex_adapted_s0")
    ARR_CAL = os.path.join(ARR_DIR, "arrays_rex_adaptedcal.npz")
    ARR_EVAL = os.path.join(ARR_DIR, "arrays_rex_adaptedeval.npz")
    BASELINE_JSON = os.path.join(REPO, "runs", "rex_phase7", "baselines",
                                 "rex_adapted_s0", "rex_adapted_s0", "baseline_comparison.json")
    RETRIEVAL_DIR = os.path.join(REPO, "runs", "rex_phase7", "retrieval",
                                 "rex_adapted_raddino_full")
    MANIFEST = os.path.join(REPO, "data", "rex_manifest_fs.parquet")
    CACHE_DIR = os.path.join(REPO, "runs", "app_rex_adapted")
    APP_PORT = 7862
else:
    ARR_DIR = os.path.join(REPO, "runs", "eval_arrays", "rex")
    ARR_CAL = os.path.join(ARR_DIR, "arrays_rexcal.npz")
    ARR_EVAL = os.path.join(ARR_DIR, "arrays_rexeval.npz")
    BASELINE_JSON = os.path.join(REPO, "runs", "baselines", "rex", "baseline_comparison.json")
    RETRIEVAL_DIR = os.path.join(REPO, "runs", "retrieval", "rex_raddino")
    MANIFEST = os.path.join(REPO, "data", "rex_manifest.parquet")
    CACHE_DIR = os.path.join(REPO, "runs", "app_rex")
    APP_PORT = 7861
CACHE_JSON = os.path.join(CACHE_DIR, "calibration.json")

CONF_PCT = 10.0        # confident set = top-10% by Youden-anchored confidence
UNC_PCT = 50.0         # disagreement flag = top-50% epistemic_std within the confident set
MAHA_HIGH_PCT = 95.0   # Mahalanobis OOD banner percentile (vs eval-split reference)
K_NEIGHBORS = 8        # Agent-2 retrieval depth
MAHA_CLASS = "Pneumonia"   # class-conditional density class (matches build_meta.json)

# Agent 3 -- 8th output. The explanation is rendered on demand by
# agent3.render_cached() over the evidence pack cached in predict().
PLACEHOLDER_MD = (
    "*Agent 3 explanation appears here after a prediction -- run **Explain this "
    "prediction** below, or click an example. Non-diagnostic: explains model "
    "behavior from measured evidence (quality metrics, member disagreement, "
    "label ambiguity, dataset bias, retrieved neighbors), never reads pixels.*")

# ---------------------------------------------------------------------------
# Singletons (built once at startup, reused per request)
# ---------------------------------------------------------------------------
_ENS: CXREnsemble | None = None
_CFG: RiskConfig | None = None
_CAL: dict = {}          # cal npz dict
_EVAL: dict = {}         # eval npz dict
_EVAL_ROWS: dict[str, int] = {}   # image_id -> row idx in the eval npz
_EVAL_IDS: set[str] = set()
_PATH_BY_ID: dict[str, str] = {}
_VIEW_BY_ID: dict[str, str] = {}   # image_id -> projection tag (AP/PA), Agent 3
_QUALITY_STATS: dict | None = None  # agent3_quality_stats.json (bands + bias)
_GT_BY_ID: dict[str, np.ndarray] = {}
_TEMP_T: float = 1.0
_YOUDEN: dict[str, float | None] = {}
_CONF_CUT: float = 0.0
_UNC_CUT: float = 0.0
_CAL_ROWS: dict[str, int] = {}    # image_id -> row idx in the cal npz
_TRAIN_MM = None                  # lazy fp16 224px train-cache memmap
_TRAIN_ROW_BY_ID: dict[str, int] = {}
_TRAIN_PRED_CACHE: dict[str, list[str]] = {}   # computed train predicted sets
_MAHA: MahalanobisOOD | None = None
_MAHAR_REF: np.ndarray = np.array([])
_RETR_INDEX = None
_RETR_FEATS: np.ndarray = np.zeros((0, 1), dtype=np.float32)
_RETR_IDS: np.ndarray = np.array([])
_RETR_META: pd.DataFrame | None = None
_RETR_DEC: np.ndarray = np.zeros((0, 0), dtype=bool)   # (N,14) model decision per library row
_EXAMPLES: list[dict] = []
_GT_LOOKUP: dict[str, dict] = {}   # user-uploaded labels (optional CSV/JSON


def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _ts(p):
    """Temperature-scale scalar or array; None/NaN pass through."""
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float64)
    if _TEMP_T == 1.0:
        return float(a) if a.ndim == 0 else a
    out = a.copy()
    m = np.isfinite(out)
    pc = np.clip(out[m], 1e-9, 1 - 1e-9)
    out[m] = 1.0 / (1.0 + np.exp(-np.log(pc / (1.0 - pc)) / _TEMP_T))
    return float(out) if a.ndim == 0 else out


def _banner(high: bool, lines: list[str]) -> str:
    if high:
        bg, bd, head = "#fdecea", "#d03b3b", "&#9888; HIGH RISK"
    else:
        bg, bd, head = "#eafaf1", "#1e8e5a", "&#10003; LOW RISK"
    body = "<br>".join(lines)
    return (f'<div style="background:{bg};border-left:6px solid {bd};'
            f'padding:10px 14px;border-radius:4px">'
            f'<div style="font-size:1.15em;font-weight:700;color:{bd}">{head}</div>'
            f'<div style="margin-top:4px;color:#222">{body}</div></div>')


def _fmt_gt(v) -> str:
    """Manifest label value -> compact string (-1 = labeler-uncertain)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    n = int(v)
    return "uncertain" if n == -1 else str(n)


# Label parser shared with the OpenI demo (same `basename -> {pathology: 0/1}`
# CSV/JSON contract as scripts/export_gt_labels.py).
try:
    from demo_app import parse_labels_table  # type: ignore
except Exception:  # run-from-repo-root fallback
    _S = os.path.join(REPO, "scripts")
    if _S not in sys.path:
        sys.path.insert(0, _S)
    from demo_app import parse_labels_table  # type: ignore


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _load_public_temperature() -> float:
    """Read the published ReX TS temperature (no re-fit)."""
    if _exists(BASELINE_JSON):
        try:
            with open(BASELINE_JSON) as f:
                base = json.load(f)
            return float(base.get("ts_calibrated_confidence", {}).get("temperature", 1.0))
        except Exception as e:
            print(f"[startup] temperature read failed: {e}")
    return 1.0


def _fit_calibration(cal: dict) -> dict:
    """Per-pathology Youden thresholds + two-regions cuts, fit on cal only.

    Raw pooled-mean scale (identical to the published eval_baselines records):
      youden[pat] cutoff; conf_cut = (100-CONF_PCT)pct of Youden-anchored
      confidence over valid certain-label cal records (Region A abstain line);
      unc_cut = (100-UNC_PCT)pct of epistemic_std WITHIN the cal confident set
      (Region B triage line, the eval_baselines convention).
    """
    probs = cal["probs"]                                   # (N,M,P)
    valid = cal["valid"].astype(bool)
    gt = np.asarray(cal["gt"], dtype=np.float64)
    mean_p = np.nanmean(probs, axis=1)                     # (N,P)
    pats = [str(p) for p in cal["pathologies"]]

    youden: dict[str, float | None] = {}
    for j, pat in enumerate(pats):
        v = valid[:, j] & (gt[:, j] >= 0) & np.isfinite(mean_p[:, j])
        y = gt[v, j].astype(int)
        youden[pat] = (None if (y == 1).sum() < 2 or (y == 0).sum() < 2
                       else _youden_threshold(y, mean_p[v, j]))

    thr = np.array([0.5 if youden[p] is None else youden[p] for p in pats])
    conf_mat = _confidence(mean_p, thr[None, :])           # (N,P) in [0,1]
    rec = valid & (gt >= 0) & np.isfinite(mean_p)
    conf_flat = conf_mat[rec]
    std_flat = np.stack([np.nanstd(probs[:, :, j], axis=1)
                         for j in range(len(pats))], axis=1)[rec]
    conf_cut = float(np.percentile(conf_flat, 100.0 - CONF_PCT))
    is_conf = conf_flat >= conf_cut
    unc_cut = (float(np.percentile(std_flat[is_conf], 100.0 - UNC_PCT))
               if is_conf.sum() else float(np.percentile(std_flat, 100.0 - UNC_PCT)))
    print(f"[fit] youden fit on cal; conf_cut={conf_cut:.4f} "
          f"({int(is_conf.sum())}/{len(conf_flat)} confident); unc_cut={unc_cut:.4f}")
    return {"youden": youden, "conf_cut": conf_cut, "unc_cut": unc_cut}


def _refit_mahalanobis() -> None:
    """Class-conditional RAD-DINO density refit on cal; percentile reference =
    the out-of-sample eval-split scores cached in the eval npz."""
    global _MAHA, _MAHAR_REF
    j = NIH_PATHOLOGIES.index(MAHA_CLASS)
    feats = np.asarray(_CAL["rad_feats"], dtype=np.float64)
    v = _CAL["valid"].astype(bool)[:, j] & (np.asarray(_CAL["gt"])[:, j] >= 0)
    _MAHA = MahalanobisOOD().fit(feats[v], np.asarray(_CAL["gt"])[v, j].astype(int))
    if "mahalanobis" in _EVAL:
        ref, src = np.asarray(_EVAL["mahalanobis"], dtype=np.float64), "eval-npz cached"
    else:
        ref = _MAHA.score(np.asarray(_EVAL["rad_feats"], dtype=np.float64))
        src = "refit on eval feats"
    _MAHAR_REF = np.sort(ref)
    print(f"[startup] mahalanobis class={_MAHA.classes_.tolist()} "
          f"n_per_class={_MAHA.n_per_class_} ref={src}")


def _load_retrieval() -> None:
    global _RETR_INDEX, _RETR_FEATS, _RETR_IDS, _RETR_META, _PATH_BY_ID, _GT_BY_ID
    global _RETR_GT, _RETR_GTV, _RETR_DEC, _VIEW_BY_ID
    import hnswlib
    store = np.load(os.path.join(RETRIEVAL_DIR, "store.npz"), allow_pickle=True)
    _RETR_FEATS = np.asarray(store["feats"], dtype=np.float32)
    _RETR_IDS = np.array([str(i) for i in store["ids"]])
    _RETR_META = pd.read_parquet(os.path.join(RETRIEVAL_DIR, "sidecar.parquet")) \
                     .set_index("image_id").reindex(_RETR_IDS)
    # report-derived label vectors (scripts/augment_sidecar_labels.py) for the
    # label-agreement rerank; absent on pre-augmentation sidecars -> rerank off
    if "gt_labels" in _RETR_META.columns and _RETR_META["gt_labels"].notna().all():
        _RETR_GT = np.stack(_RETR_META["gt_labels"].values).astype(np.int8)
        _RETR_GTV = np.stack(_RETR_META["gt_valid"].values).astype(bool)
    else:
        _RETR_GT = _RETR_GTV = None
    # per-library-row model decision vector (N,14): cal/eval rows from the npz
    # pooled mean vs their Youden threshold; train rows False (no stored
    # per-class probs -> excluded from the contrast pool)
    _RETR_DEC = np.zeros((len(_RETR_IDS), len(NIH_PATHOLOGIES)), dtype=bool)
    pos = {iid: k for k, iid in enumerate(_RETR_IDS)}
    thr_vec = np.array([0.5 if _YOUDEN.get(p) is None else float(_YOUDEN[p])
                        for p in NIH_PATHOLOGIES])
    for d, rows in ((_EVAL, _EVAL_ROWS), (_CAL, _CAL_ROWS)):
        if not d or not rows:
            continue
        pbar = np.nanmean(np.asarray(d["probs"], dtype=np.float64), axis=1)  # (N,14)
        dec_rows = np.asarray(d["valid"]).astype(bool) & (pbar >= thr_vec[None, :])
        for iid, k in rows.items():
            j = pos.get(iid)
            if j is not None:
                _RETR_DEC[j] = dec_rows[k]
    _RETR_INDEX = hnswlib.Index(space="cosine", dim=_RETR_FEATS.shape[1])
    _RETR_INDEX.load_index(os.path.join(RETRIEVAL_DIR, "index.bin"))
    _RETR_INDEX.set_ef(max(256, K_NEIGHBORS * 10))
    _PATH_BY_ID, _GT_BY_ID = {}, {}
    if _exists(MANIFEST):
        m = pd.read_parquet(MANIFEST)
        _PATH_BY_ID = dict(zip(m["image_id"].astype(str), m["image_path"].astype(str)))
        _GT_BY_ID = {str(i): np.asarray(l, dtype=float)
                     for i, l in zip(m["image_id"].astype(str), m["labels"])}
        if "view" in m.columns:
            _VIEW_BY_ID = {str(i): (None if pd.isna(v) else str(v))
                           for i, v in zip(m["image_id"].astype(str), m["view"])}
    print(f"[startup] retrieval n={len(_RETR_IDS)} dim={_RETR_FEATS.shape[1]} "
          f"rerank={'on' if _RETR_GT is not None else 'off'}")


def _build_examples() -> list[dict]:
    """Five eval-split examples by label predicate (cached paths only)."""
    if not (_exists(ARR_EVAL) and _exists(MANIFEST)):
        return []
    m = pd.read_parquet(MANIFEST)
    ev = m[m["split_role"] == "eval"].copy()
    ev["_lab"] = ev["labels"].apply(lambda x: np.asarray(list(x), dtype=float))
    ev["_ok"] = ev["image_path"].apply(_exists)
    ok = ev[ev["_ok"] & ev["image_id"].astype(str).isin(_EVAL_IDS)]

    out: list[dict] = []

    def pick(pred, kind):
        sub = ok[ok["_lab"].apply(pred)]
        if len(sub):
            r = sub.iloc[0]
            out.append({"image_id": str(r["image_id"]), "kind": kind,
                        "image_path": str(r["image_path"]),
                        "gt": {p: int(v) for p, v in zip(NIH_PATHOLOGIES, r["_lab"])}})

    pn = NIH_PATHOLOGIES.index("Pneumonia")
    eff = NIH_PATHOLOGIES.index("Effusion")
    car = NIH_PATHOLOGIES.index("Cardiomegaly")
    pick(lambda l: bool((l[l >= 0] == 0).all()) and bool((l >= 0).any()),
         "ReX normal (labeler-negative)")
    pick(lambda l: l[pn] == 1, "ReX Pneumonia (labeler-positive)")
    pick(lambda l: l[eff] == 1, "ReX Effusion (labeler-positive)")
    pick(lambda l: l[car] == 1, "ReX Cardiomegaly (labeler-positive)")
    pick(lambda l: bool((l == -1).any()), "ReX with labeler-UNCERTAIN labels (-1)")

    # Flag-firing examples: the clean picks above can all sit in the low-
    # disagreement half of the confident set, so the demo never shows Region B.
    # Select from the eval npz with the SAME math the app scores rows with
    # (Youden-anchored confidence >= conf_cut AND epistemic_std >= unc_cut):
    # one case where Region B fires at all, and -- if one exists -- a CONFIDENT
    # ERROR (disagreement fires on a wrong prediction), the defining case.
    try:
        probs = np.asarray(_EVAL["probs"], dtype=np.float64)         # (N,M,P)
        mean_p = np.nanmean(probs, axis=1)
        gt = np.asarray(_EVAL["gt"], dtype=np.float64)
        valid = np.asarray(_EVAL["valid"], dtype=bool)
        pats = [str(p) for p in _EVAL["pathologies"]]
        thr = np.array([0.5 if _YOUDEN is None or _YOUDEN.get(p) is None
                        else float(_YOUDEN[p]) for p in pats])
        cm = _confidence(mean_p, thr[None, :])
        std = np.stack([np.nanstd(probs[:, :, j], axis=1)
                        for j in range(len(pats))], axis=1)
        is_conf = cm >= _CONF_CUT
        fires = is_conf & valid & (std >= _UNC_CUT)
        wrong = fires & np.isfinite(gt) & (gt >= 0) & \
            ((mean_p >= thr[None, :]) != (gt > 0))                    # decision != gt

        def pick_flag(mask, kind):
            rows = np.where(mask.any(axis=1))[0]
            for k in rows:
                rid = str(_EVAL["ids"][k])
                r = ok[ok["image_id"].astype(str) == rid]
                if len(r) and _exists(r["image_path"].iloc[0]):
                    row = r.iloc[0]
                    out.append({"image_id": rid, "kind": kind,
                                "image_path": str(row["image_path"]),
                                "gt": {p: int(v) for p, v in
                                       zip(NIH_PATHOLOGIES, row["_lab"])}})
                    return

        pick_flag(fires, "ReX flagged: Region B disagreement (confident + high std)")
        pick_flag(wrong, "ReX FLAGGED confident-ERROR (disagreement on a wrong call)")
        print(f"[examples] Region B fires on {int(fires.sum())} eval rows; "
              f"confident-error rows {int(wrong.sum())}")
    except Exception as e:
        print(f"[examples] flag-firing selection skipped: {type(e).__name__}: {e}")
    return out


def load_state() -> None:
    """Build the ensemble + all calibration artifacts once (cache-aware)."""
    global _ENS, _CFG, _CAL, _EVAL, _EVAL_ROWS, _EVAL_IDS, _EXAMPLES
    global _YOUDEN, _CONF_CUT, _UNC_CUT, _TEMP_T, _CAL_ROWS
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device)
    ens = CXREnsemble(members=MEMBERS, cfg=cfg)
    # Adapted regime: the live-forward path must ride the SAME adapted weights
    # the arrays/retrieval index were built with (mirrors build_eval_arrays
    # --ckpt-overlay) -- otherwise new uploads get stock predictions/embeddings
    # scored against adapted calibration.
    if REGIME == "adapted":
        _CKPT = {mb.key: os.path.join(REPO, "checkpoints", "rex_adapted", mb.key + ".pt")
                 for mb in ens.members}
        _CKPT["convnextv2"] = os.path.join(REPO, "checkpoints", "rex_adapted",
                                           "convnextv2_s0.pt")  # s0 headline seed
        for k, path in _CKPT.items():
            assert os.path.isfile(path), f"adapted ckpt missing: {path}"
            mb = next(m for m in ens.members if m.key == k)
            mb.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[overlay] {k} <- {path}")
    cfg.img_size = max(getattr(mb, "native_size", 224) for mb in ens.members)
    _ENS, _CFG = ens, cfg

    _CAL = {k: v for k, v in np.load(ARR_CAL, allow_pickle=True).items()}
    _EVAL = {k: v for k, v in np.load(ARR_EVAL, allow_pickle=True).items()}
    _EVAL_IDS = set(str(i) for i in _EVAL["ids"])
    _EVAL_ROWS = {str(i): k for k, i in enumerate(_EVAL["ids"])}
    _CAL_ROWS = {str(i): k for k, i in enumerate(_CAL["ids"])}

    _TEMP_T = _load_public_temperature()
    if _exists(CACHE_JSON) and os.path.getmtime(CACHE_JSON) >= os.path.getmtime(ARR_CAL):
        with open(CACHE_JSON) as f:
            cc = json.load(f)
        _YOUDEN = {p: (None if v is None else float(v)) for p, v in cc["youden"].items()}
        _CONF_CUT = float(cc["conf_cut"])
        _UNC_CUT = float(cc["unc_cut"])
        print(f"[startup] calibration cache loaded ({CACHE_JSON})")
    else:
        c = _fit_calibration(_CAL)
        _YOUDEN = c["youden"]
        _CONF_CUT = c["conf_cut"]
        _UNC_CUT = c["unc_cut"]
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_JSON, "w") as f:
            json.dump({"youden": _YOUDEN, "conf_cut": _CONF_CUT, "unc_cut": _UNC_CUT,
                       "temperature": _TEMP_T, "conf_pct": CONF_PCT, "unc_pct": UNC_PCT},
                      f, indent=2)
        print(f"[startup] calibration fitted on cal + cached -> {CACHE_JSON}")

    _refit_mahalanobis()
    _load_retrieval()
    _EXAMPLES = _build_examples()
    # Agent 3: eval-split quality bands + precomputed bias block (optional;
    # absent stats only downgrade the evidence pack to band="unavailable")
    global _QUALITY_STATS
    _QUALITY_STATS = agent3.load_quality_stats(
        os.path.join(CACHE_DIR, "agent3_quality_stats.json"))
    print(f"[startup] device={device} img_size={cfg.img_size} "
          f"ensemble={ens.member_keys} ({time.time() - t0:.0f}s)")


def _maha_pct(maha_raw: float) -> float:
    """Percentile of a Mahalanobis score vs the OUT-OF-SAMPLE eval reference."""
    if len(_MAHAR_REF) == 0:
        return 0.0
    return float(np.searchsorted(_MAHAR_REF, maha_raw, side="right")) / len(_MAHAR_REF) * 100.0


def _resolve(image_path: str):
    """(image_id, gt_labels) for a path; in-manifest images resolve by basename
    stem (image_id == png stem in this repo). Uploaded-labels entries
    (basename or stem) overlay the manifest labels where present."""
    stem = Path(image_path).stem
    image_id = stem if stem in _EVAL_IDS else (stem if stem in _PATH_BY_ID else None)

    up = _GT_LOOKUP.get(os.path.basename(image_path)) \
        or _GT_LOOKUP.get(stem)
    if not up and not _GT_BY_ID:
        return image_id, None
    gt = _GT_BY_ID.get(stem)
    if up:
        base = gt if gt is not None else np.full(len(NIH_PATHOLOGIES), np.nan)
        gt = np.array(base, dtype=np.float64)
        for p, v in up.items():
            if p in NIH_PATHOLOGIES:
                gt[NIH_PATHOLOGIES.index(p)] = float(v)
    elif gt is None:
        return image_id, None
    return image_id, gt


def _forward(image_path: str, image_id: str | None):
    """(probs (M,14) f64, rad_feat (D,) f32, maha_raw). Cached eval row when
    possible, one live ensemble forward otherwise."""
    if image_id is not None and image_id in _EVAL_ROWS:
        k = _EVAL_ROWS[image_id]
        return (np.asarray(_EVAL["probs"][k], dtype=np.float64),
                np.asarray(_EVAL["rad_feats"][k], dtype=np.float32),
                float(_EVAL["mahalanobis"][k]))
    x = load_image_tensor(image_path, img_size=_CFG.img_size, device=_CFG.device)
    eo = _ENS.forward_ensemble_full(x)
    probs = eo.per_member_probs[:, 0, :].detach().cpu().numpy().astype(np.float64)
    feat = np.asarray(extract_member(eo, "raddino")[1], dtype=np.float32)
    maha_raw = float(_MAHA.score(feat.astype(np.float64).reshape(1, -1))[0])
    try:
        del eo, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return probs, feat, maha_raw


def _pred_set(image_id: str) -> list[str] | None:
    """The model's full predicted set for a library row: every class above its
    Youden threshold, sorted by pooled probability desc (dominant first).

    cal/eval rows come from the local npz (instant); train rows have no stored
    per-class probs, so they are computed lazily with the SAME 3-member 224
    forward that annotated the sidecar (xrv_nih + convnextv2 + raddino over the
    local fp16 train cache) and memoized. Returns None when computable nowhere."""
    pats = [str(p) for p in NIH_PATHOLOGIES]

    def _set(p_mean: np.ndarray, valid_row: np.ndarray) -> list[str]:
        thr = np.array([0.5 if _YOUDEN.get(p) is None else float(_YOUDEN[p])
                        for p in pats])
        reaches = np.isfinite(p_mean) & valid_row & (p_mean >= thr)
        if not reaches.any():
            return ["no-finding"]        # model's own no-finding verdict
        # ONLY reaching classes -- argsort pads non-reaching entries to -inf
        # and would otherwise silently fill the top-k with them
        order = sorted(np.where(reaches)[0], key=lambda j: -p_mean[j])
        return [pats[j] for j in order[:4]]

    k = _EVAL_ROWS.get(image_id)
    if k is not None:
        pm = np.nanmean(np.asarray(_EVAL["probs"], dtype=np.float64)[k], axis=0)
        return _set(pm, np.asarray(_EVAL["valid"])[k].astype(bool))
    k = _CAL_ROWS.get(image_id)
    if k is not None:
        pm = np.nanmean(np.asarray(_CAL["probs"], dtype=np.float64)[k], axis=0)
        return _set(pm, np.asarray(_CAL["valid"])[k].astype(bool))
    # train row: lazy 3-member forward over the cache (same annotators as the
    # sidecar; arkswin omitted -- no 768 PNGs for train)
    if image_id in _TRAIN_PRED_CACHE:
        return _TRAIN_PRED_CACHE[image_id]
    global _TRAIN_MM
    if _TRAIN_MM is None:
        cache = os.path.join(REPO, "data", "adapted_cache_train.npy")
        paths_txt = os.path.join(REPO, "data", "adapted_cache_train.paths.txt")
        if not (_exists(cache) and _exists(paths_txt)):
            return None
        with open(paths_txt) as f:
            for i, ln in enumerate(f):
                ln = ln.strip()
                if ln:
                    _TRAIN_ROW_BY_ID[os.path.splitext(os.path.basename(ln))[0]] = i
        _TRAIN_MM = np.memmap(cache, dtype=np.float16, mode="r",
                              shape=(len(_TRAIN_ROW_BY_ID), 1, 224, 224))
    k = _TRAIN_ROW_BY_ID.get(image_id)
    if k is None:
        return None
    import torch
    x = torch.from_numpy(np.asarray(_TRAIN_MM[k:k + 1], dtype=np.float32))
    ms = [m for m in _ENS.members if m.key in ("xrv_nih", "convnextv2", "raddino")]
    al = Alignment(ms, list(NIH_PATHOLOGIES))   # maps xrv 18-slot -> NIH-14
    with torch.no_grad():
        outputs = [m.forward_batch(x) for m in ms]
        probs, _, _, valid = al.stack(outputs)   # (M,1,14)
    pm = np.asarray(probs)[:, 0, :]
    out = _set(np.nanmean(pm, axis=0),
               np.asarray(valid)[:, 0, :].all(axis=0))
    _TRAIN_PRED_CACHE[image_id] = out
    return out


def _agent1_rows(probs: np.ndarray, gt: np.ndarray | None) -> pd.DataFrame:
    """Per-pathology Agent-1 rows (two-regions flags), sorted by TS p_bar desc."""
    pats = [str(p) for p in NIH_PATHOLOGIES]
    rows = []
    for j, pat in enumerate(pats):
        pv = probs[:, j]                        # (M,) member probabilities
        p_raw = float(np.nanmean(pv))
        if not np.isfinite(p_raw):
            continue
        std = float(np.nanstd(pv))
        t = _YOUDEN.get(pat)
        gt_s = _fmt_gt(gt[j]) if gt is not None else ""
        if t is None:
            rows.append({"pathology": pat, "p_bar_ts": None, "youden_thr": None,
                         "decision": -1, "calib_confidence": None,
                         "epistemic_std": round(std, 4),
                         "abstain_flag": False, "disagreement_flag": False,
                         "gt": gt_s, "error_type": "excluded (no labels in cal)"})
            continue
        conf = float(np.clip(_confidence(np.array([p_raw]), np.array([float(t)]))[0], 0, 1))
        dec = int(p_raw >= float(t))
        is_conf = conf >= _CONF_CUT
        if gt is None:
            err = "unknown (no labels)"
        elif int(gt[j]) == -1:
            err = "gt labeler-uncertain"
        elif dec == int(gt[j]):
            err = "confident_correct" if is_conf else "tentative_correct"
        else:
            err = "confident_wrong" if is_conf else "tentative_wrong"
        rows.append({
            "pathology": pat,
            "p_bar_ts": round(float(_ts(p_raw)), 4),
            "youden_thr": round(float(_ts(float(t))), 4),
            "decision": dec,
            "calib_confidence": round(conf, 4),
            "epistemic_std": round(std, 4),
            "abstain_flag": not is_conf,
            "disagreement_flag": bool(is_conf and std >= _UNC_CUT),
            "gt": gt_s,
            "error_type": err,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("p_bar_ts", ascending=False,
                          na_position="last").reset_index(drop=True)


def _member_bar(probs: np.ndarray, pred_idx: int):
    """Per-member probability bar for the dominant pathology (+ caption)."""
    vals = probs[:, pred_idx].astype(float)
    member_df = pd.DataFrame({"member": MEMBERS,
                              "prob": [round(float(v), 4) if np.isfinite(v) else 0.0
                                       for v in vals]})
    std = float(np.nanstd(vals))
    pooled = float(np.nanmean(vals))
    agree = ("members AGREE (low std) -- only the OOD/retrieval-review signals can "
             "catch a confident mistake here" if std < _UNC_CUT else
             "members DISAGREE (high std) -- Region-B triage territory")
    cap = (f"**Per-member probability for {NIH_PATHOLOGIES[pred_idx]}** "
           f"(pooled p_raw={pooled:.3f}, std={std:.3f}). {agree}.")
    return member_df, cap


def _retrieval(q: np.ndarray, pred_pathology: str, mode: str, query_id: str | None,
               p_q: np.ndarray | None = None,
               lam: float = 0.3, flag_class: str | None = None):
    """HNSW traversal-time filtered query on the FULL index (train+cal+eval;
    same filter semantics as scripts/query_retrieval.py).

    Modes:
      correct   -- model was correct (cal+eval only: train rows are excluded,
                   the adapted members saw those labels -- memorization bias).
      normal    -- true no-finding references (gt_normal & ~uncertain); the
                   high-precision lane for normal queries.
      flag      -- KEYED ON AGENT 1: candidates whose REPORT confirms the
                   Region-B-flagged class (flag_class, from model internals --
                   no query GT at deployment). This is the confident-error
                   review lane: Agent 1 says "the flag on class X may be a
                   confident miss", Agent 2 shows references whose reports
                   confirm X.
      contrast  -- the other half of the flag review: references whose report
                   confirms class X AND whose model read is X-positive (the
                   model DID call X there). X = flagged class when Region B
                   fires, else the query's argmax. cal+eval pool only (per-class
                   decisions come from the npz; train rows have no stored
                   per-class probs).

    On plain/flag/contrast modes a second-stage RERANK blends visual
    cosine with report-label agreement: score = (1-lam)*cosine + lam*agree,
    where agree is the fraction of the query's predicted probability mass that
    the candidate's report confirms (certain positives, -1 excluded). The
    QUERY side uses only the model's own pooled posterior -- no ground truth
    exists for a query at retrieval time; report labels are reference-side
    metadata only.
    Returns (rows, n_allow, lane_note)."""
    meta = _RETR_META
    allow = np.ones(len(_RETR_IDS), dtype=bool)
    allow &= meta["pred_pathology"].notna().values  # only rows with metadata (CLI parity)
    split_col = meta["split"].values if "split" in meta \
        else np.full(len(_RETR_IDS), "cal", dtype=object)
    gt_normal = meta["gt_normal"].fillna(False).values \
        if "gt_normal" in meta else np.zeros(len(_RETR_IDS), dtype=bool)
    lane_note = ""
    if mode == "correct":
        allow &= meta["correct"].fillna(False).values \
            & ~meta["uncertain"].fillna(False).values \
            & (split_col != "train")   # train excluded: adapted members memorized these labels
        lane_note = ("pool = cal+eval only (train excluded: adapted members "
                     "saw those labels)")
    elif mode == "normal":
        allow &= gt_normal & ~meta["uncertain"].fillna(False).values
        lane_note = "GT-normal references (no positive among certain labels)"
    elif mode == "flag":
        pats_l = [str(p) for p in NIH_PATHOLOGIES]
        if _RETR_GT is not None and flag_class in pats_l:
            fi = pats_l.index(flag_class)
            allow &= (_RETR_GT[:, fi] == 1) & _RETR_GTV[:, fi]
            lane_note = (f"Region-B flag on {flag_class} -> report-confirmed "
                         f"{flag_class} references (no query GT used)")
        else:
            # graceful fallback: unfiltered nearest neighbors (the retired
            # argmax pathology bucket was the exact failure mode the flag
            # lane exists to avoid)
            lane_note = ("flag fallback (" +
                         ("no label sidecar" if _RETR_GT is None else
                          "no Region-B flag") + ") -> plain nearest neighbors")
    elif mode == "contrast":
        pats_l = [str(p) for p in NIH_PATHOLOGIES]
        key = flag_class if flag_class in pats_l else pred_pathology
        if key in pats_l and _RETR_GT is not None:
            xi = pats_l.index(key)
            allow &= _RETR_DEC[:, xi] & (_RETR_GT[:, xi] == 1) & _RETR_GTV[:, xi]
            lane_note = (f"contrast: model read {key} AND report confirms {key} "
                         "(cal+eval pool -- the model DID call it there)")
        else:
            allow &= meta["correct"].fillna(False).values \
                & ~meta["uncertain"].fillna(False).values & (split_col != "train")
            lane_note = "contrast fallback -> correct-mode pool"
    allow &= _RETR_IDS != (query_id or "")
    n_allow = int(allow.sum())
    if n_allow == 0:
        return [], 0, lane_note
    labels, _ = _RETR_INDEX.knn_query(q, k=min(K_NEIGHBORS * 4,
                                               _RETR_INDEX.get_current_count()),
                                      filter=lambda li: bool(allow[int(li)]))
    cands = []
    for li in np.asarray(labels).ravel():
        li = int(li)
        if allow[li]:
            cands.append((li, float(_RETR_FEATS[li] @ q)))

    do_rerank = mode in ("plain", "flag", "contrast") \
        and p_q is not None and _RETR_GT is not None
    if do_rerank:
        def _agree(li: int) -> float:
            y, v = _RETR_GT[li], _RETR_GTV[li]
            denom = float(p_q[v & (y != 0)].sum())
            return float(p_q[v & (y == 1)].sum() / denom) if denom > 1e-6 else 0.0
        cands.sort(key=lambda rc: -((1.0 - lam) * rc[1] + lam * _agree(rc[0])))
        lane_note = (lane_note + " | " if lane_note else "") \
            + f"rerank (lambda={lam:.1f}): + report-label agreement"

    rows = []
    # flag/contrast modes: surface what the lane actually guarantees -- the
    # neighbor's report-positive set (key class first) AND the neighbor's full
    # predicted set (dominant first) in the pred_pathology column -- for
    # multilabel films the argmax alone is a co-finding, not the model's read.
    key_class = None
    if mode == "flag" and lane_note.startswith("Region-B flag"):
        key_class = flag_class
    elif mode == "contrast" and lane_note.startswith("contrast:"):
        key_class = flag_class if flag_class in [str(p) for p in NIH_PATHOLOGIES] \
            else pred_pathology
    show_report = key_class is not None
    for li, cos in cands[:K_NEIGHBORS]:
        if show_report:
            y, v = _RETR_GT[li], _RETR_GTV[li]
            pos = [str(NIH_PATHOLOGIES[j]) for j in range(len(y))
                   if v[j] and y[j] == 1]
            pos.sort(key=lambda p: p != key_class)   # keyed class first
            confirm = ", ".join(pos)
        else:
            confirm = ""
        pred_cell = str(meta["pred_pathology"].iloc[li])
        if show_report:
            ps = _pred_set(str(_RETR_IDS[li]))
            if ps:
                pred_cell = " + ".join(ps)
        rows.append({"rank": len(rows) + 1, "image_id": _RETR_IDS[li],
                     "sim": round(cos, 3),
                     "pred_pathology": pred_cell,
                     "pred_conf": round(float(meta["pred_conf"].iloc[li]), 3),
                     "split": str(split_col[li]),
                     "correct": bool(meta["correct"].iloc[li]),
                     "uncertain": bool(meta["uncertain"].iloc[li]),
                     "report_confirms": confirm})
    return rows, n_allow, lane_note


def _gallery_items(rows: list[dict]) -> list:
    out = []
    for r in rows:
        p = _PATH_BY_ID.get(r["image_id"])
        if not (p and _exists(p)):
            continue
        tag = "?" if r["uncertain"] else ("correct" if r["correct"] else "wrong")
        out.append((p, f"#{r['rank']} {r['pred_pathology']} sim {r['sim']:.2f} "
                       f"[{tag}·{r.get('split', '?')}]"))
    return out


def predict(image_path: str, retrieval_mode: str = "auto"):
    """One image -> Agent-1 table + banner + Agent-2 retrieval (UI payloads)."""
    empty_df = pd.DataFrame(columns=["pathology", "p_bar_ts", "youden_thr", "decision",
                                     "calib_confidence", "epistemic_std", "abstain_flag",
                                     "disagreement_flag", "gt", "error_type"])
    empty_retr = pd.DataFrame(columns=["rank", "image_id", "sim", "pred_pathology",
                                       "pred_conf", "split", "correct",
                                       "report_confirms"])
    empty_members = pd.DataFrame({"member": MEMBERS, "prob": [0.0] * len(MEMBERS)})
    banner = _banner(False, ["Processing..."])
    df, member_df, retr_df = empty_df, empty_members, empty_retr
    member_caption = retr_caption = ""
    gal: list = []
    agent3._LAST_EVIDENCE.clear()   # stale evidence from the previous case must not leak
    if not image_path or not _exists(image_path):
        return (_banner(False, ["Upload a CXR or pick a built-in example to begin."]),
                empty_df, empty_members, "", [], empty_retr, "",
                PLACEHOLDER_MD)
    try:
        image_id, gt = _resolve(image_path)
        probs, feat, maha_raw = _forward(image_path, image_id)
        df = _agent1_rows(probs, gt)

        p_raw = np.nanmean(probs, axis=0)
        valid = np.isfinite(p_raw)
        pred_idx = int(np.where(valid, p_raw, -np.inf).argmax())
        pred_pat = NIH_PATHOLOGIES[pred_idx]
        pred_conf = float(p_raw[pred_idx])
        member_df, member_caption = _member_bar(probs, pred_idx)

        maha_pct = _maha_pct(maha_raw)
        ood = maha_pct >= MAHA_HIGH_PCT
        abstain = [r["pathology"] for _, r in df.iterrows() if r["abstain_flag"]]
        disagree = [r["pathology"] for _, r in df.iterrows() if r["disagreement_flag"]]
        high = ood or bool(abstain) or bool(disagree)
        # Flag-keyed retrieval key: the Region-B-flagged class with the highest
        # calibrated confidence. Model-internal ONLY -- no query GT at deployment.
        flag_class = None
        if disagree:
            drows = df[df["disagreement_flag"]]
            flag_class = str(drows.loc[
                drows["calib_confidence"].astype(float).idxmax(), "pathology"])

        feat_n = (feat / max(float(np.linalg.norm(feat)), 1e-12)).astype(np.float32)
        # AUTO lane: the two-agent coupling without user action -- when Region B
        # fires, retrieve evidence for the flagged class; otherwise the
        # trust-calibration contrast set ("where the model was correct").
        eff_mode = retrieval_mode
        if retrieval_mode == "auto":
            eff_mode = "flag" if flag_class else "correct"
        retr, n_allow, lane = _retrieval(feat_n, pred_pat, eff_mode, image_id,
                                         p_q=np.where(valid, p_raw, 0.0),
                                         flag_class=flag_class)
        retr_df = pd.DataFrame(retr)
        gal = _gallery_items(retr)

        # Agent 3 evidence pack (deployment: include_gt=False -- no label
        # leaves the builder). Failures never break the prediction outputs.
        try:
            pred_set = _pred_set(image_id) or [pred_pat]
            zero_prev = set(((_QUALITY_STATS or {}).get("bias", {})
                             or {}).get("zero_prevalence_classes", []))
            bias = {"zero_prevalence_classes": sorted(zero_prev),
                    "predicted_in_zero_prevalence":
                        [c for c in pred_set if c in zero_prev],
                    "query_view": _VIEW_BY_ID.get(image_id)}
            ev = agent3.build_evidence(
                path=image_path, view=_VIEW_BY_ID.get(image_id), probs=probs,
                pathologies=list(NIH_PATHOLOGIES), youden=_YOUDEN,
                p_ts=_ts(p_raw) if valid.any() else None,
                pred_set=pred_set, flag_class=flag_class,
                maha_pct=maha_pct, ood=ood, neighbors=retr, gt=gt,
                include_gt=False, quality_stats=_QUALITY_STATS, bias=bias,
                pool_note=f"FULL library (train+cal+eval) n={len(_RETR_IDS)}")
            agent3._LAST_EVIDENCE.clear()
            agent3._LAST_EVIDENCE.update(ev)
        except Exception as e3:  # noqa: BLE001
            agent3._LAST_EVIDENCE.clear()
            print(f"[agent3] evidence build failed: "
                  f"{type(e3).__name__}: {e3}")

        lines = [("**Mahalanobis OOD** (fit on ReX-cal, scored vs the eval-split "
                  "reference): " + (f"**{maha_pct:.0f}th percentile** -- likely "
                  "out-of-distribution / shift. The one flag that survives shift "
                  "(though NOT an error ranker: 0.53 at power)." if ood else
                  f"{maha_pct:.0f}th percentile -- in-distribution range."))]
        if abstain:
            lines.append("**Abstain (Region A -- confidence)**: " + ", ".join(abstain)
                         + f" -- calibrated confidence under the cal top-{CONF_PCT:.0f}% "
                         "cutoff (bulk selective-prediction rule; full-set AUAC "
                         + ("0.965" if REGIME == "adapted" else "0.955") + ").")
        if disagree:
            if REGIME == "adapted":
                lines.append("**Disagreement triage (Region B)**: " + ", ".join(disagree)
                             + f" -- confident AND epistemic_std in the top {100 - UNC_PCT:.0f}% "
                             "within the confident set. On this ADAPTED ensemble disagreement "
                             "separates confident errors in the top-10%-confident stratum "
                             "(tail AUROC 0.679 vs 0.542 for confidence itself, DeLong "
                             "p=0.006): treat as an in-domain confident-error triage signal.")
            else:
                lines.append("**Disagreement triage (Region B)**: " + ", ".join(disagree)
                             + f" -- confident AND epistemic_std in the top {100 - UNC_PCT:.0f}% "
                             "within the confident set. On this PRETRAINED ensemble disagreement "
                             "is statistically equivalent to confidence (DeLong p=0.99): treat "
                             "as second-order triage, not a proven-error detector.")
        if gt is None:
            lines.append("*No ReX labels for this image -- error_type shows 'unknown'. "
                         "Upload a labels CSV/JSON to add GT.*")
        banner = _banner(high, lines)
        retr_caption = (f"Query prediction **{pred_pat}** (pooled p_raw={pred_conf:.3f}) | "
                        f"mode **{retrieval_mode}"
                        + (f" -> {eff_mode}" if eff_mode != retrieval_mode else "")
                        + f"** | candidates {n_allow}/{len(_RETR_IDS)} "
                        f"of the FULL library (train+cal+eval) | top-{len(retr)} self-excluded"
                        + (f" | {lane}" if lane else ""))
        if eff_mode == "flag":
            retr_caption += (" -- flag-keyed: key = Agent-1 Region-B flag "
                             f"({flag_class or 'none fired'}), from model internals; "
                             "candidates confirmed by reference-side reports only")
        if eff_mode == "contrast":
            retr_caption += (" -- contrast lane: references the model read positive "
                             f"on {flag_class or pred_pat} AND whose report confirms it")
    except Exception as e:  # never hard-crash the UI
        import traceback
        banner = _banner(True, [f"Prediction failed: <code>{type(e).__name__}: {e}</code>",
                                "<details><summary>traceback</summary><pre>"
                                + traceback.format_exc() + "</pre></details>"])
        df, member_df, retr_df, gal = empty_df, empty_members, empty_retr, []
        member_caption = retr_caption = ""
    return (banner, df, member_df, member_caption, gal, retr_df, retr_caption,
            PLACEHOLDER_MD)


# ---------------------------------------------------------------------------
# Agent 4 -- population improvement suggestions (lazy singleton)
# ---------------------------------------------------------------------------
_SUGG_EVIDENCE: dict | None = None
_SUGG_LOCK = threading.Lock()


def app_suggest(backend="auto"):
    """Population-level improvement suggestions (Agent 4). Never raises."""
    global _SUGG_EVIDENCE
    if _SUGG_EVIDENCE is None:
        with _SUGG_LOCK:
            if _SUGG_EVIDENCE is None:
                try:
                    if os.path.exists(agent4.EVIDENCE_JSON):
                        with open(agent4.EVIDENCE_JSON) as f:
                            _SUGG_EVIDENCE = json.load(f)
                        print(f"[agent4] loaded committed pack "
                              f"{agent4.EVIDENCE_JSON}")
                    else:
                        print("[agent4] no committed pack; building population "
                              "evidence (quality sweep, a few minutes)...")
                        inp = agent4.load_inputs()
                        _SUGG_EVIDENCE = agent4.build_evidence(inp)
                except Exception as e:                  # noqa: BLE001
                    print(f"[agent4] evidence pack unavailable: "
                          f"{type(e).__name__}: {e}")
                    _SUGG_EVIDENCE = {}
    try:
        return agent4.render_cached(_SUGG_EVIDENCE, backend=backend)
    except Exception as e:                              # noqa: BLE001
        return (f"**Agent 4 unavailable**: `{type(e).__name__}: {e}`. "
                "Run `python scripts/agent4_batch.py --analyze` to build the "
                "population evidence pack.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
if REGIME == "adapted":
    INTRO = """# CXR Agentic Demo -- Agent 1 (UQ + confident-error flagging) + Agent 2 (retrieval)

4-member ensemble (xrv NIH DenseNet + ConvNeXt-V2-L + RAD-DINO ViT +
Ark+ Swin-L@768), **LP-FT adapted on ReX-train** (production-realistic regime,
Phase 7) on 14 NIH pathologies, calibrated on ReX-cal (temperature T and
per-pathology Youden thresholds), with the **two-regions** deployment rule
(docs/uncertainty_evaluation.md §4.6.2): **confidence abstains in the bulk**
(full-set AUAC 0.965); **disagreement triages the confident stratum** -- on this
in-domain-adapted ensemble epistemic std separates confident errors in the
top-10%-confident stratum (tail AUROC 0.679 vs 0.542 for confidence itself,
DeLong p=0.006; entropy 0.755 is the strongest tail signal); **Mahalanobis
watches for shift** (OOD flag; not an error ranker at power). **Agent 2**
retrieves similar reference cases from a FULL-library HNSW index (train+cal+eval,
129,113 adapted RAD-DINO embeddings, recall@10 = 1.0000). **Auto mode** (default)
runs the two-agent coupling by itself: when Region B flags class X, it retrieves
**evidence** -- references whose reports confirm X (the flag lane, keyed from
model internals, no query GT); otherwise the **trust contrast** -- similar films
where the model was right. Manual modes: correct (cal+eval pool, train excluded:
memorization), GT-normal references, flag, contrast (references the model read
positive on X AND whose report confirms it -- the other half of the flag
review), or plain nearest neighbors -- the retrieval answers for the SAME model
as Agent 1 (sidecar consistency; train-row annotations come from the 3-member
224 subset). **Agent 3** turns the flag into a clinician-readable paragraph: it
synthesizes only over the measured evidence pack (image-quality metrics, member
disagreement, label ambiguity, dataset bias, retrieved neighbors), every claim
must cite its evidence id, and a mechanical faithfulness audit replaces any
uncited rendering with the deterministic template. **Agent 4** (button below)
is population-level: it analyses the eval-split error set across five lanes
(class data need, augmentation gaps, threshold levers, retrain vs fine-tune,
metadata slices) and returns *hypothesis-only, non-executing* suggestions,
under the same cite-everything + faithfulness-audit gate.
"""
else:
    INTRO = """# CXR Agentic Demo -- Agent 1 (UQ + confident-error flagging) + Agent 2 (retrieval)

4-member **pretrained (un-adapted)** ensemble (xrv NIH DenseNet + ConvNeXt-V2-L +
RAD-DINO ViT + Ark+ Swin-L@768) on 14 NIH pathologies, calibrated on ReX-cal
(temperature T and per-pathology Youden thresholds), with the **two-regions**
deployment rule (docs/uncertainty_evaluation.md): **confidence abstains in the
bulk** (AUAC 0.955); **disagreement triages the confident stratum** (honestly: on
this off-domain ensemble disagreement is statistically equivalent to confidence
at top-10%, DeLong p=0.99 -- the zero-adaptation comparison lens); **Mahalanobis
watches for shift** (OOD flag; not an error ranker at power). **Agent 2**
retrieves similar cal-split cases from the production HNSW index
(recall@10 = 1.0000) filtered by "where the model was correct" / "same predicted
pathology" -- the retrieval answers for the SAME model as Agent 1 (sidecar
consistency).
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(),
                   title=f"CXR Agentic Demo (ReX · {REGIME})") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="filepath", label="CXR input", height=400)
                gr.Markdown("### Built-in examples (eval split)")
                gallery = gr.Gallery(
                    value=[(e["image_path"], e["caption"]) for e in _EXAMPLES
                           if "caption" in e] or
                           [(e["image_path"], e["kind"]) for e in _EXAMPLES],
                    label="Examples", columns=2, height=260, show_label=False,
                    object_fit="cover")
                retr_mode = gr.Radio(
                    ["auto", "correct", "normal", "flag", "contrast",
                     "plain"], value="auto",
                    label="Agent 2 -- retrieval mode",
                    info="auto = the two-agent coupling (flag lane when Region B "
                         "fires, correct otherwise); correct = the model was right "
                         "(cal+eval pool; train excluded -- memorization); "
                         "normal = GT no-finding references; flag = keyed on "
                         "Agent-1's Region-B flag: candidates whose report "
                         "confirms the flagged class (evidence lane); contrast = "
                         "the other half of the flag review: references the model "
                         "read positive on the keyed class AND whose report "
                         "confirms it; plain = unfiltered nearest neighbors")
                labels_file = gr.File(
                    label="Optional labels CSV/JSON (basename -> pathology 0/1)",
                    file_types=[".csv", ".json"], type="filepath")
            with gr.Column(scale=2):
                banner = gr.HTML(_banner(False,
                                         ["Upload a CXR or pick an example to begin."]))
                table = gr.Dataframe(
                    headers=["pathology", "p_bar_ts", "youden_thr", "decision",
                             "calib_confidence", "epistemic_std", "abstain_flag",
                             "disagreement_flag", "gt", "error_type"],
                    datatype=["str", "number", "number", "number", "number", "number",
                              "bool", "bool", "str", "str"],
                    interactive=False, wrap=True,
                    label="Agent 1 -- per-pathology, two-regions flags. p_bar_ts / "
                          "youden_thr are temperature-scaled for display; decisions use "
                          "the cal-fit Youden cutoff on the raw scale (matches the "
                          "published evaluation). abstain_flag = Region A (low "
                          "calibrated confidence); disagreement_flag = Region B "
                          "(confident AND high ensemble std -- second-order triage only "
                          "on this ensemble).")
                with gr.Row():
                    member_bar = gr.BarPlot(
                        x="member", y="prob",
                        title="Per-member probability (dominant pathology)",
                        x_title="ensemble member", y_title="prob", height=240)
                    member_caption = gr.Markdown("")
                gr.Markdown("### Agent 2 -- similar reference cases "
                            "(full library: train+cal+eval)")
                with gr.Row():
                    retr_gallery = gr.Gallery(label="Top-k neighbors", columns=4,
                                              height=240, show_label=False,
                                              object_fit="cover")
                    retr_table = gr.Dataframe(
                        headers=["rank", "image_id", "sim", "pred_pathology",
                                 "pred_conf", "split", "correct", "report_confirms"],
                        datatype=["number", "str", "number", "str", "number", "str",
                                  "bool", "str"],
                        interactive=False, wrap=True,
                        label="Neighbors (sim = cosine on RAD-DINO space; "
                              "report_confirms = the reference's report confirms "
                              "the flagged class, flag mode only)")
                retr_caption = gr.Markdown("")
                gr.Markdown("### Agent 3 -- error explanation "
                            "(evidence-grounded, non-diagnostic)")
                explain_backend = gr.Radio(
                    ["auto", "template"], value="auto", label="Agent 3 backend",
                    info="auto = LLM synthesis (ollama local, then ollama cloud "
                         "if OLLAMA_API_KEY is set) gated by the mechanical "
                         "faithfulness audit -- any rendering that fails is "
                         "replaced by the deterministic template + red note; "
                         "template = deterministic rendering only, no LLM")
                expl_md = gr.Markdown(PLACEHOLDER_MD)
                explain_btn = gr.Button("Explain this prediction (Agent 3)",
                                        variant="secondary")
                status = gr.Markdown("Uploaded-labels status appears here.")
                gr.Markdown("### Agent 4 -- improvement suggestions "
                            "(population-level, non-executing)")
                sugg_md = gr.Markdown(PLACEHOLDER_MD)
                sugg_btn = gr.Button("Suggest improvements (Agent 4)",
                                     variant="secondary")

        outputs = [banner, table, member_bar, member_caption, retr_gallery,
                   retr_table, retr_caption, expl_md]

        gr.Examples(examples=[[e["image_path"]] for e in _EXAMPLES],
                    inputs=[img_in], fn=predict, outputs=outputs,
                    run_on_click=True, label="Click-to-run an example")
        img_in.change(predict, inputs=[img_in, retr_mode], outputs=outputs)
        retr_mode.change(predict, inputs=[img_in, retr_mode], outputs=outputs)
        explain_btn.click(
            lambda backend: agent3.render_cached(backend=backend),
            inputs=[explain_backend], outputs=[expl_md])
        sugg_btn.click(app_suggest, inputs=[explain_backend], outputs=[sugg_md])

        def on_labels_upload(path, current_image, mode):
            global _GT_LOOKUP
            if not path:
                status = f"Loaded **{len(_GT_LOOKUP)}** uploaded labels total."
            else:
                try:
                    new = parse_labels_table(path)
                    merged = dict(_GT_LOOKUP)
                    for bname, labs in new.items():
                        merged[bname] = {**merged.get(bname, {}), **labs}
                    _GT_LOOKUP = merged
                    status = (f"Loaded **{len(new)}** labels (**{len(_GT_LOOKUP)}** total).")
                except Exception as e:
                    status = f"Failed to parse labels file: `{type(e).__name__}: {e}`"
            return (status,) + tuple(predict(current_image, mode))

        labels_file.change(on_labels_upload, inputs=[labels_file, img_in, retr_mode],
                           outputs=[status] + outputs)

        def on_gallery(evt: gr.SelectData):
            v = evt.value
            path = None
            if isinstance(v, dict):
                img = v.get("image", v)
                path = (img.get("path") or img.get("url")) if isinstance(img, dict) else img
            elif isinstance(v, (tuple, list)):
                path = v[0]
            else:
                path = v
            if isinstance(path, dict):
                path = path.get("path") or path.get("url")
            return path

        gallery.select(on_gallery, outputs=[img_in])
    return demo


if __name__ == "__main__":
    print(f"[startup] regime={REGIME}")
    load_state()
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=APP_PORT, show_error=True)