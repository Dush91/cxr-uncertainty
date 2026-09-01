"""Interactive demo: CXR epistemic-uncertainty & confident-error flagging.

A small Gradio Blocks app that drives the existing per-image inference path
(`cli.py` recipe) on a single CXR and renders FOUR confident-error flags:

  * **disagreement** (threshold policy) -- in-distribution; provably at chance
    under distribution shift.
  * **conformal triage** (LAC refer) -- distribution-free coverage guarantee
    in-distribution; breaks under shift.
  * **Mahalanobis OOD** -- image-level feature-density score on frozen RAD-DINO
    [CLS] features; the only flag that beats chance under shift (Kermany 0.82,
    COVID 0.65) -- the star under shift.
  * **representation-mismatch** (class-conditional density) -- per-pathology pos/neg
    Gaussians; flags a CONFIDENT prediction whose representation sits in the opposite
    class cluster. The ONE flag that catches confident in-distribution FP/FN where the
    other three are blind (members agree -> no disagreement; in-distribution -> no
    Mahalanobis; confident single label -> no conformal refer). SPLIT across two feature
    spaces: RAD-DINO [CLS] for the 6 pathologies it ships (Pneumonia/Pneumothorax/Edema/
    Cardiomegaly/Atelectasis/Fibrosis) and Ark+ (Swin-L@768, PCA-256) for the 2 it
    uniquely lifts (Infiltration/Hernia) -- 8 total, re-fit on the 4-member confident
    set. Each beats the confidence/disagreement baseline on leak-free OpenI-D.
    Production deploys FFR=0.20 (catches 20-100% of confident errors at a ~4-30%
    false-flag). Ark+ is now a VOTING member of the ensemble; the Mahalanobis flag
    (still RAD-DINO [CLS], which beats Ark+ for global OOD) is unchanged.

The ensemble is the 4-member architecture-diverse set **xrv DenseNet** + **ConvNeXt-V2-L**
+ **RAD-DINO ViT** + **Ark+ Swin-L** (this session: Ark+ joins voting -- in-distribution
confident errors 10->4, AURC halved, coverage holds; Mahalanobis stays on RAD-DINO).

This is a thin read-only wrapper: it imports the live inference stack and reads
pre-fit artifacts at startup; it does NOT modify any ``cxr_uncertainty`` module.

Run:
    pip install "gradio>=4.0"
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/demo_app.py
then open http://0.0.0.0:7860 in the studio's port-forwarded browser.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

import gradio as gr

from cxr_uncertainty.config import RiskConfig, NIH_PATHOLOGIES
from cxr_uncertainty.models import CXREnsemble
from cxr_uncertainty.utils import load_image_tensor
from cxr_uncertainty.uncertainty import estimate
from cxr_uncertainty.interfaces import get_risk_policy
from cxr_uncertainty.feature_uq import extract_member, MahalanobisOOD
from cxr_uncertainty.repmismatch import RepMismatch
# Importing these (also pulled in by the package __init__) registers the
# "conformal_triage" risk policy + the ts_only* calibrators on import.
import cxr_uncertainty.conformal  # noqa: F401
import cxr_uncertainty.production  # noqa: F401

PNEUMONIA_IDX = NIH_PATHOLOGIES.index("Pneumonia")        # 6
# 4-member production ensemble (plan §R.5 + this session's 4-member re-fit):
# Ark+ (Swin-L@768) JOINS the voting ensemble -- compare_ensembles.py showed Ark+
# alone (4-member) is a net win over the 3-member: in-distribution confident errors
# 10->4, AURC 0.0030->0.0015, conformal coverage holds (0.897); Kermany confident
# errors 291->196 with Mahalanobis still beating chance (0.77). biomedclip (the
# 5th member) is the one that adds confident-agree-wrong, so it is NOT included.
# The Mahalanobis OOD flag (flag 3) STAYS on RAD-DINO [CLS] features -- Ark+ is
# decisively WORSE than RAD-DINO for global OOD under shift (plan §R.6: Kermany
# 0.63 vs 0.82, Covid 0.43 vs 0.62) even though it WINS class-conditional
# rep-mismatch -- opposite optimal feature spaces for the two tasks.
MEMBERS = ["xrv_nih", "convnextv2", "raddino", "arkswin"]
# 4-member production sidecar (build_4mem_production.py): temperature_T=0.6335,
# Mondrian tau_p (OpenI-C, unchanged -- coverage guarantee preserved), and an
# augmented `youden` field. On the 4-member the model is accurate enough that the
# §O Pareto guard reverts EVERY pathology to its openiC Youden (Mass openiC-only
# already gives FP 47.7%/TPR 100% -- the 3-member needed role-B augmentation to
# reach 47%; the 4-member gets there naturally), so youden_source="augmented_4mem"
# but every per-pathology source is "openiC". The youden values are on the 4-member
# temp-scaled (T=0.6335) scale -- consistent with the 4-member voting p_bar.
CONFORMAL_SIDECAR = "runs/augmented_4mem/conformal_sidecar_4mem.json"
# Mahalanobis Gaussian is re-fit from OpenI-C (in-distribution, Pneumonia pos/neg);
# the percentile REFERENCE is the out-of-sample in-distribution score distribution
# (OpenI-D, fit-on-C/scored-on-D). Using the in-sample OpenI-C scores as the
# reference would be biased low (training points sit closer to their own centroid),
# making every out-of-sample image look "OOD" -- which would wreck the
# in-distribution-low / OOD-high contrast the demo is built around.
MAHA_FIT_ARRAYS = "runs/phase4_features/arrays_openiC.npz"
MAHA_REF_ARRAYS = "runs/phase4_features/arrays_openiD.npz"
# The 4th flag: class-conditional-density (representation-mismatch) confident-error
# detector (plan §Q/§R). TWO detectors are loaded and SPLIT by pathology so each
# pathology is scored by exactly one detector (no double-FFR). Re-fit on the 4-member
# confident set (this session): the better-calibrated 4-member has fewer confident
# errors AND a stronger confidence baseline, so the shippable set shifts -- it GAINS
# Fibrosis (RAD-DINO 0.632 > conf 0.543) but LOSES Effusion (0.740 < conf 0.784) and
# Nodule (0.673 < conf 0.704) where mismatch no longer beats the now-stronger
# confidence. Net flag-4 coverage: 8 pathologies (was 9 on the 3-member).
#
#   * RAD-DINO [CLS] detector (runs/repmismatch_4mem/repmismatch.npz) -- ships 6:
#     {Pneumonia, Pneumothorax, Edema, Cardiomegaly, Atelectasis, Fibrosis}, scored on
#     the RAD-DINO [CLS] feature (first 768 of the 1536-d zero-padded layout, the same
#     `feat` the Mahalanobis flag uses). Pneumonia still ships (rec 67%) so the flagship
#     CXR2177 Pneumonia confident-FN (which RAD-DINO catches) is preserved.
#   * Ark+ (Swin-L@768) detector (runs/repmismatch_arkswin_4mem/repmismatch.npz) --
#     ships 8 total; the SPLIT assigns it the Ark+-ONLY 2 {Infiltration, Hernia} (the
#     ones RAD-DINO does not ship). Scored on the Ark+ 1536-d penultimate feature
#     **PCA-reduced to 256 dims** (fit leak-free on OpenI-C). Hernia rec 100%,
#     Infiltration rec 32%.
#
# Why a SPLIT and not "Ark+ only": per-image Ark+ is not a superset of RAD-DINO even
# though it ships more pathologies in aggregate. CXR2177's Pneumonia confident-FN is
# caught by RAD-DINO and MISSED by Ark+ (Ark+ catches ~75% of Pneumonia confident-FN;
# CXR2177 is in the missed 25%). Shipping Ark+ only would silently drop that flagship
# catch. The split keeps RAD-DINO where it is proven and uses Ark+ only for what it
# uniquely lifts -- no double-FFR, no regression on the flagship.
#
# CRITICAL (Ark+ path): the Gaussians were fit on PCA-256-reduced features but the PCA
# transform is NOT in the artifact. At startup we re-fit PCA(256, seed=0) on the same
# OpenI-C Ark+ features (deterministic -> byte-identical transform) and reduce the live
# 1536-d feature before scoring. Now that Ark+ is a VOTING member, the live 1536-d
# feature comes straight from the ensemble forward via `extract_member(eo, "arkswin")`
# (Ark+ native dim = 1536 = D_max, so Alignment.stack stores it raw -- identical to the
# fit-time arkswin_feats). No separate sidecar forward needed. The Mahalanobis OOD flag
# (flag 3) STAYS on RAD-DINO [CLS] (Ark+ is worse for global OOD, plan §R.6).
REPM_ARTIFACT_RD = "runs/repmismatch_4mem/repmismatch.npz"        # RAD-DINO [CLS] (6 shipped)
REPM_ARTIFACT_ARK = "runs/repmismatch_arkswin_4mem/repmismatch.npz"  # Ark+ Swin-L (Ark+-only 2)
ARK_FEATS_OPENIC = "runs/phase4_features/arkswin_feats_openiC.npz"
ARK_RAW_DIM = 1536          # Swin-L final-stage penultimate embedding
ARK_FEAT_DIM = 256          # PCA-reduced dims the Ark+ Gaussians were fit on
ARK_PCA_SEED = 0            # matches fit_repmismatch_member.py SPLIT_SEED
REPM_CONF_THRESH = 0.5   # confident gate for the flag (model leans to its decision)
MANIFEST = "data/manifest.parquet"
OOD_MANIFEST = "data/ood_manifest.parquet"
MAHA_HIGH_PCT = 95.0   # percentile threshold for the OOD banner
# Portable GT lookup (basename -> {pathology: 0/1}), generated by
# scripts/export_gt_labels.py from the leak-free manifest. Extends GT/error_type
# beyond the 6 hardcoded gallery examples to any of the 4,014 OpenI PA/AP images
# a user drops in via the file picker -- without a manifest.parquet dependency at
# request time. A user can also upload their own CSV/JSON via the UI (matched the
# same way, by basename) to check predictions against labels for other images.
GT_LABELS_DEFAULT = "data/openi_gt_labels.csv"

# ---------------------------------------------------------------------------
# Module singletons (built once at startup, reused per request)
# ---------------------------------------------------------------------------
_ENS: CXREnsemble | None = None
_CFG: RiskConfig | None = None
_MAHA: MahalanobisOOD | None = None
_MAHAR_REF: np.ndarray | None = None            # sorted OpenI-C scores for percentile
_REPM_RD: RepMismatch | None = None             # 4th flag, RAD-DINO [CLS] detector (5 shared paths)
_REPM_ARK: RepMismatch | None = None            # 4th flag, Ark+ (Swin-L) detector (4 new paths)
_GT_LOOKUP: dict[str, dict] = {}                 # basename -> {pathology: 0/1}, extends GT beyond the gallery
_ARK_NEW: set[str] = set()                       # pathologies scored by Ark+ only (Infiltration/Hernia)
_ARK_PCA = None                                  # PCA 1536->256 fit on OpenI-C (reproduces fit transform)
_YOUDEN: dict[str, float | None] = {}            # per-pathology calibrated decision cutoff
_TEMP_T: float = 1.0                            # conformal sidecar temperature (probability scale)


def _temp_scale(p) -> float:
    """Apply the conformal sidecar's temperature T to a single probability (or a
    scalar). T<1 sharpens away from 0.5. Returns p unchanged when T==1 or p is
    non-finite. This puts ``calibrated_decision`` on the SAME probability scale
    the conformal flag uses (the sidecar applies T to ``p_bar`` before LAC), so
    the two columns are directly comparable instead of disagreeing by a units
    artifact (raw p_bar vs temperature-scaled p_bar)."""
    if _TEMP_T == 1.0:
        return float(p)
    p = float(p)
    if not np.isfinite(p):
        return p
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1.0 - p)) / _TEMP_T))


def _temp_scale_arr(v: np.ndarray) -> np.ndarray:
    """Vectorized temperature scale that preserves NaN (invalid pathology slots)."""
    if _TEMP_T == 1.0:
        return np.asarray(v, dtype=np.float64)
    out = np.asarray(v, dtype=np.float64).copy()
    m = np.isfinite(out)
    pc = np.clip(out[m], 1e-9, 1 - 1e-9)
    out[m] = 1.0 / (1.0 + np.exp(-np.log(pc / (1.0 - pc)) / _TEMP_T))
    return out
_EX_BY_ABSPATH: dict[str, dict] = {}
_EX_BY_BNAME: dict[str, dict] = {}
_EXAMPLES: list[dict] = []


def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def build_examples() -> list[dict]:
    """Pick 6 examples (4 OpenI in-distribution + 2 OOD) by filter + .iloc[0].

    Selection is by predicate (no hardcoded IDs) and every chosen path is
    checked with os.path.exists, so evicted files (e.g. covid train) are
    skipped automatically. covid uses the **test** split only (train partly
    evicted -- see plan §I.5 caveat).
    """
    ex: list[dict] = []

    def add(row, caption, kind):
        labels = list(row["labels"])
        gt = dict(zip(NIH_PATHOLOGIES, labels))
        ex.append({
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "kind": kind,
            "gt": gt,
            "caption": caption,
        })

    if _exists(MANIFEST):
        m = pd.read_parquet(MANIFEST)
        oi = m[(m["source"] == "openi") & (m["split_role"] == "D")].copy()
        oi["_lab"] = oi["labels"].apply(lambda x: list(x))
        oi["_ok"] = oi["image_path"].apply(_exists)
        oi_ok = oi[oi["_ok"]]

        def pick(pred, caption, kind):
            sub = oi_ok[oi_ok["_lab"].apply(pred)]
            if len(sub):
                add(sub.iloc[0], caption, kind)

        EFF = NIH_PATHOLOGIES.index("Effusion")
        ATE = NIH_PATHOLOGIES.index("Atelectasis")
        pick(lambda x: sum(x) == 0, "OpenI normal (in-distribution, GT negative)", "openi")
        pick(lambda x: int(x[PNEUMONIA_IDX]) == 1, "OpenI Pneumonia (GT Pneumonia=1)", "openi")
        pick(lambda x: int(x[EFF]) == 1, "OpenI Effusion (GT Effusion=1)", "openi")
        pick(lambda x: int(x[ATE]) == 1 and int(x[PNEUMONIA_IDX]) == 1,
             "OpenI Atelectasis+Pneumonia", "openi")

    if _exists(OOD_MANIFEST):
        om = pd.read_parquet(OOD_MANIFEST)
        om["_lab"] = om["labels"].apply(lambda x: list(x))
        om["_ok"] = om["image_path"].apply(_exists)

        def pick_ood(src, caption, kind):
            sub = om[(om["source"] == src) & (om["split"] == "test")
                     & om["_lab"].apply(lambda x: int(x[PNEUMONIA_IDX]) == 1)
                     & om["_ok"]]
            if len(sub):
                add(sub.iloc[0], caption, kind)

        pick_ood("kermany",
                 "OOD: Kermany pediatric Pneumonia (shift; Mahalanobis should fire, "
                 "disagreement at chance)", "kermany")
        pick_ood("covid",
                 "OOD: COVID-19 (shift; confident-agree-wrong -- only Mahalanobis "
                 "survives)", "covid")
    return ex


def parse_labels_table(path: str) -> dict[str, dict]:
    """Parse a ground-truth labels file into ``{basename: {pathology: 0/1}}``.

    Accepts two shapes:
      * CSV -- a ``basename`` column plus any subset of NIH_PATHOLOGIES columns
        (see scripts/export_gt_labels.py for the canonical layout). Blank/NaN
        cells mean "not labeled" and are omitted, matching how the manifest's
        ``valid`` mask excludes e.g. Consolidation for OpenI.
      * JSON -- either a top-level ``{basename: {pathology: 0/1, ...}}`` dict,
        or a list of records each carrying a ``"basename"`` key.

    Best-effort: malformed rows/files raise so the caller can report a clear
    error instead of silently loading nothing.
    """
    # Pathology column names are matched case-insensitively (a lowercase
    # "pneumonia" or "Pneumonia " with trailing whitespace both resolve to the
    # canonical "Pneumonia") so a differently-cased label file doesn't silently
    # load rows with zero real label data -- if nothing matches at all, raise
    # rather than succeed with an empty mapping per row.
    if path.lower().endswith(".json"):
        import json
        raw = json.load(open(path))
        records = raw.items() if isinstance(raw, dict) else (
            (r.get("basename") or r.get("filename"), r) for r in raw)
        out = {}
        matched = set()
        for bname, r in records:
            key_lookup = {str(k).strip().lower(): k for k in r.keys()}
            row = {}
            for p in NIH_PATHOLOGIES:
                k = key_lookup.get(p.lower())
                if k is not None and r[k] not in (None, ""):
                    row[p] = int(r[k])
                    matched.add(p)
            out[bname] = row
        if not matched:
            raise ValueError(
                f"no recognized pathology keys found in {os.path.basename(path)} "
                f"(expected any of {NIH_PATHOLOGIES})")
        return out

    df = pd.read_csv(path)
    col_lookup = {str(c).strip().lower(): c for c in df.columns}
    bname_col = col_lookup.get("basename") or col_lookup.get("filename")
    if bname_col is None:
        raise ValueError(
            f"labels CSV must have a 'basename' (or 'filename') column; "
            f"got columns {list(df.columns)}")
    cols = [(p, col_lookup[p.lower()]) for p in NIH_PATHOLOGIES if p.lower() in col_lookup]
    if not cols:
        raise ValueError(
            f"no recognized pathology columns found in {os.path.basename(path)} "
            f"(expected any of {NIH_PATHOLOGIES}); got columns {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        out[str(r[bname_col])] = {p: int(r[c]) for p, c in cols if pd.notna(r[c])}
    return out


def load_state() -> None:
    """Build the ensemble, re-fit Mahalanobis, load the conformal sidecar, and
    the example set. Called once before demo.launch."""
    global _ENS, _CFG, _MAHA, _MAHAR_REF, _REPM_RD, _REPM_ARK, _ARK_NEW, _ARK_PCA
    global _EXAMPLES, _EX_BY_ABSPATH, _EX_BY_BNAME, _GT_LOOKUP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device)
    ens = CXREnsemble(members=MEMBERS, cfg=cfg)
    cfg.img_size = max(getattr(mb, "native_size", 224) for mb in ens.members)  # 224
    _ENS, _CFG = ens, cfg

    d = np.load(MAHA_FIT_ARRAYS, allow_pickle=True)
    rad_feats = np.asarray(d["rad_feats"], dtype=np.float64)
    gt = np.asarray(d["gt"], dtype=np.float64)
    _MAHA = MahalanobisOOD().fit(rad_feats, gt[:, PNEUMONIA_IDX].astype(int))
    # Percentile reference: out-of-sample in-distribution scores (OpenI-D). Fall
    # back to scoring the fit set if the reference file is missing.
    if _exists(MAHA_REF_ARRAYS):
        rd = np.load(MAHA_REF_ARRAYS, allow_pickle=True)
        ref = np.asarray(rd["mahalanobis"], dtype=np.float64) if "mahalanobis" in rd \
            else _MAHA.score(np.asarray(rd["rad_feats"], dtype=np.float64))
    else:
        ref = _MAHA.score(rad_feats)
    _MAHAR_REF = np.sort(np.asarray(ref, dtype=np.float64))

    # Youden-optimal per-pathology decision thresholds, fit on the OpenI-C
    # in-distribution calibration set. IMPORTANT: thresholds are fit on the
    # TEMPERATURE-SCALED pooled mean (the same T the conformal sidecar applies
    # before LAC), so `calibrated_decision` / `youden_thr` share the conformal
    # flag's probability scale. Without this the two columns disagree partly by
    # a units artifact (raw p_bar vs temperature-scaled p_bar); with it, any
    # remaining Youden-says-present / conformal-says-absent disagreement is an
    # honest operating-point difference (Youden = sensitive ROC-optimal cutoff;
    # conformal tau_p = coverage-guaranteed, higher bar), not a scale mismatch.
    global _YOUDEN, _TEMP_T
    import json as _json
    _sc = _json.load(open(CONFORMAL_SIDECAR))
    _TEMP_T = float(_sc.get("temperature_T", 1.0))
    # Youden thresholds: prefer an augmented sidecar (plan §O) that carries a
    # `youden` field fit on the importance-weighted OpenI-C + role-B set --
    # already on the temperature-scaled scale (same T as the sidecar). Fall back
    # to re-fitting on the OpenI-C arrays when the sidecar has no `youden`
    # (the original path; backward-compatible with runs/conformal/...).
    if str(_sc.get("youden_source", "")).startswith("augmented") and "youden" in _sc:
        _YOUDEN = {pat: (None if _sc["youden"].get(pat) is None
                         else float(_sc["youden"].get(pat)))
                   for pat in NIH_PATHOLOGIES}
        _YOUDEN_SOURCE = "augmented (importance-weighted OpenI-C + role-B)"
    else:
        from sklearn.metrics import roc_curve
        _yd_probs = np.asarray(d["probs"], dtype=np.float64)      # (N, M, P)
        _yd_pbar_raw = np.nanmean(_yd_probs, axis=1) if _yd_probs.ndim == 3 else _yd_probs
        _yd_pbar = _temp_scale_arr(_yd_pbar_raw)                  # temperature-scaled
        _yd_gt = np.asarray(d["gt"], dtype=np.float64)            # (N, P)
        _YOUDEN = {}
        for i, pat in enumerate(NIH_PATHOLOGIES):
            p, y = _yd_pbar[:, i], _yd_gt[:, i]
            v = ~np.isnan(p) & ~np.isnan(y)
            p, y = p[v], y[v].astype(int)
            if y.sum() < 2 or y.sum() == len(y):
                _YOUDEN[pat] = None     # no positives (e.g. Consolidation absent in OpenI)
            else:
                fpr, tpr, thr = roc_curve(y, p)
                _YOUDEN[pat] = float(thr[int(np.argmax(tpr - fpr))])
        _YOUDEN_SOURCE = "openiC-only (temperature-scaled)"

    # 4th flag (representation-mismatch) -- SPLIT design (plan §Q + §R.5), re-fit on the
    # 4-member confident set (this session). The RAD-DINO [CLS] detector scores the 6
    # pathologies it ships {Pneumonia, Pneumothorax, Edema, Cardiomegaly, Atelectasis,
    # Fibrosis} (incl. the flagship CXR2177 Pneumonia confident-FN it catches); the Ark+
    # (Swin-L) detector scores the Ark+-ONLY pathologies {Infiltration, Hernia} that
    # RAD-DINO does not ship. 8 pathologies total, each scored by exactly one detector.
    # Ark+ is now a VOTING member (so its live feature comes from the ensemble forward,
    # `extract_member(eo, "arkswin")` -- no separate sidecar) and is NOT used by the
    # Mahalanobis flag (flag 3, still RAD-DINO [CLS], which beats Ark+ for global OOD).

    # (a) RAD-DINO detector -- scored on the raddino [CLS] feature (`feat`) extracted
    # from the ensemble output (same vector the Mahalanobis flag uses).
    if _exists(REPM_ARTIFACT_RD):
        _REPM_RD = RepMismatch.load(REPM_ARTIFACT_RD)
        _rd_shipped = [p for p in NIH_PATHOLOGIES if _REPM_RD.enabled.get(p, False)]
        print(f"[startup] rep-mismatch RAD-DINO detector loaded; shipped (6): "
              f"{_rd_shipped}")
    else:
        _REPM_RD = None
        print(f"[startup] RAD-DINO rep-mismatch artifact not found ({REPM_ARTIFACT_RD}); "
              "the RAD-DINO flag-4 pathologies are disabled.")

    # (b) Ark+ PCA. The Ark+ Gaussians were fit on PCA-256-reduced features but the PCA
    # transform is NOT in the artifact. Re-fit PCA(256, seed=0) on the SAME OpenI-C Ark+
    # features used at fit time (sklearn PCA with full SVD is deterministic given the
    # data + seed) -> the live 1536-d feature is reduced to the identical 256-d subspace
    # before scoring. The live 1536-d feature itself comes from the voting ensemble
    # forward (Ark+ is a voting member), NOT a separate sidecar -- extracted per-image in
    # predict() via extract_member(eo, "arkswin"). Best-effort: if the PCA fit fails the
    # Ark+-only pathologies simply have no flag-4 (the RAD-DINO 6 are unaffected).
    if "arkswin" in MEMBERS and _exists(ARK_FEATS_OPENIC):
        try:
            from sklearn.decomposition import PCA
            _Fc_ark = np.asarray(
                np.load(ARK_FEATS_OPENIC, allow_pickle=True)["feats"], dtype=np.float64)
            assert _Fc_ark.shape[1] == ARK_RAW_DIM, \
                f"expected {ARK_RAW_DIM}-d ark feats, got {_Fc_ark.shape[1]}"
            _ARK_PCA = PCA(n_components=ARK_FEAT_DIM, random_state=ARK_PCA_SEED).fit(_Fc_ark)
            _ark_var = float(_ARK_PCA.explained_variance_ratio_.sum())
            print(f"[startup] ark+ PCA {ARK_RAW_DIM}->{ARK_FEAT_DIM} on OpenI-C "
                  f"(retained var={_ark_var:.3f}) -- reproduces the fit-time transform")
        except Exception as e:
            _ARK_PCA = None
            print(f"[startup] ark+ PCA fit FAILED ({type(e).__name__}: {e}); "
                  "the Ark+-only rep-mismatch pathologies are disabled.")
    else:
        _ARK_PCA = None
        if "arkswin" not in MEMBERS:
            print("[startup] ark+ not in voting ensemble -- Ark+-only flag-4 disabled.")

    # (c) Ark+ detector + the per-pathology SPLIT. _ARK_NEW = pathologies scored by Ark+
    # = (Ark+ shipped) minus (RAD-DINO shipped) -- the ones it uniquely lifts. The
    # remaining shipped pathologies stay on the RAD-DINO detector.
    if _exists(REPM_ARTIFACT_ARK):
        _REPM_ARK = RepMismatch.load(REPM_ARTIFACT_ARK)
        _ark_shipped = {p for p in NIH_PATHOLOGIES if _REPM_ARK.enabled.get(p, False)}
        _rd_ship = {p for p in NIH_PATHOLOGIES if _REPM_RD is not None
                    and _REPM_RD.enabled.get(p, False)} if _REPM_RD is not None else set()
        _ARK_NEW = sorted(_ark_shipped - _rd_ship)
        _repm_all = sorted(_ark_shipped | _rd_ship)
        _ark_ready = _ARK_PCA is not None and "arkswin" in MEMBERS
        print(f"[startup] rep-mismatch Ark+ detector loaded; Ark+-only: "
              f"{_ARK_NEW}"
              + ("" if _ark_ready else " -- WARNING: PCA missing, Ark+-only inactive"))
        print(f"[startup] rep-mismatch flag 4 TOTAL shipped pathologies "
              f"({len(_repm_all)}): {_repm_all}")
    else:
        _REPM_ARK = None
        _ARK_NEW = set()
        print(f"[startup] Ark+ rep-mismatch artifact not found ({REPM_ARTIFACT_ARK}); "
              "the Ark+-only pathologies lose flag 4.")

    _EXAMPLES = build_examples()
    _EX_BY_ABSPATH = {os.path.abspath(e["image_path"]): e for e in _EXAMPLES}
    _EX_BY_BNAME = {os.path.basename(e["image_path"]): e for e in _EXAMPLES}

    if _exists(GT_LABELS_DEFAULT):
        try:
            _GT_LOOKUP = parse_labels_table(GT_LABELS_DEFAULT)
        except Exception as e:
            _GT_LOOKUP = {}
            print(f"[startup] failed to load {GT_LABELS_DEFAULT} "
                  f"({type(e).__name__}: {e}); GT lookup limited to the gallery.")

    print(f"[startup] device={device} img_size={cfg.img_size}")
    print(f"[startup] ensemble={ens.member_keys} predictable={len(ens.predictable)}")
    print(f"[startup] mahalanobis classes={_MAHA.classes_.tolist()} "
          f"n_per_class={_MAHA.n_per_class_}")
    qs = np.quantile(_MAHAR_REF, [0.5, 0.85, 0.95, 0.99])
    print(f"[startup] maha ref quantiles (50/85/95/99) = {qs}")
    print(f"[startup] conformal temperature T={_TEMP_T:.4f} (calibrated_decision & "
          f"conformal flag share this scale)")
    print(f"[startup] youden calibrated decision thresholds ({_YOUDEN_SOURCE}):")
    for pat, t in _YOUDEN.items():
        print(f"   - {pat:22s} {('%.4f' % t) if t is not None else 'n/a (no positives)'}")
    print(f"[startup] loaded {len(_EXAMPLES)} examples")
    for e in _EXAMPLES:
        print(f"   - [{e['kind']}] {e['image_id']}  "
              f"gt_pos={[p for p,v in e['gt'].items() if v]}")
    print(f"[startup] GT lookup: {len(_GT_LOOKUP)} labeled images from "
          f"{GT_LABELS_DEFAULT if _exists(GT_LABELS_DEFAULT) else '(none found)'} "
          "-- upload a CSV/JSON in the UI to extend it")


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------
def _maha_percentile(maha_raw: float) -> float:
    """Percentile of the live score vs the OpenI-C reference (high = OOD)."""
    n = len(_MAHAR_REF)
    if n == 0:
        return 0.0
    rank = int(np.searchsorted(_MAHAR_REF, maha_raw, side="right"))
    return rank / n * 100.0


def _calibrated_call(pathology: str, p_bar: float, gt, conf_thresh: float):
    """Calibrated (Youden) decision for one pathology + a consistent
    confidence / error_type, on the SAME temperature-scaled probability scale
    the conformal flag uses. The incoming ``p_bar`` is the raw pooled mean;
    it is temperature-scaled here (the Youden thresholds were fit on the
    temperature-scaled scale in load_state), so ``calibrated_decision`` and
    the conformal ``lac_refer`` are directly comparable.

    Returns (youden_thr, calibrated_decision, calib_confidence, error_type).
    ``calib_confidence`` is the (temperature-scaled) probability the model
    assigns to the call it actually made (p if present, 1-p if absent). A weak
    present sits just above the sensitive Youden cutoff but well below the
    conformal tau_p -- tentative, not confident -- which the old
    |p_bar-0.5|*2 "confidence" got wrong.

    Honest caveat: for pathologies where the model discriminates poorly
    (e.g. Pneumonia AUROC ~0.66 on OpenI), the Youden cutoff is forced very low
    and flags a large fraction of GT-negative images too, so a "weak present"
    here is NOT selectively flaggable -- it is statistically indistinguishable
    from many normals. No flag catches that; only a better model does.
    """
    t = _YOUDEN.get(pathology)
    if t is None:
        return None, -1, None, "excluded (no labels in OpenI)"
    p = _temp_scale(p_bar)                  # temperature-scaled, matches Youden + conformal
    dec = int(p >= t)
    calib_conf = p if dec == 1 else (1.0 - p)
    if gt is None:
        err = "unknown"
    elif dec == gt:
        err = "confident_correct" if calib_conf >= conf_thresh else "tentative_correct"
    elif dec == 1 and gt == 0:
        err = "confident_fp" if calib_conf >= conf_thresh else "tentative_fp"
    elif dec == 0 and gt == 1:
        err = "confident_fn" if calib_conf >= conf_thresh else "tentative_fn"
    else:
        err = "tentative_wrong"
    return t, dec, calib_conf, err


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


def predict(image_path: str):
    """Run the 4-member ensemble + all four flags on one image; return UI payloads."""
    if not image_path or not _exists(image_path):
        return ("", gr.update(), 0.0, "", gr.update(), "",
                _banner(False, ["Upload a CXR or pick a built-in example to begin."]))

    try:
        # Resolve GT: built-in examples first (by abspath, then basename), else the
        # GT_LOOKUP table (default export + any user-uploaded CSV/JSON, matched by
        # basename) -- covers any of the 4,014 OpenI images, not just the gallery.
        # Still None (fully unknown) for images that appear in neither.
        ex = _EX_BY_ABSPATH.get(os.path.abspath(image_path))
        if ex is None:
            ex = _EX_BY_BNAME.get(os.path.basename(image_path))
        if ex is not None:
            gt, image_id = ex["gt"], ex["image_id"]
        else:
            gt = _GT_LOOKUP.get(os.path.basename(image_path))
            image_id = os.path.splitext(os.path.basename(image_path))[0]

        x = load_image_tensor(image_path, img_size=_CFG.img_size, device=_CFG.device)
        eo = _ENS.forward_ensemble_full(x)
        probs = eo.per_member_probs[:, 0, :]                  # (M, P)
        unc = estimate(probs, None, pathologies=eo.pathologies)

        cfg_thr = replace(_CFG, risk_policy="threshold")
        rep_thr = get_risk_policy("threshold")(image_id, unc, cfg_thr, gt_labels=gt)
        cfg_conf = replace(_CFG, risk_policy="conformal_triage",
                           conformal_sidecar=CONFORMAL_SIDECAR)
        rep_conf = get_risk_policy("conformal_triage")(image_id, unc, cfg_conf,
                                                       gt_labels=gt)

        # Mahalanobis OOD flag (flag 3) -- STILL on RAD-DINO [CLS] features (the only
        # detector that beats chance under shift; unchanged by the Ark+ upgrade).
        _, feat = extract_member(eo, "raddino")                # (1536,) zero-padded [CLS]
        maha_raw = float(_MAHA.score(feat)[0])
        maha_pct = _maha_percentile(maha_raw)

        # Rep-mismatch flag (flag 4) -- SPLIT: the Ark+-ONLY pathologies (_ARK_NEW, on the
        # 4-member = Infiltration/Hernia) are scored on Ark+ (Swin-L) features; the 6
        # RAD-DINO pathologies stay on the RAD-DINO [CLS] `feat` above (unchanged). The
        # Ark+ detector was fit on PCA-256-reduced OpenI-C features, so reduce the live
        # 1536-d Ark+ embedding through the startup PCA before scoring. Ark+ is now a
        # VOTING member, so its live feature comes straight from the ensemble forward
        # (extract_member(eo, "arkswin") -- Ark+ native dim 1536 = D_max, so Alignment.stack
        # stores it raw, identical to the fit-time arkswin_feats). No separate sidecar
        # forward. Decisions/confidence come from the 4-member pooled p_bar. Best-effort:
        # if the PCA is missing the Ark+-only pathologies stay quiet (RAD-DINO 6 live).
        ark_feat = None
        if _ARK_PCA is not None and "arkswin" in MEMBERS:
            try:
                _, _raw_ark = extract_member(eo, "arkswin")     # (1536,) raw Swin-L feat
                _raw_ark = np.asarray(_raw_ark, dtype=np.float64).reshape(1, -1)
                ark_feat = _ARK_PCA.transform(_raw_ark)[0]      # (256,) PCA-reduced
                del _raw_ark
            except Exception:
                ark_feat = None

        # ---- per-pathology table (sorted by p_bar desc) ----
        rows = []
        for pr in rep_thr.pathologies:
            # match the conformal row for the same pathology
            cf = next((c for c in rep_conf.pathologies if c.pathology == pr.pathology), None)
            gt_val = pr.gt if pr.gt is not None else ""
            maha_note = (f"⚠ OOD (pct={maha_pct:.0f}%)" if maha_pct >= MAHA_HIGH_PCT
                         else f"pct={maha_pct:.0f}%")
            yt, cal_dec, cal_conf, err = _calibrated_call(
                pr.pathology, float(pr.p_bar), pr.gt, _CFG.conf_thresh)
            p_cal = _temp_scale(float(pr.p_bar))   # display on the shared (temp-scaled) scale
            # 4th flag: representation-mismatch fires only on a CONFIDENT prediction
            # (calib_conf >= 0.5) for a SHIPPED pathology, when the member representation
            # favours the opposite class of the confident call -- the one signal that
            # catches confident in-distribution FP/FN where the other three flags are
            # blind. SPLIT: the 2 Ark+-only pathologies (_ARK_NEW = Infiltration/Hernia
            # on the 4-member) are scored on the PCA-256 Ark+ embedding `ark_feat`; the
            # 6 RAD-DINO-shipped pathologies stay on the RAD-DINO [CLS] `feat` (the same
            # vector the Mahalanobis flag uses) -- byte-identical to the pre-Ark+ demo,
            # so CXR2177's Pneumonia confident-FN still fires. 8 total shipped.
            if pr.pathology in _ARK_NEW:
                rep_flag = bool(
                    _REPM_ARK is not None and ark_feat is not None and cal_dec in (0, 1)
                    and _REPM_ARK.flag(ark_feat, cal_dec, cal_conf, pr.pathology,
                                       REPM_CONF_THRESH))
            else:
                rep_flag = bool(
                    _REPM_RD is not None and cal_dec in (0, 1)
                    and _REPM_RD.flag(feat, cal_dec, cal_conf, pr.pathology,
                                      REPM_CONF_THRESH)) if _REPM_RD is not None else False
            rows.append({
                "pathology": pr.pathology,
                "p_bar": round(p_cal, 4),
                "youden_thr": round(yt, 4) if yt is not None else None,
                "calibrated_decision": cal_dec,
                "calib_confidence": round(cal_conf, 4) if cal_conf is not None else None,
                "epistemic_std": round(float(pr.epistemic_std), 4),
                "threshold_flag": bool(pr.risk_flag),
                # Exclude masked pathologies (lac_set_size == -1, e.g. Consolidation
                # absent from OpenI): they are "refer" by construction, not a real
                # high-risk signal, so they must NOT light the refer column.
                "conformal_refer": (bool(getattr(cf, "lac_refer", False))
                                    and getattr(cf, "lac_set_size", 1) != -1) if cf else False,
                "repmismatch_flag": rep_flag,
                "maha_note": maha_note,
                "gt": gt_val,
                "error_type": err,
            })
        df = pd.DataFrame(rows).sort_values("p_bar", ascending=False).reset_index(drop=True)

        # ---- per-member disagreement bar (dominant / top-flagged pathology) ----
        flagged = [pr for pr in rep_thr.pathologies if pr.risk_flag] + \
                  [c for c in rep_conf.pathologies
                   if getattr(c, "lac_refer", False)
                   and getattr(c, "lac_set_size", 1) != -1]
        if flagged:
            pi = list(eo.pathologies).index(
                max(flagged, key=lambda r: r.p_bar).pathology)
        else:
            pi = int(np.nanargmax(unc.p_bar))
        member_vals = eo.per_member_probs[:, 0, pi].cpu().numpy().astype(float)
        member_df = pd.DataFrame({
            "member": MEMBERS,
            "prob": [float(v) if not np.isnan(v) else 0.0 for v in member_vals],
        })
        std = float(np.nanstd(member_vals))
        dom_path = eo.pathologies[pi]
        agree = "members AGREE (low std) -- confident-agree-wrong risk if OOD" \
            if std < _CFG.unc_thresh else \
            "members DISAGREE (high std) -- disagreement flag earns its keep"
        member_caption = (f"**Per-member probability for {dom_path}** (p_bar="
                          f"{unc.p_bar[pi]:.2f}, epistemic_std={std:.3f}). {agree}.")

        # ---- banner ----
        # Real conformal refers: exclude masked pathologies (lac_set_size == -1, e.g.
        # Consolidation is absent from OpenI so it is "refer" by construction -- NOT a
        # high-risk signal). Counting it would turn the banner HIGH on every image.
        conf_refer_paths = [p.pathology for p in rep_conf.pathologies
                            if getattr(p, "lac_refer", False)
                            and getattr(p, "lac_set_size", 1) != -1]
        conf_any = len(conf_refer_paths) > 0
        # 4th flag: confident in-distribution FP/FN the other three miss.
        repm_paths = [r["pathology"] for r in rows if r["repmismatch_flag"]]
        repm_any = len(repm_paths) > 0
        high = maha_pct >= MAHA_HIGH_PCT or rep_thr.any_high_risk or conf_any or repm_any
        lines = []
        if maha_pct >= MAHA_HIGH_PCT:
            lines.append(f"**Mahalanobis OOD**: score at the {maha_pct:.0f}th percentile "
                         f"of in-distribution OpenI-C &rarr; likely out-of-distribution.")
        if rep_thr.any_high_risk:
            lines.append(f"**Disagreement flag**: {', '.join(rep_thr.high_risk_pathologies)}")
        else:
            lines.append("**Disagreement flag**: none (in-distribution; at chance under shift).")
        if conf_any:
            lines.append(f"**Conformal triage (refer)**: {', '.join(conf_refer_paths)}")
        else:
            lines.append("**Conformal triage**: no refer (coverage holds in-distribution; "
                         "breaks under shift).")
        if repm_any:
            lines.append(f"**Rep-mismatch flag**: {', '.join(repm_paths)} -- the "
                         f"representation (RAD-DINO [CLS] for the 6 pathologies it ships, "
                         f"Ark+ Swin-L for the 2 it uniquely lifts -- Infiltration/Hernia) "
                         f"favours the OPPOSITE of a confident call; likely confident FP/FN "
                         f"(the case the other flags miss).")
        else:
            lines.append("**Rep-mismatch flag**: none (no confident prediction whose "
                         "representation sits in the opposite class cluster).")
        if gt is None:
            lines.append("*Ground truth unknown for this image -- error_type shows "
                         "'unknown'. Upload a labels CSV/JSON below to add it.*")
        elif ex is None:
            lines.append("*Ground truth from the uploaded/bundled labels file "
                         "(not the built-in gallery).*")
        banner = _banner(high, lines)

        out_err = ""
    except Exception as e:  # never hard-crash the UI
        import traceback
        banner = _banner(True, [f"Prediction failed: <code>{type(e).__name__}: {e}</code>",
                                "<details><summary>traceback</summary><pre>"
                                + traceback.format_exc() + "</pre></details>"])
        member_df = pd.DataFrame({"member": MEMBERS, "prob": [0.0] * len(MEMBERS)})
        return (banner, pd.DataFrame(), 0.0, "", member_df, "", "")
    finally:
        try:
            del x  # noqa: F821
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    maha_pct_str = (f"{maha_pct:.1f}%  (raw score {maha_raw:.1f}; "
                    f"ref 50/85/95 pct = "
                    f"{np.quantile(_MAHAR_REF, .5):.0f}/"
                    f"{np.quantile(_MAHAR_REF, .85):.0f}/"
                    f"{np.quantile(_MAHAR_REF, .95):.0f})")
    return (banner, df, float(maha_pct), maha_pct_str, member_df, member_caption, "")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
INTRO = """# CXR Epistemic-Uncertainty & Confident-Error Flagging Demo
Upload a chest X-ray (or pick a built-in example) to see FOUR confident-error flags side by side,
computed live from a 4-member architecture-diverse ensemble (**xrv DenseNet** + **ConvNeXt-V2-L** +
**RAD-DINO ViT** + **Ark+ Swin-L@768**) on 14 NIH pathologies. Ark+ joins the voting ensemble this
session (in-distribution confident errors 10→4, AURC halved, coverage holds); the Mahalanobis OOD
flag stays on RAD-DINO [CLS] (Ark+ is worse for global OOD). The rep-mismatch flag (flag 4) is SPLIT:
RAD-DINO [CLS] for the 6 pathologies it ships + Ark+ (Swin-L, PCA-256) for the 2 it uniquely lifts
(Infiltration/Hernia) -- 8 total.

| Flag | What it measures | Works when |
|---|---|---|
| **Disagreement** (threshold) | ensemble members disagree (`epistemic_std &ge; 0.15` while confident) | in-distribution; **at chance under shift** |
| **Conformal triage** (LAC refer) | distribution-free set is not a single label &rarr; refer | in-distribution coverage guarantee; **breaks under shift** |
| **Mahalanobis OOD** | RAD-DINO feature is far from the in-distribution density | **survives shift** (the star under shift) |
| **Rep-mismatch** (class-conditional density) | a CONFIDENT call whose representation (RAD-DINO [CLS] or **Ark+ Swin-L**) sits in the *opposite* class cluster | **confident in-distribution FP/FN** (the case the other three miss) |

Two complementary headlines:
* **Under shift** (Kermany pediatric pneumonia, COVID-19): the members *confidently agree*, so
  **disagreement** stays quiet and the conformal coverage guarantee breaks -- but **Mahalanobis**
  fires high (the only detector that beats chance under shift: Kermany ~0.82, COVID ~0.65).
* **In-distribution confident errors** (FP/FN): all of disagreement / Mahalanobis / conformal can
  miss a confident *agree*-wrong call. The **rep-mismatch** flag catches it -- the representation
  betrays the error even when the members agree. It ships NINE pathologies: the original five on
  RAD-DINO [CLS] (Pneumonia, Pneumothorax, Edema, Cardiomegaly, Atelectasis) plus four the RAD-DINO
  detector could NOT separate, lifted by Ark+ (Swin-L) features -- Infiltration ~0.87, Effusion ~0.80,
  Nodule ~0.76, Hernia ~0.97 -- each beating the confidence baseline on leak-free OpenI-D. Together
  they flag 20-85% of confident FP/FN at a ~4-30% false-flag rate (FFR=0.20 production). Only shipped
  where it measurably beats the baseline on leak-free OpenI-D.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="CXR Uncertainty Demo") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="filepath", label="CXR input", height=420)
                gr.Markdown("### Built-in examples")
                gallery = gr.Gallery(
                    value=[(e["image_path"], e["caption"]) for e in _EXAMPLES],
                    label="Examples", columns=3, height=220, show_label=False,
                    object_fit="cover")
                ex_paths = [e["image_path"] for e in _EXAMPLES]
                gr.Markdown("### Ground-truth labels (optional)")
                labels_file = gr.File(
                    label="Upload a labels CSV/JSON to check GT for images outside "
                          "the gallery (basename -> pathology 0/1; see "
                          "scripts/export_gt_labels.py for the CSV layout)",
                    file_types=[".csv", ".json"], type="filepath")
                labels_status = gr.Markdown(
                    f"Loaded **{len(_GT_LOOKUP)}** labels from `{GT_LABELS_DEFAULT}` "
                    "at startup." if _GT_LOOKUP else
                    f"No default labels file found at `{GT_LABELS_DEFAULT}` -- run "
                    "scripts/export_gt_labels.py, or upload one above.")
            with gr.Column(scale=2):
                banner = gr.HTML(_banner(False,
                                         ["Upload a CXR or pick a built-in example to begin."]))
                with gr.Row():
                    maha_slider = gr.Slider(0, 100, value=0, interactive=False,
                                            label="Mahalanobis OOD percentile (vs OpenI-C)",
                                            info="high = more out-of-distribution / likely wrong")
                    maha_text = gr.Textbox(label="Mahalanobis score (raw + reference)",
                                           interactive=False)
                table = gr.Dataframe(
                    headers=["pathology", "p_bar", "youden_thr", "calibrated_decision",
                             "calib_confidence", "epistemic_std", "threshold_flag",
                             "conformal_refer", "repmismatch_flag", "maha_note", "gt",
                             "error_type"],
                    datatype=["str", "number", "number", "number", "number", "number",
                              "bool", "bool", "bool", "str", "str", "str"],
                    interactive=False, wrap=True,
                    label="Per-pathology predictions & flags. p_bar / youden_thr are "
                          "temperature-scaled (same scale as the conformal flag). "
                          "calibrated_decision = Youden cutoff (sensitive); conformal_refer "
                          "= coverage-guaranteed LAC set ambiguous. repmismatch_flag = the "
                          "representation (RAD-DINO [CLS] for the 6 pathologies it ships, "
                          "Ark+ Swin-L for the 2 it uniquely lifts -- Infiltration/Hernia) "
                          "favours the opposite of a "
                          "confident call (confident FP/FN suspicion; only for the 8 shipped "
                          "pathologies, only on confident predictions). They CAN disagree -- an "
                          "honest operating-point difference, not a bug.")
                with gr.Row():
                    member_bar = gr.BarPlot(
                        x="member", y="prob",
                        title="Per-member probability (dominant / top-flagged pathology)",
                        x_title="ensemble member", y_title="prob", height=260)
                    member_caption = gr.Markdown("")
                err_html = gr.HTML(visible=False)

        outputs = [banner, table, maha_slider, maha_text, member_bar,
                   member_caption, err_html]

        gr.Examples(examples=[[p] for p in ex_paths], inputs=[img_in], fn=predict,
                    outputs=outputs, run_on_click=True, label="Click-to-run an example")

        img_in.change(predict, inputs=[img_in], outputs=outputs)

        def on_labels_upload(path, current_image):
            """Merge an uploaded labels file into _GT_LOOKUP (uploaded entries
            override any default for the same basename), then re-run predict()
            on the current image so the effect is visible immediately."""
            global _GT_LOOKUP
            if not path:
                status = f"Loaded **{len(_GT_LOOKUP)}** labels total."
            else:
                try:
                    new = parse_labels_table(path)
                    # Per-pathology merge (not a whole-dict overwrite): an uploaded
                    # file naming only e.g. Hernia for an image that already has 13
                    # other known labels should override just Hernia, not erase the
                    # rest.
                    merged = dict(_GT_LOOKUP)
                    for bname, labs in new.items():
                        merged[bname] = {**merged.get(bname, {}), **labs}
                    _GT_LOOKUP = merged
                    status = (f"Loaded **{len(new)}** labels from the uploaded file "
                              f"(**{len(_GT_LOOKUP)}** known total).")
                except Exception as e:
                    status = f"⚠ Failed to parse labels file: `{type(e).__name__}: {e}`"
            return (status,) + predict(current_image)

        labels_file.change(on_labels_upload, inputs=[labels_file, img_in],
                            outputs=[labels_status] + outputs)

        def on_gallery(evt: gr.SelectData):
            # Gradio 6 normalizes a Gallery item to a dict
            # {"image": {"path":.., "url":..}, "caption":..}; older versions used a
            # (url, caption) tuple. Extract a real path string in any shape so the
            # Image component + predict() get a file path, not a dict.
            v = evt.value
            path = None
            if isinstance(v, dict):
                img = v.get("image", v)
                if isinstance(img, dict):
                    path = img.get("path") or img.get("url")
                else:
                    path = img
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
    load_state()
    demo = build_ui()
    demo.queue()  # one GPU -- serialize inference
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)