"""Agent 3 -- Error Explanation Agent.

A *constrained renderer over structured evidence*: deterministic code computes
every signal (prediction state, member disagreement, label ambiguity, image
quality, dataset bias, retrieval-neighbor comparison); the LLM only synthesizes
a short clinician-readable paragraph under a cite-everything prompt; and a
mechanical faithfulness audit gates what reaches the UI (0-tolerance -- a
failing rendering is replaced by the deterministic template rendering plus a
red note).

Design constraints (approved plan):
  - The LLM NEVER sees pixels. Image-level claims come from computed metrics
    only (the ReXGradient DUA forbids sending images anywhere; pixel claims
    would be unauditable). Pixel-reading adjudicators (MAIRA-2, ChestX-Reasoner)
    stay proposed-not-built -- see docs/agent3_explanation.md.
  - Privacy in code: build_evidence cannot emit image_path / patient_id; the
    case reference is a sha1 prefix; ground-truth leaves are added only when
    include_gt=True (offline evaluation), never at deployment.
  - Backend chain: local ollama daemon (if reachable) -> Ollama cloud
    (https://ollama.com, Bearer $OLLAMA_API_KEY; env-only, never logged) ->
    deterministic template renderer.

stdlib + numpy + PIL only (no new pip deps).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from typing import Any

import numpy as np

from .utils import load_raw_gray

DISCLAIMER = "Non-diagnostic explanation of model behavior; not a medical diagnosis."

DEFAULT_CLOUD_MODEL = "glm-4.6"
CLOUD_BASE = "https://ollama.com"
LOCAL_BASE = "http://127.0.0.1:11434"
CHAT_TIMEOUT_S = 60
CHAT_TEMPERATURE = 0.2
CHAT_SEED = 1234


# ---------------------------------------------------------------------------
# Provider registry (kept OUT of interfaces.py: I/O layer, documented deviation)
# ---------------------------------------------------------------------------

PROVIDERS = {}


def register_provider(name):
    def _wrap(fn):
        PROVIDERS[name] = fn
        return fn
    return _wrap


def _chat(base, model, messages, api_key=None):
    """POST {base}/api/chat -> assistant content. stdlib urllib only."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": CHAT_TEMPERATURE, "seed": CHAT_SEED},
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base + "/api/chat", data=body, headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["message"]["content"])


@register_provider("ollama_local")
def ollama_local(messages, model=DEFAULT_CLOUD_MODEL, **kw):
    return _chat(LOCAL_BASE, model, messages)


@register_provider("ollama_cloud")
def ollama_cloud(messages, model=DEFAULT_CLOUD_MODEL, **kw):
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError("OLLAMA_API_KEY not set")
    # key is never logged or echoed; only its presence is used
    return _chat(CLOUD_BASE, model, messages, api_key=key)


@register_provider("template")
def template_provider(messages, model="", evidence=None, **_):
    if evidence is None:
        raise RuntimeError("template backend requires the evidence dict")
    return render_template(evidence)


def provider_chain(backend="auto"):
    """auto: local daemon, then Ollama cloud (only when OLLAMA_API_KEY is set),
    then the deterministic template renderer."""
    if backend == "template":
        return ["template"]
    chain = ["ollama_local"]
    if os.environ.get("OLLAMA_API_KEY"):
        chain.append("ollama_cloud")
    chain.append("template")
    return chain


# ---------------------------------------------------------------------------
# Prompt: cite-everything system prompt; the evidence pack IS the chain of
# thought (single-shot, temperature 0.2, fixed seed).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Agent 3 of a chest X-ray triage assistant. You receive a \
structured evidence pack (items labelled EV:<id>) computed by deterministic code about \
ONE image and a chest X-ray ensemble's prediction on it. Write a clinician-readable \
explanation.

HARD RULES:
1. Cite the evidence id for every factual claim, like [EV:pred.argmax]. Every claim must \
be traceable to a cited evidence item.
2. Never invent image findings. The pixels are NOT in the evidence; only measured \
image-quality statistics are. The only pathology names that exist are those in the \
evidence.
3. Mirror-asymmetry is a measured left/right positional difference. It is NOT a rotation \
estimate -- never use the word rotation. AP vs PA is a projection difference.
4. Neighbors are retrieval results, not ground truth. Say what they support and what \
they do not.
5. Dataset-bias statements only from bias evidence, with the actual numbers, phrased as \
"in this dataset ...".
6. Preserve uncertainty: use "consistent with", "supports", "suggests" -- never "the \
diagnosis is" or "this is". State absences plainly.
7. No patient identifiers, no file paths, no system or device names.

Output EXACTLY three short paragraphs, at most 180 words total:
P1: why this case was flagged (or why the ensemble agrees), from prediction, \
disagreement and quality evidence.
P2: what the retrieved neighbors support and do not support.
P3: the single best explanation category, exactly one of:
[image-quality artifact] / [label ambiguity] / [dataset bias: unseen class] / \
[genuine model error] / [no error signal present]

End with exactly this line:
""" + DISCLAIMER


def evidence_prompt(evidence):
    """[system, user] messages. The evidence pack IS the chain of thought:
    single-shot, temperature 0.2, fixed seed."""
    leaves = flatten_leaves(evidence)
    lines = ["EV:%s = %s" % kv for kv in sorted(leaves.items())]
    user = ("Evidence pack for one case:\n" + "\n".join(lines)
            + "\n\nWrite the three-paragraph explanation per the rules.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Image-quality metrics (measured quantities only, never interpretations).
# ---------------------------------------------------------------------------

def _resize_long_side(img01, long_side=1024):
    """Long side to 1024 so metrics are comparable across source resolutions."""
    h, w = img01.shape
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img01
    from PIL import Image
    im = Image.fromarray((np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8))
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                   Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def compute_quality(path):
    """Measured quality metrics; pure functions of the raw grayscale."""
    img2d, meta = load_raw_gray(path)
    if meta["dicom_hu"]:
        img01 = (np.clip(img2d.astype(np.float64), -1024, 1024) + 1024.0) / 2048.0
    else:
        img01 = img2d.astype(np.float64) / float(meta["maxval"])
    img01 = np.nan_to_num(img01, nan=0.0, posinf=1.0, neginf=0.0)

    h, w = img01.shape
    small = _resize_long_side(img01, 1024)
    sh, sw = small.shape
    flat = small.ravel()

    p1, p99 = np.percentile(flat, 1), np.percentile(flat, 99)
    # Laplacian variance (manual 3x3 kernel; no scipy dependency).
    lap = (-4.0 * small[1:-1, 1:-1]
           + small[:-2, 1:-1] + small[2:, 1:-1]
           + small[1:-1, :-2] + small[1:-1, 2:])
    # Tenengrad: mean squared Sobel response (manual kernels).
    gx = (small[:-2, 2:] + 2.0 * small[1:-1, 2:] + small[2:, 2:]
          - small[:-2, :-2] - 2.0 * small[1:-1, :-2] - small[2:, :-2])
    gy = (small[2:, :-2] + 2.0 * small[2:, 1:-1] + small[2:, 2:]
          - small[:-2, :-2] - 2.0 * small[:-2, 1:-1] - small[2:, :-2])

    # Left/right mirror correlation -> positional asymmetry. NOT rotation.
    if sw >= 4 and sh >= 2:
        a = small[:, : sw // 2]
        b = small[:, sw - sw // 2:][:, ::-1]
        av, bv = a.ravel() - a.mean(), b.ravel() - b.mean()
        den = float(np.linalg.norm(av) * np.linalg.norm(bv))
        mc = float(np.dot(av, bv) / den) if den > 0 else 1.0
        mc = max(-1.0, min(1.0, mc))
    else:
        mc = 1.0

    ys, xs = np.mgrid[0:sh, 0:sw]
    wsum = float(small.sum())
    if wsum > 0:
        cy = float((ys * small).sum() / wsum)
        cx = float((xs * small).sum() / wsum)
        off = float(np.hypot(cy - (sh - 1) / 2.0, cx - (sw - 1) / 2.0))
        co = off / max(0.5 * float(np.hypot(sh - 1, sw - 1)), 1e-12)
    else:
        co = 0.0

    return {
        "contrast_span": float(p99 - p1),
        "contrast_std": float(np.std(small)),
        "exposure_mean": float(np.mean(small)),
        "exposure_median": float(np.median(small)),
        "midgray_dev": float(abs(np.mean(small) - 0.5)),
        "clip_lo_frac": float(np.mean(flat < 0.02)),
        "clip_hi_frac": float(np.mean(flat > 0.98)),
        "laplacian_var": float(np.var(lap)),
        "tenengrad": float(np.mean(gx * gx + gy * gy)),
        "mirror_asymmetry": float(1.0 - mc),
        "centroid_offset": co,
        "aspect": float(w / max(h, 1)),
        "bit_depth": meta["bit_depth"],
        "width": int(w),
        "height": int(h),
    }


# ---------------------------------------------------------------------------
# Band assignment against eval-split percentile tables (fixed quintiles).
# ---------------------------------------------------------------------------

def load_quality_stats(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def assign_band(value, table, clip_frac=None):
    """Fixed quintile bands from the per-metric/per-view percentile table.
    Special case: total clipped fraction > 0.10 -> 'high clipping'.
    Missing/unusable table -> 'unavailable' (the demo still works)."""
    if table is None:
        return "unavailable"
    if clip_frac is not None and clip_frac > 0.10:
        return "high clipping"
    if not np.isfinite(value):
        return "unavailable"
    pct = table.get("percentiles", {})
    e = [pct.get(k) for k in ("p20", "p40", "p60", "p80")]
    if any(x is None for x in e):
        return "unavailable"
    if value < e[0]:
        return "very low"
    if value < e[1]:
        return "low"
    if value < e[2]:
        return "moderate"
    if value < e[3]:
        return "high"
    return "very high"


# ---------------------------------------------------------------------------
# Evidence builder -- every scalar leaf gets a dotted id for [EV:<id>] cites.
# ---------------------------------------------------------------------------

def flatten_leaves(ev, _prefix=""):
    """{dotted_id: value-string} over every scalar leaf (dicts/lists walked;
    None -> 'none')."""
    out = {}

    def _fmt(v):
        if v is None:
            return "none"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return "%.4f" % v
        return str(v)

    if isinstance(ev, dict):
        for k, v in ev.items():
            out.update(flatten_leaves(v, _prefix + str(k) + "."))
    elif isinstance(ev, (list, tuple)):
        for i, v in enumerate(ev):
            out.update(flatten_leaves(v, _prefix + str(i) + "."))
    else:
        out[_prefix[:-1]] = _fmt(ev)
    return out


def _sha1_prefix(path, n=8):
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:n]


def build_evidence(*, path, view, probs, pathologies, youden, p_ts, pred_set,
                   flag_class, maha_pct, ood, neighbors, gt=None,
                   include_gt=False, quality_stats=None, bias=None,
                   pool_note=""):
    """Deterministic evidence pack. PRIVACY IN CODE: `path` is used for the
    sha1 case_ref and for quality metrics and is never emitted; gt leaves are
    added only when include_gt=True (offline evaluation)."""
    probs = np.asarray(probs, dtype=np.float64)
    p_raw = np.nanmean(probs, axis=0)
    std = np.nanstd(probs, axis=0)
    valid = np.isfinite(p_raw)
    pred_idx = int(np.where(valid, p_raw, -np.inf).argmax())
    key_class = flag_class or str(pathologies[pred_idx])
    kj = list(pathologies).index(key_class) if key_class in pathologies else pred_idx
    thr_raw = youden.get(key_class)
    thr = 0.5 if thr_raw is None else float(thr_raw)

    pkj = probs[:, kj]
    above = int(np.sum(pkj >= thr))              # NaN >= thr is False
    below = int(np.sum(pkj[np.isfinite(pkj)] < thr))
    mstd = float(std[kj])
    p_ts_key = None
    if p_ts is not None and np.isfinite(p_ts[kj]):
        p_ts_key = float(p_ts[kj])

    ev = {
        "schema": "agent3_evidence_v1",
        "case_ref": _sha1_prefix(path),
        "view": view,
        "prediction": {
            "argmax": str(pathologies[pred_idx]),
            "argmax_p_raw": float(p_raw[pred_idx]),
            "argmax_p_ts": (float(p_ts[pred_idx])
                            if p_ts is not None and np.isfinite(p_ts[pred_idx])
                            else None),
            "max_class_p": float(np.nanmax(np.where(valid, p_raw, np.nan))),
            "predicted_set": list(pred_set),
            "flag_class": flag_class,
            "key_class": key_class,
            "key_class_p_raw": float(p_raw[kj]),
            "key_class_youden_thr": thr,
            "mahalanobis_pct": float(maha_pct) if maha_pct is not None else None,
            "ood_flag": bool(ood),
            "n_members": int(probs.shape[0]),
        },
        "member_disagreement": {
            "cls": key_class,
            "member_probs": {("m%d" % i): (float(v) if np.isfinite(v) else None)
                             for i, v in enumerate(probs[:, kj])},
            "pooled_p_raw": float(np.nanmean(pkj)),
            "pooled_p_ts": p_ts_key,
            "youden_thr": thr,
            "cross_member_std": mstd,
            "sign_split": "%d above / %d below threshold" % (above, below),
            "interpretation": ("cross-architecture member disagreement "
                               "(inductive-bias difference; NOT an image "
                               "interpretation)"),
        },
    }

    near = []
    for j, pat in enumerate(pathologies):
        if not valid[j]:
            continue
        t = youden.get(pat)
        t = 0.5 if t is None else float(t)
        if abs(float(p_raw[j]) - t) <= 0.10:
            near.append(pat)
    ev["label_ambiguity"] = {
        "near_threshold_classes": near,
        "max_class_p": float(np.nanmax(np.where(valid, p_raw, np.nan))),
    }
    if include_gt and gt is not None:
        unlabeled = [str(pathologies[j]) for j in range(len(pathologies))
                     if valid[j] and gt[j] == -1]
        if unlabeled:
            ev["label_ambiguity"]["unlabeled_flagged"] = unlabeled
        ev["prediction"]["gt_vector"] = {str(pathologies[j]): float(gt[j])
                                         for j in range(len(pathologies))
                                         if valid[j]}
        key_gt = float(gt[kj]) if gt[kj] in (-1.0, 0.0, 1.0) else None
        ev["label_ambiguity"]["key_class_gt"] = {
            -1.0: "unlabeled", 0.0: "negative", 1.0: "positive"}.get(key_gt)

    # image quality + percentile bands
    q = compute_quality(path)
    vtab = (quality_stats or {}).get("by_view", {}).get(view or "__none__", {})
    ev["image_quality"] = {}
    for k, v in q.items():
        ev["image_quality"][k] = v
    for metric in ("contrast_span", "contrast_std", "exposure_mean",
                   "midgray_dev", "clip_lo_frac", "clip_hi_frac",
                   "laplacian_var", "tenengrad", "mirror_asymmetry",
                   "centroid_offset"):
        cf = (q["clip_lo_frac"] + q["clip_hi_frac"]) \
            if metric.endswith("_frac") else None
        tab = vtab.get(metric)
        ev["image_quality"][metric + "_band"] = assign_band(q[metric], tab,
                                                            clip_frac=cf)
    ev["image_quality"]["view"] = view
    ev["image_quality"]["view_note"] = ("view tag from the acquisition manifest: "
                                        "AP vs PA is a projection difference, "
                                        "never rotation")
    ev["image_quality"]["asymmetry_note"] = ("mirror_asymmetry is a measured "
                                             "left/right positional difference, "
                                             "NOT a rotation estimate")

    # dataset bias (precomputed offline; see scripts/agent3_batch.py)
    ev["dataset_bias"] = dict(bias or {})

    # neighbors
    nb = {"n": len(neighbors)}
    if neighbors:
        sims = [float(x["sim"]) for x in neighbors]
        nb["sim_min"] = min(sims)
        nb["sim_max"] = max(sims)
        nb["split_counts"] = {s: sum(1 for x in neighbors
                                     if x.get("split") == s)
                              for s in ("train", "cal", "eval")}
        nb["model_correct_n"] = sum(1 for x in neighbors if x.get("correct"))
        nb["model_uncertain_n"] = sum(1 for x in neighbors
                                      if x.get("uncertain"))
        nb["report_confirms_n"] = sum(1 for x in neighbors
                                      if x.get("report_confirms"))
        nb["top"] = [{k: x.get(k) for k in
                      ("rank", "image_id", "sim", "pred_pathology",
                       "pred_conf", "split", "correct", "report_confirms")}
                     for x in neighbors[:5]]
        nb["note"] = "neighbors are retrieval results, not ground truth"
    if pool_note:
        nb["pool"] = pool_note
    ev["neighbors"] = nb
    return ev


# ---------------------------------------------------------------------------
# Deterministic template renderer (test oracle + audit-failure fallback).
# Produces the same 3-paragraph shape, fully cited, from the evidence alone.
# ---------------------------------------------------------------------------

def render_template(evidence):
    leaves = flatten_leaves(evidence)

    def g(key, default="unavailable"):
        return leaves.get(key, default)

    pred = evidence.get("prediction", {})
    mem = evidence.get("member_disagreement", {})
    amb = evidence.get("label_ambiguity", {})
    dq = evidence.get("image_quality", {})
    bias = evidence.get("dataset_bias", {})
    nb = evidence.get("neighbors", {})

    # P1 -- flag / agreement status
    if pred.get("flag_class"):
        p1 = ("Flagged on {k}: pooled p={p:.3f} vs threshold {t:.3f} with "
              "cross-member std {s:.4f} ({sp}); Mahalanobis {mp} "
              "[EV:prediction.flag_class][EV:member_disagreement.pooled_p_raw]"
              "[EV:member_disagreement.youden_thr][EV:member_disagreement.cross_member_std]"
              "[EV:member_disagreement.sign_split][EV:prediction.mahalanobis_pct]."
              ).format(k=pred.get("flag_class"),
                       p=float(leaves.get("prediction.key_class_p_raw", 0.0)),
                       t=float(leaves.get("prediction.key_class_youden_thr", 0.5)),
                       s=float(leaves.get("member_disagreement.cross_member_std", 0.0)),
                       sp=leaves.get("member_disagreement.sign_split", "?"),
                       mp=("%.1fth percentile" % pred["mahalanobis_pct"])
                       if pred.get("mahalanobis_pct") is not None else "not computed")
    else:
        p1 = ("No Region-B disagreement flag fired; argmax {a} at pooled "
              "p={p:.3f} [EV:prediction.argmax][EV:prediction.argmax_p_raw]."
              ).format(a=pred.get("argmax"), p=float(pred.get("argmax_p_raw", 0.0)))
    lowq = [m for m in ("contrast_span", "laplacian_var", "tenengrad")
            if dq.get(m + "_band") in ("very low", "low")]
    if lowq:
        # cite each printed band leaf -- a sentence naming several metrics but
        # citing only one band would fail the band-agreement audit
        p1 += (" Measured quality flags: "
               + ", ".join(
                   m.replace("_", " ") + "=" + str(dq.get(m + "_band"))
                   + " [EV:image_quality." + m + "_band]"
                   for m in lowq)
               + ".")
    else:
        p1 += (" No measured quality metric is in a low band "
               "[EV:image_quality.contrast_span_band].")

    # P2 -- neighbors
    if nb.get("n", 0) > 0:
        p2 = ("{n} neighbors retrieved (similarity {mn:.3f}-{mx:.3f}); the model "
              "read itself correct on {c} of them, reference reports confirm the "
              "flagged reading on {r} [EV:neighbors.n][EV:neighbors.sim_min]"
              "[EV:neighbors.sim_max][EV:neighbors.model_correct_n]"
              "[EV:neighbors.report_confirms_n].").format(
                  n=nb.get("n"), mn=float(nb.get("sim_min", 0.0)),
                  mx=float(nb.get("sim_max", 0.0)),
                  c=nb.get("model_correct_n", 0),
                  r=nb.get("report_confirms_n", 0))
        sc = nb.get("split_counts", {})
        if sc.get("train", 0) > sum(v for k2, v in sc.items() if k2 != "train"):
            p2 += " Most neighbors come from the train split [EV:neighbors.split_counts.train], consistent with retrieval over the training library, not independent confirmation."
    else:
        p2 = "No neighbor evidence was retrieved for this case [EV:neighbors.n]."

    # P3 -- category (deterministic priority: bias unseen > ambiguity >
    # quality artifact > flagged w/o explanation > no signal)
    zero_pred = bias.get("predicted_in_zero_prevalence", [])
    if zero_pred:
        cat = "[dataset bias: unseen class]"
        p3 = ("Category {c}: {cls} has zero prevalence in this dataset "
              "(prevalence {pv}) [EV:dataset_bias.predicted_in_zero_prevalence.0]"
              "[EV:dataset_bias.class_prevalence.{cls}].".format(
                  c=cat, cls=zero_pred[0],
                  pv=float((bias.get("class_prevalence", {}) or {})
                           .get(zero_pred[0], 0.0))))
    elif amb.get("near_threshold_classes"):
        cat = "[label ambiguity]"
        p3 = ("Category {c}: the pooled probability for {cls} sits near its "
              "decision threshold (borderline call) "
              "[EV:label_ambiguity.near_threshold_classes.0]."
              ).format(c=cat, cls=", ".join(amb["near_threshold_classes"][:3]))
    elif lowq:
        cat = "[image-quality artifact]"
        p3 = ("Category {c}: measured quality bands are degraded "
              "({bands}).").format(
                  c=cat,
                  bands=", ".join(
                      m.replace("_", " ") + "=" + str(dq.get(m + "_band"))
                      + " [EV:image_quality." + m + "_band]"
                      for m in lowq))
    elif pred.get("flag_class"):
        cat = "[genuine model error]"
        p3 = ("Category {c}: members disagree with no measured quality or "
              "label-ambiguity explanation present in the evidence "
              "[EV:member_disagreement.cross_member_std].".format(c=cat))
    else:
        cat = "[no error signal present]"
        p3 = ("Category {c}: the ensemble agrees and no measured quality, "
              "ambiguity or bias signal fires [EV:prediction.argmax]."
              ).format(c=cat)

    text = "\n\n".join([p1, p2, p3]) + "\n" + DISCLAIMER
    return text


# ---------------------------------------------------------------------------
# Faithfulness audit (mechanical; 0-tolerance gate before anything reaches UI).
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")
_CITE_RE = re.compile(r"\[EV:([^\]\[]+)\]")

# words that are band vocabulary; used for band-agreement checking
_BAND_WORDS = ("very low", "low", "moderate", "high", "very high",
               "high clipping", "unavailable")


def _numeric_leaves(leaves):
    """id -> list of finite floats: direct numeric leaves, plus numbers
    embedded in string leaves (e.g. '2 above / 1 below threshold')."""
    out = {}
    for k, v in leaves.items():
        try:
            f = float(v)
            if np.isfinite(f):
                out[k] = [f]
                continue
        except (ValueError, TypeError):
            pass
        vals = [float(x) for x in _NUM_RE.findall(str(v))]
        vals = [x for x in vals if np.isfinite(x)]
        if vals:
            out[k] = vals
    return out


def _matches(x, y, tol_rel=0.02, tol_abs=0.011):
    return abs(x - y) <= max(tol_abs, tol_rel * max(abs(x), abs(y)))


def audit(text, evidence):
    """Mechanical faithfulness audit of a rendering against the evidence.

    Returns dict(ok=bool, failures=[str], warnings=[str]). Hard checks
    (0-tolerance): citation validity, numeric grounding, class-name grounding,
    band agreement, length cap. Coverage of the evidence axes is advisory."""
    failures, warnings = [], []
    leaves = flatten_leaves(evidence)
    nums = _numeric_leaves(leaves)

    cited = _CITE_RE.findall(text)
    if not cited:
        failures.append("no citations at all")
    for cid in cited:
        if cid not in leaves:
            failures.append("invalid citation [EV:%s]" % cid)

    # numeric grounding: every number in a cited sentence must match a number
    # from one of that sentence's cited leaves. Ranges ("0.83-0.90") are split
    # first so the hyphen is not parsed as a minus sign. Citation ids are
    # stripped before scanning -- "near_threshold_classes.0" must not count
    # the leaf-index digit as a claimed number.
    for sent in re.split(r"(?<=[.!?])\s+", text):
        bare = _CITE_RE.sub("", sent)
        sent_nums = [float(x) for x in
                     _NUM_RE.findall(re.sub(r"(?<=\d)-(?=\d)", " ", bare))]
        sent_cites = _CITE_RE.findall(sent)
        if not sent_nums or not sent_cites:
            continue
        pool = []
        for cid in sent_cites:
            if cid in leaves:
                pool.extend(nums.get(cid, []))
        for x in sent_nums:
            if not any(_matches(x, y) for y in pool):
                failures.append(
                    "ungrounded number %.4g in sentence citing %s"
                    % (x, ",".join(sent_cites)))

    # class-name grounding (exact, lexicon-based): a rendering may only name
    # pathologies from the NIH-14 lexicon, and any pathology it names must be
    # present somewhere in the evidence -- an invented finding fails here.
    from .config import NIH_PATHOLOGIES
    ev_blob = " ".join(leaves.values()) + " " + " ".join(leaves.keys())
    lex = {p for p in NIH_PATHOLOGIES} | {"No Finding", "no-finding",
                                          "Pleural Thickening"}
    for tok in set(re.findall(r"\b[A-Z][a-zA-Z_]{2,}\b", text)):
        if tok in lex:
            if tok.replace("_", " ") not in ev_blob.replace("_", " "):
                failures.append("pathology named but absent from evidence: %s"
                                % tok)

    # band agreement: when a sentence names a quality metric, the band phrase
    # immediately following it must match the evidence band (longest phrase
    # wins, so "low" inside "very low" is not a mismatch). Matching is local
    # to the metric ("contrast span=very low, laplacian var=low" is valid);
    # a band word elsewhere in the sentence is not attributed to this metric.
    for sent in re.split(r"(?<=[.!?])\s+", text):
        low = sent.lower()
        for leaf_id, val in leaves.items():
            if not leaf_id.endswith("_band") or val is None:
                continue
            stem = leaf_id.split(".")[1].replace("_band", "").replace("_", " ")
            if not stem or stem not in low:
                continue
            m = re.search(re.escape(stem) + r"\s*[=:]*\s*([a-z ]{1,20})", low)
            if not m:
                continue
            present = [bw for bw in _BAND_WORDS if re.search(
                r"\b%s\b" % re.escape(bw), m.group(1))]
            if not present:
                continue
            said = max(present, key=len)
            if said != str(val).lower():
                failures.append(
                    "band mismatch for %s: text says '%s', evidence '%s'"
                    % (leaf_id, said, val))

    # length cap (hard; plan: <=180 words + disclaimer)
    body = text.replace(DISCLAIMER, "").strip()
    if len(body.split()) > 200:
        failures.append("too long: %d words" % len(body.split()))

    # coverage (advisory): which axes the text touches
    touched = [ax for ax in ("prediction", "member_disagreement",
                             "image_quality", "neighbors")
               if any(cid.startswith(ax) for cid in cited)]
    if len(touched) < 3:
        warnings.append("coverage touches only %d of 4 axes: %s"
                        % (len(touched), touched))
    return {"ok": not failures, "failures": failures, "warnings": warnings}


# ---------------------------------------------------------------------------
# Public entry point: render with the provider chain + faithfulness gate.
# ---------------------------------------------------------------------------

_LAST_EVIDENCE = {}       # set by the demo app after each predict()
_LAST_RENDER = {}         # {"text", "backend", "audit", "error"}


def render(evidence, backend="auto", model=DEFAULT_CLOUD_MODEL):
    """Render an explanation for an evidence pack.

    Chain (auto): ollama_local -> ollama_cloud (only when OLLAMA_API_KEY is
    set) -> deterministic template renderer. Any LLM rendering that fails the
    mechanical faithfulness audit is replaced by the template rendering plus a
    red note (0-tolerance gate). Never raises.
    """
    msgs = evidence_prompt(evidence)
    last_err = None
    for name in provider_chain(backend):
        if name == "template":
            text = render_template(evidence)
            return {"text": text, "backend": "template",
                    "audit": audit(text, evidence),
                    "error": (None if last_err is None else
                              "LLM backends unavailable (%s); template "
                              "rendering shown." % last_err)}
        try:
            text = PROVIDERS[name](messages=msgs, model=model, evidence=evidence)
        except Exception as e:                       # noqa: BLE001 -- chain step
            last_err = "%s: %s" % (type(e).__name__, e)
            continue
        res = audit(text, evidence)
        if res["ok"]:
            return {"text": text, "backend": name, "audit": res, "error": None}
        # 0-tolerance: a failed rendering never reaches the UI
        return {"text": render_template(evidence),
                "backend": name + "->template", "audit": res,
                "error": ("LLM rendering failed the faithfulness audit "
                          "(%d failures); template rendering shown."
                          % len(res["failures"]))}
    text = render_template(evidence)
    return {"text": text, "backend": "template", "audit": audit(text, evidence),
            "error": "no backend rendered (%s)" % last_err}


def render_cached(evidence=None, backend="auto", model=DEFAULT_CLOUD_MODEL):
    """UI entry point (gradio button): renders the last evidence pack. Reads
    the module-level cache when called with no argument. Never raises."""
    if evidence is None:
        evidence = _LAST_EVIDENCE
    if not evidence:
        return "*Agent 3 has nothing to explain yet -- run a prediction first.*"
    try:
        res = render(evidence, backend=backend, model=model)
    except Exception as e:                           # noqa: BLE001 -- UI safety
        return ("**Agent 3 unavailable** (%s: %s). The explanation lane needs "
                "an LLM backend or the template fallback; see "
                "docs/agent3_explanation.md." % (type(e).__name__, e))
    _LAST_RENDER.clear()
    _LAST_RENDER.update(res)
    if res.get("error"):
        head = "**Agent 3 note**: %s\n\n" % res["error"]
    elif res.get("backend") == "template":
        head = "*Template rendering (deterministic; no LLM call).*\n\n"
    else:
        head = ""
    if res["audit"].get("warnings"):
        head += ("*Faithfulness warnings*: %s\n\n"
                 % "; ".join(res["audit"]["warnings"]))
    return head + res["text"]
