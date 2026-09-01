"""Agent 4 batch tooling: population analysis + constrained lane rendering.

Two modes (repo root):

  # 1) build the five-lane population evidence pack over the adapted eval split
  #    (--quality auto re-reads eval PNGs once and caches to quality_cache.npz)
  python scripts/agent4_batch.py --analyze --quality auto

  # 2) render the five lanes and write the improvement report
  python scripts/agent4_batch.py --render --backend template
  python scripts/agent4_batch.py --render --backend auto --model glm-4.6

Both may be combined:  --analyze --render --backend template

Outputs (adapted regime) land in runs/app_rex_adapted/agent4/:
  quality_cache.npz            per-image quality metrics (one-off, ~8k images)
  suggestions_evidence.json    the evidence pack (schema agent4_suggestion_evidence_v1)
  faithfulness_report.json     per-lane backend + 0-tolerance audit record
and the prose report at docs/agent4_improvement.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cxr_uncertainty.agent3 as a3
import cxr_uncertainty.agent4 as a4


def _log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# --analyze
# ---------------------------------------------------------------------------
def analyze(args) -> dict:
    """Load inputs -> five-lane evidence pack -> suggestions_evidence.json."""
    t0 = time.time()
    _log("[agent4] loading inputs ...")
    inp = a4.load_inputs()
    es = a4.build_error_set(inp)
    _log("[agent4] eval rows=%d  error rows=%d  error cells=%d  labelled=%d"
         % (len(inp["ids"]), len(es["err_rows"]), int(es["err"].sum()),
            int(es["labelled"].sum())))

    if args.quality == "auto":
        _log("[agent4] quality pass (mode=auto): re-reads eval PNGs once, "
             "then caches to %s" % os.path.basename(a4.QUALITY_NPZ))
    ev = a4.build_evidence(inp, k=args.k, n_boot=args.n_boot, seed=args.seed,
                           quality_mode=args.quality)

    os.makedirs(a4.OUT_DIR, exist_ok=True)
    with open(a4.EVIDENCE_JSON, "w") as f:
        json.dump(ev, f, indent=2)
    prov = ev["provenance"]
    _log("[agent4] evidence -> %s" % a4.EVIDENCE_JSON)
    _log("[agent4]   n_eval=%s  err_rate=%.2f%%  leaves=%d  (%.1fs)"
         % (prov["n_eval"], prov["err_rate_pct"],
            len(a3.flatten_leaves(ev)), time.time() - t0))
    return ev


def load_evidence() -> dict:
    if not os.path.exists(a4.EVIDENCE_JSON):
        raise SystemExit("no evidence pack at %s -- run --analyze first"
                         % a4.EVIDENCE_JSON)
    with open(a4.EVIDENCE_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# --render
# ---------------------------------------------------------------------------
def render_all(ev, args) -> None:
    """Render every lane through the provider chain + 0-tolerance gate."""
    rows, n_gate_fail = [], 0
    for lane in a4.LANES:
        t0 = time.time()
        rejected_text = None
        if args.backend == "template":
            text = a4.render_template_lane(ev, lane)
            aud, backend_used, llm_err = a4.audit(text, ev, lane), "template", None
        else:
            res = a4.render(ev, lane, backend="auto", model=args.model)
            text, aud = res["text"], res["audit"]
            backend_used, llm_err = res["backend"], res.get("error")
        lat = time.time() - t0
        # The pre-swap verdict is the scientific record: it says WHY an LLM
        # rendering was rejected. a4.render() substitutes the template itself,
        # so keep both the verdict and the text that lost.
        llm_audit = {"ok": bool(aud["ok"]), "failures": list(aud["failures"]),
                     "warnings": list(aud.get("warnings", []))}
        if not aud["ok"]:                    # 0-tolerance: template + red note
            n_gate_fail += 1
            rejected_text = text
            text = a4.render_template_lane(ev, lane)
            aud = a4.audit(text, ev, lane)
        rows.append({"lane": lane, "backend": backend_used,
                     "latency_s": round(lat, 2), "llm_error": llm_err,
                     "audit_ok": bool(aud["ok"]), "failures": aud["failures"],
                     "warnings": aud.get("warnings", []),
                     "llm_audit": llm_audit, "rejected_rendering": rejected_text,
                     "rendering": text})
        _log("[agent4] %-24s backend=%-22s audit_ok=%s (%.1fs)%s"
             % (lane, backend_used, aud["ok"], lat,
                ("" if llm_audit["ok"] else
                 "  REJECTED: " + "; ".join(llm_audit["failures"]))))

    os.makedirs(a4.OUT_DIR, exist_ok=True)
    with open(a4.FAITH_JSON, "w") as f:
        json.dump({"n_lanes": len(rows), "n_gate_failures": n_gate_fail,
                   "backend": args.backend, "model": args.model,
                   "schema": ev.get("schema"),
                   "provenance": ev.get("provenance"),
                   "rows": rows}, f, indent=2)

    write_report(ev, rows, n_gate_fail, args)
    _log("[agent4] %d lanes, %d gate failures -> %s"
         % (len(rows), n_gate_fail, a4.FAITH_JSON))


def write_report(ev, rows, n_gate_fail, args) -> None:
    """docs/agent4_improvement.md -- the 'Error Analysis & Improvement
    Recommendation Report' deliverable."""
    prov = ev.get("provenance", {})
    out = args.report or a4.DEFAULT_REPORT
    n_ok = sum(1 for r in rows if r["audit_ok"])
    with open(out, "w") as f:
        f.write("# Agent 4 — Error Analysis & Improvement Recommendation "
                "Report\n\n")
        f.write("*Generated by `scripts/agent4_batch.py` from "
                "`%s` (schema `%s`). Population-level and correlational: "
                "every lane is a hypothesis over one fixed evaluation split, "
                "not a validated intervention.*\n\n"
                % (prov.get("eval_artifact_path"), ev.get("schema")))
        f.write("## Provenance\n\n| field | value |\n|---|---|\n")
        for k in ("eval_artifact_path", "n_eval", "n_err_rows", "n_err_cells",
                  "labelled_cells", "err_rate_pct", "calibration_sha",
                  "temperature", "seed", "n_boot", "generated_utc"):
            if k in prov:
                f.write("| `%s` | %s |\n" % (k, prov[k]))
        f.write("\n## Faithfulness gate\n\n")
        f.write("Backend requested: `%s` (model `%s`). "
                "**%d/%d lanes passed** the 0-tolerance audit; %d rendering(s) "
                "were replaced by the deterministic template.\n\n"
                % (args.backend, args.model, n_ok, len(rows), n_gate_fail))
        f.write("| lane | shipped backend | shipped audit | why the LLM "
                "rendering was rejected |\n|---|---|---|---|\n")
        for r in rows:
            la = r.get("llm_audit") or {"ok": True, "failures": []}
            f.write("| %s | `%s` | %s | %s |\n"
                    % (r["lane"], r["backend"],
                       "pass" if r["audit_ok"] else "**FAIL**",
                       ("; ".join(la["failures"]) if not la["ok"]
                        else "—")))
        f.write("\n## Suggestions\n")
        for r in rows:
            f.write("\n### %s\n\n" % r["lane"].replace("_", " "))
            if r["llm_error"]:
                f.write("> **Agent 4 note**: %s\n\n" % r["llm_error"])
            if r["warnings"]:
                f.write("> *Faithfulness warnings*: %s\n\n"
                        % "; ".join(r["warnings"]))
            f.write(r["rendering"] + "\n")
    _log("[agent4] report -> %s" % out)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true",
                    help="build the five-lane evidence pack")
    ap.add_argument("--render", action="store_true",
                    help="render the five lanes + write the report")
    ap.add_argument("--quality", choices=("auto", "skip", "never"),
                    default="auto",
                    help="image-quality pass for lane (b): auto computes and "
                         "caches, skip uses the cache only, never disables")
    ap.add_argument("--backend", choices=("auto", "template"),
                    default="template",
                    help="template = deterministic renderer only (the gate "
                         "that must pass first); auto walks the LLM chain")
    ap.add_argument("--model", default=a3.DEFAULT_CLOUD_MODEL)
    ap.add_argument("--k", type=int, default=6, help="embedding clusters, lane (e)")
    ap.add_argument("--n-boot", type=int, default=a4.N_BOOT)
    ap.add_argument("--seed", type=int, default=a4.SEED)
    ap.add_argument("--report", default=None,
                    help="override docs/agent4_improvement.md")
    args = ap.parse_args()

    if not (args.analyze or args.render):
        ap.error("nothing to do: pass --analyze and/or --render")

    ev = analyze(args) if args.analyze else None
    if args.render:
        render_all(ev if ev is not None else load_evidence(), args)


if __name__ == "__main__":
    main()
