"""Emit a publishable summary of the Agent 3 faithfulness batch.

    python scripts/redact_agent3_batch.py

The per-case batch records (`<image_id>.json`, `side_by_side.md`, and the
per-row `faithfulness_report.json`) are keyed by ReXGradient DICOM instance
UIDs and contain per-image measured quantities. They are patient-derived and
are NOT published; see PUBLISHING.md. This writes `gate_summary.json`, which
carries the citable gate result -- case counts, failure counts and the audit
checks that ran -- with no identifier and no per-image content.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES = [("agent3_batch", "deterministic template gate"),
           ("agent3_batch_llm", "live LLM lane")]
OUT = os.path.join(REPO, "runs", "app_rex_adapted", "agent3_gate_summary.json")


def summarise(d, label):
    path = os.path.join(REPO, "runs", "app_rex_adapted", d,
                        "faithfulness_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rep = json.load(f)
    rows = rep.get("rows", [])
    kinds = collections.Counter(r.get("kind") for r in rows)
    backends = collections.Counter(r.get("backend") for r in rows)
    failures = collections.Counter(
        f.split(" ")[0] + " " + f.split(" ")[1] if len(f.split(" ")) > 1 else f
        for r in rows for f in (r.get("failures") or []))
    errors = collections.Counter(
        (r.get("llm_error") or "none").split(";")[0] for r in rows)
    return {
        "label": label,
        "n_cases": rep.get("n_cases", len(rows)),
        "n_gate_failures": rep.get("n_gate_failures"),
        "backend_requested": rep.get("backend"),
        "cases_by_kind": dict(kinds),
        "renderings_by_backend": dict(backends),
        "audit_failure_kinds": dict(failures),
        "llm_outcomes": dict(errors),
        # lets a reader confirm this summary derives from the local record
        # without the record itself being published
        "source_sha256_prefix": hashlib.sha256(
            open(path, "rb").read()).hexdigest()[:16],
    }


def main():
    out = {
        "schema": "agent3_gate_summary_v1",
        "note": ("Aggregate only. The per-case records are keyed by dataset "
                 "image identifiers and carry per-image measured quantities; "
                 "they are withheld under the ReXGradient data-use agreement "
                 "(see PUBLISHING.md). No identifier appears in this file."),
        "batches": [s for s in (summarise(d, lab) for d, lab in BATCHES)
                    if s is not None],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.relpath(OUT, REPO))
    for b in out["batches"]:
        print("  %-28s %s cases, %s gate failures"
              % (b["label"], b["n_cases"], b["n_gate_failures"]))


if __name__ == "__main__":
    main()
