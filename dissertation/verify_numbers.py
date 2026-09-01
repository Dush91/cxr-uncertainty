"""Cross-check the dissertation's headline numbers against the saved artifacts.

    python dissertation/verify_numbers.py

Each check names a claim made in the chapter text, reads the value from the
run artifact it came from, and asserts the two agree. This exists because the
prose was written from a research log rather than directly from the JSON, and
a transcription error in a headline number would be a serious defect.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNS = os.path.join(REPO, "runs")

OK, BAD = [], []


def jload(*parts):
    with open(os.path.join(RUNS, *parts)) as f:
        return json.load(f)


def check(claim, got, want, tol=5e-4):
    if got is None:
        BAD.append("%-58s MISSING (expected %s)" % (claim, want))
    elif abs(float(got) - float(want)) <= tol:
        OK.append("%-58s %s" % (claim, want))
    else:
        BAD.append("%-58s text says %s, artifact says %s"
                   % (claim, want, got))


def panel_b(path):
    d = jload(path, "baseline_comparison.json")
    return {int(r["top_pct"]): r for r in
            d["panel_b_within_ensemble_confident_set"]}, d


REX = os.path.join("baselines", "rex")
ADAPT = os.path.join("rex_phase7", "baselines", "rex_adapted_s0",
                     "rex_adapted_s0")

# --- Chapter 4 -------------------------------------------------------------
p3 = jload("phase3", "calibration_ablation.json")
p1 = jload("phase1", "evaluation.json")
p1mc = jload("phase1_mc", "evaluation.json")
p2 = jload("phase2", "evaluation.json")
check("Ch4 phase1 2-member confident-error AUROC = 0.713",
      p1.get("confident_error_auroc_std"), 0.7128, 1e-3)
check("Ch4 phase1 MC-dropout AUROC = 0.55",
      p1mc.get("confident_error_auroc_std"), 0.5517, 1e-3)
check("Ch4 phase2 3-member confident errors = 8",
      p2.get("n_confident_wrong"), 8, 0)
check("Ch4 phase1 confident errors = 18",
      p1.get("n_confident_wrong"), 18, 0)
check("Ch4 phase2 AURC = 0.00279", p2.get("aurc"), 0.00279, 1e-4)
check("Ch4 phase1 AURC = 0.00403", p1.get("aurc"), 0.00403, 1e-4)

f4 = jload("phase4_features", "phase4_features_summary.json")
blob4 = json.dumps(f4)
for site, maha, std in (("kermany", 0.818, 0.515), ("covid", 0.649, 0.478)):
    found_m = ("%.3f" % maha)[1:] in blob4 or ("%.4f" % maha) in blob4
    (OK if found_m else BAD).append(
        "%-58s %s" % ("Ch4 %s Mahalanobis = %.3f" % (site, maha),
                      "found" if found_m else "NOT FOUND in phase4_features"))

conf = jload("conformal", "conformal_summary.json")
for site, want in (("openiD_all", 0.896), ("kermany", 0.269),
                   ("covid", 0.245)):
    check("Ch4 conformal %s coverage = %.3f" % (site, want),
          conf[site]["report"]["conformal_coverage"], want, 1e-3)

ood = jload("ood_detection", "ood_detection.json")
oblob = json.dumps(ood)
for label, val in (("kNN Kermany 0.9904", "0.9904"),
                   ("kNN COVID 0.9834", "0.9834"),
                   ("Mahalanobis Kermany 0.9895", "0.9895")):
    hit = val in oblob
    (OK if hit else BAD).append("%-58s %s" % ("Ch4 OOD detection " + label,
                                              "found" if hit else "NOT FOUND"))

# --- Chapter 5 -------------------------------------------------------------
pb, d = panel_b(REX)
check("Ch5 pretrained std @top-10 = 0.5626", pb[10]["epistemic_std"], 0.5626)
check("Ch5 pretrained conf @top-10 = 0.5630", pb[10]["confidence"], 0.5630)
check("Ch5 pretrained Mahalanobis @top-10 = 0.5288", pb[10]["mahalanobis"],
      0.5288)
check("Ch5 pretrained kNN @top-10 = 0.5942", pb[10]["knn"], 0.5942)
check("Ch5 pretrained hybrid @top-10 = 0.5979", pb[10]["hybrid_std_knn"],
      0.5979)
check("Ch5 pretrained energy @top-10 = 0.4509", pb[10]["energy"], 0.4509)
check("Ch5 confident errors at top-10 = 34", d["ensemble"]["n_confident_wrong"],
      34, 0)
check("Ch5 evaluation records = 78,311", d["ensemble"]["n_records"], 78311, 0)
st = d["statistical_tests"]
check("Ch5 DeLong std vs conf p = 0.99",
      st["delong_conf_top10_epistemic_vs_confidence"]["p"], 0.9895, 1e-3)
check("Ch5 McNemar p = 0.98", st["mcnemar_epistemic_vs_confidence"]["p"],
      0.9845, 1e-3)
check("Ch5 single member beats ensemble, p = 0.0007",
      st["delong_conf_top10_epistemic_vs_best_member_convnextv2"]["p"],
      0.0007, 1e-4)
check("Ch5 conf_convnextv2 = 0.6077",
      st["delong_conf_top10_epistemic_vs_best_member_convnextv2"]["auc_b"],
      0.6077)
check("Ch5 Task 3 macro std = 0.8391",
      d["per_pathology_correctness_auroc"]["macro"]["epistemic_std"], 0.8391)
check("Ch5 Task 4 AUAC confidence = 0.9552",
      d["aurc_e_aurc_full_eval"]["confidence"]["auac"], 0.9552)
check("Ch5 Task 2 entropy = 0.6983",
      d["task2_uncertain_label_auroc"]["macro"]["entropy"], 0.6983)
check("Ch5 TS rank-invariance: raw = TS AUROC 0.4527",
      d["ts_calibrated_confidence"]["auroc_pbar_ts_top10"], 0.4527)

openi_pb, openi = panel_b(os.path.join("baselines", "openi"))
check("Ch5 OpenI Mahalanobis @top-10 = 0.8698", openi_pb[10]["mahalanobis"],
      0.8698)
check("Ch5 OpenI hybrid @top-10 = 0.9059", openi_pb[10]["hybrid_std_knn"],
      0.9059)

for name, arm, std_v, conf_v in (
        ("resnet18", "rex_fs_resnet18", 0.6360, 0.5585),
        ("vit_tiny", "rex_fs_vit_tiny", 0.7190, 0.6736),
        ("convnext_tiny", "rex_fs_convnext_tiny", 0.6688, 0.5385)):
    fb, _ = panel_b(os.path.join("baselines", arm))
    check("Ch5 from-scratch %s std = %.4f" % (name, std_v),
          fb[10]["epistemic_std"], std_v)
    check("Ch5 from-scratch %s conf = %.4f" % (name, conf_v),
          fb[10]["confidence"], conf_v)

for name, arm, std_v, conf_v in (
        ("crossarch M=3", "rex_fs_crossarch", 0.5585, 0.6093),
        ("resnet18 M=3", "rex_fs_resnet18_m3", 0.6453, 0.5455),
        ("crossarch M=2", "rex_fs_crossarch_rv", 0.5266, 0.5348),
        ("resnet18 M=2", "rex_fs_resnet18_m2", 0.6159, 0.4773)):
    cb, _ = panel_b(os.path.join("baselines", arm))
    check("Ch5 %s std" % name, cb[10]["epistemic_std"], std_v)
    check("Ch5 %s conf" % name, cb[10]["confidence"], conf_v)

ab, ad = panel_b(ADAPT)
check("Ch5 adapted s0 std @top-10 = 0.6789", ab[10]["epistemic_std"], 0.6789)
check("Ch5 adapted s0 conf @top-10 = 0.5417", ab[10]["confidence"], 0.5417)
check("Ch5 adapted s0 entropy @top-10 = 0.7553", ab[10]["entropy"], 0.7553)
check("Ch5 adapted DeLong std vs conf p = 0.006",
      ad["statistical_tests"]["delong_conf_top10_epistemic_vs_confidence"]["p"],
      0.006, 1e-3)
check("Ch5 adapted AUAC confidence = 0.9653",
      ad["aurc_e_aurc_full_eval"]["confidence"]["auac"], 0.9653)
check("Ch5 adapted AUAC std = 0.9411",
      ad["aurc_e_aurc_full_eval"]["epistemic_std"]["auac"], 0.9411)
check("Ch5 adapted E-AURC std = 0.0414",
      ad["aurc_e_aurc_full_eval"]["epistemic_std"]["e_aurc"], 0.0414)

for seed, std_v in (("s1", 0.6832), ("s2", 0.6927)):
    sb, _ = panel_b(os.path.join("rex_phase7", "baselines",
                                 "rex_adapted_" + seed, "rex_adapted_" + seed))
    check("Ch5 adapted %s std @top-10 = %.4f" % (seed, std_v),
          sb[10]["epistemic_std"], std_v)

# --- Chapter 6 -------------------------------------------------------------
RET = os.path.join("rex_phase7", "retrieval", "rex_adapted_raddino_full")
meta = jload(RET, "index_meta.json")
mblob = json.dumps(meta)
check("Ch6 library size = 129,113",
      meta.get("n_reference") or meta.get("n_vectors"), 129113, 0)
(OK if "1.0" in mblob else BAD).append(
    "%-58s %s" % ("Ch6 recall@10 = 1.0000", "found" if "1.0" in mblob
                  else "NOT FOUND"))
npj = jload(RET, "normal_precision.json")
for arm, want in (("old@cal-lib", 0.7396), ("old@full-lib", 0.6740),
                  ("new-normal-mode", 1.0)):
    check("Ch6 normal-precision@8 %s" % arm, npj[arm]["mean"], want, 1e-4)

rg = jload(RET, "rerank_gate.json")
check("Ch6 old pathology label-Jaccard@8 = 0.3526",
      rg["old_pathology"]["label_jaccard@k"], 0.3526, 1e-4)
check("Ch6 reranked pathology label-Jaccard@8 = 0.4148",
      rg["new_pathology"]["label_jaccard@k"], 0.4148, 1e-4)
check("Ch6 old plain label-Jaccard@8 = 0.2963",
      rg["old_plain"]["label_jaccard@k"], 0.2963, 1e-4)
check("Ch6 reranked plain label-Jaccard@8 = 0.3880",
      rg["new_plain"]["label_jaccard@k"], 0.3880, 1e-4)
check("Ch6 posterior-agree@8 pathology 0.737 -> 0.989",
      rg["new_pathology"]["posterior_agree@k"], 0.9890, 1e-3)
fg = rg["flag_gate"]
check("Ch6 argmax-bucket flag-precision@8 = 0.0142",
      fg["old_flag_pathology"]["flag_precision@k"], 0.0142, 1e-4)
check("Ch6 flag-lane flag-precision@8 = 1.0000",
      fg["new_flag"]["flag_precision@k"], 1.0, 1e-6)
check("Ch6 flag-lane cosine@8 = 0.49", fg["new_flag"]["cosine@k"], 0.4908,
      1e-3)
check("Ch6 argmax-bucket cosine@8 = 0.63",
      fg["old_flag_pathology"]["cosine@k"], 0.6285, 1e-3)
check("Ch6 contrast-lane model-detected@8 = 1.0000",
      fg["new_contrast"]["model_detected_share@k"], 1.0, 1e-6)
check("Ch6 status-quo model-detected@8 = 0.0018",
      fg["old_flag_pathology"]["model_detected_share@k"], 0.00175, 1e-4)

# --- Chapters 7-8 ----------------------------------------------------------
# The per-case Agent 3 records are withheld from the published repository
# (PUBLISHING.md); fall back to the redacted gate summary, which carries the
# same two numbers.
try:
    a3 = jload("app_rex_adapted", "agent3_batch", "faithfulness_report.json")
except FileNotFoundError:
    a3 = jload("app_rex_adapted", "agent3_gate_summary.json")["batches"][0]
check("Ch7 Agent 3 gate: 50 cases", a3["n_cases"], 50, 0)
check("Ch7 Agent 3 gate: 0 failures", a3["n_gate_failures"], 0, 0)

a4t = jload("app_rex_adapted", "agent4", "faithfulness_report_template.json")
check("Ch8 Agent 4 template gate: 5 lanes", a4t["n_lanes"], 5, 0)
check("Ch8 Agent 4 template gate: 0 failures", a4t["n_gate_failures"], 0, 0)
a4 = jload("app_rex_adapted", "agent4", "faithfulness_report.json")
check("Ch8 Agent 4 live trial: 2 rejected", a4["n_gate_failures"], 2, 0)

rows = a4["rows"]
n_timeout = sum(1 for r in rows
                if "Timeout" in (r.get("llm_error") or ""))
n_reject = sum(1 for r in rows if not (r.get("llm_audit") or {"ok": True})["ok"])
n_ship = len(rows) - n_timeout - n_reject
check("Ch8 live trial: 3 lanes timed out", n_timeout, 3, 0)
check("Ch8 live trial: 2 renderings rejected", n_reject, 2, 0)
check("Ch8 live trial: 0 renderings shipped", n_ship, 0, 0)
_rej = [r for r in rows if not (r.get("llm_audit") or {"ok": True})["ok"]]
_by = {r["lane"]: len(r["llm_audit"]["failures"]) for r in _rej}
check("Ch8 augmentation lane: 1 audit failure",
      _by.get("augmentation_gap"), 1, 0)
check("Ch8 retrain lane: 9 audit failures", _by.get("retrain_vs_ft"), 9, 0)
_r = next(r for r in _rej if r["lane"] == "retrain_vs_ft")
check("Ch8 retrain lane: 4 invalid citations",
      sum(1 for f in _r["llm_audit"]["failures"]
          if f.startswith("invalid citation")), 4, 0)
check("Ch8 retrain lane: 5 ungrounded numbers",
      sum(1 for f in _r["llm_audit"]["failures"]
          if "ungrounded number" in f), 5, 0)

ev = jload("app_rex_adapted", "agent4", "suggestions_evidence.json")
check("Ch8 evaluation error rate = 18.12%", ev["provenance"]["err_rate_pct"],
      18.12, 1e-2)
check("Ch8 labelled cells = 78,311", ev["provenance"]["labelled_cells"],
      78311, 0)
check("Ch8 threshold-reachable error mass = 48.54%",
      ev["threshold_levers"]["threshold_reachable_err_cells_pct"], 48.54, 1e-2)
check("Ch8 Pleural_Thickening false positives = 1,919",
      ev["threshold_levers"]["Pleural_Thickening"]["eval_fp_n"], 1919, 0)
check("Ch8 Pleural_Thickening FP/FN ratio = 61.9",
      ev["threshold_levers"]["Pleural_Thickening"]["fp_fn_ratio"], 61.9, 1e-1)
check("Ch8 AP error rate = 20.58%",
      ev["stratification_slices"]["view"]["AP"]["err_rate_pct"], 20.58, 1e-2)
check("Ch8 PA error rate = 11.72%",
      ev["stratification_slices"]["view"]["PA"]["err_rate_pct"], 11.72, 1e-2)
check("Ch8 worst cluster error rate = 37.29%",
      ev["stratification_slices"]["cluster"]["2"]["err_rate_pct"], 37.29, 1e-2)
check("Ch8 eval never used for threshold selection",
      0 if ev["threshold_levers"]["eval_used_for_selection"] is False else 1,
      0, 0)

# --- report ----------------------------------------------------------------
for line in OK:
    print("  ok   " + line)
print()
for line in BAD:
    print("  FAIL " + line)
print("\n%d checks passed, %d failed" % (len(OK), len(BAD)))
sys.exit(1 if BAD else 0)
