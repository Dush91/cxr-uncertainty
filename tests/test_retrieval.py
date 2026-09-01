"""Smoke tests for the Agent-2 similar-case retrieval scripts.

Builds a tiny synthetic array set (N=300, D=32, M=4 members) in tmp_path,
runs build_retrieval_index.main() and query_retrieval via the module
functions, and checks: label-free store, self-exclusion, correct-mode
filter semantics, and HNSW == exact recall at this scale.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from build_retrieval_index import build_sidecar, exact_topk  # noqa: E402


def _make_arrays(arr_dir, source, n_cal=300, n_eval=40, dim=32, m=4, p_cls=10, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(arr_dir, source), exist_ok=True)
    for split, n in (("cal", n_cal), ("eval", n_eval)):
        feats = rng.normal(size=(n, dim)).astype(np.float32)
        probs = rng.uniform(0, 1, size=(n, m, p_cls))
        valid = np.ones((n, p_cls), dtype=bool)
        valid[:, 6:] = False  # like NIH valid mask
        gt = (probs.mean(1) > 0.5).astype(float) * rng.integers(0, 2, size=(n, p_cls))
        gt[:, 0] = 1.0
        gt[::7, 0] = -1.0  # uncertain rows
        ids = np.array([f"img_{split}_{i}" for i in range(n)], dtype=object)
        np.savez_compressed(
            os.path.join(arr_dir, source, f"arrays_{source}{split}.npz"),
            probs=probs, gt=gt, valid=valid, rad_feats=feats, ids=ids,
            member_keys=np.array([f"memb{j}" for j in range(m)], dtype=object),
            pathologies=np.array([f"cls{j}" for j in range(p_cls)], dtype=object),
        )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrieval")
    arr_dir = str(root / "eval_arrays")
    out = str(root / "retrieval" / "test_src_raddino")
    _make_arrays(arr_dir, "test_src")
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "build_retrieval_index.py"),
         "--arr-dir", arr_dir, "--source", "test_src", "--embed", "raddino",
         "--out", out, "--recall-queries", "20"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    return {"arr_dir": arr_dir, "out": out, "stdout": r.stdout}


def test_label_free_store(built):
    store = np.load(os.path.join(built["out"], "store.npz"), allow_pickle=True)
    assert set(store.files) == {"feats", "ids", "member_keys"}  # no gt/probs in store
    sc = pd.read_parquet(os.path.join(built["out"], "sidecar.parquet"))
    assert {"image_id", "pred_pathology", "pred_conf", "correct", "uncertain"} <= set(sc.columns)


def test_recall_validation(built):
    import json
    with open(os.path.join(built["out"], "index_meta.json")) as f:
        meta = json.load(f)
    assert meta["recall_at_k"] >= 0.95  # 300 vectors: HNSW must be exact enough


def test_query_modes_and_self_exclusion(built):
    qid = "img_eval_0"
    for mode in ("correct", "plain"):  # (pathology mode retired 2026-09-01)
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "query_retrieval.py"),
             "--index-dir", built["out"], "--arr-dir", built["arr_dir"],
             "--source", "test_src", "--id", qid, "--mode", mode, "--k", "5"],
            capture_output=True, text=True, cwd=REPO,
        )
        assert r.returncode == 0, r.stderr
        assert f"mode={mode}" in r.stdout
    out_lines = r.stdout.splitlines()
    assert "img_cal_" in r.stdout  # neighbors come from the cal reference library only
    ids_col = [ln.split()[1] for ln in out_lines if ln.strip() and ln.strip()[0].isdigit()]
    assert qid not in ids_col and len(ids_col) == 5  # self-exclusion + k respected


def test_sidecar_correctness_semantics():
    rng = np.random.default_rng(1)
    n = 60
    probs = rng.uniform(0, 1, size=(n, 2, 3))
    valid = np.ones((n, 3), dtype=bool)
    valid[:, 2] = False
    mean = probs.mean(1)
    gt = (mean > 0.5).astype(float)
    sc = build_sidecar({"probs": probs, "valid": valid, "gt": gt,
                        "ids": np.array([f"i{i}" for i in range(n)]),
                        "pathologies": np.array(["a", "b", "c"], dtype=object)})
    pred_idx = np.where(valid, mean, -np.inf).argmax(1)
    exp_correct = np.array([gt[k, pred_idx[k]] == 1.0 for k in range(n)])
    assert (sc["correct"].values == exp_correct).all()
    assert sc["uncertain"].sum() == 0  # all labels binary here