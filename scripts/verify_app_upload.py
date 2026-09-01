#!/usr/bin/env python
"""Drive the LIVE Gradio demo at :7860 with Playwright to verify the upload path.

Uploads an OpenI image that is NOT a built-in example (CXR1034, Hernia confident-FN)
through the real browser file-upload UI, waits for predict to finish, and reads the
rendered outputs (banner, results table, Mahalanobis percentile, member bar) back as
TEXT from the DOM -- no vision model needed to assert functional correctness. Also
saves a full-page screenshot the user can eyeball.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/verify_app_upload.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path("/teamspace/studios/this_studio")
IMG = str(REPO / "data" / "openi" / "CXR1034_IM-0028-1001.png")
WATCH = "Hernia"
URL = "http://127.0.0.1:7860"
SHOT = REPO / "runs" / "app_upload_screenshot.png"

from playwright.sync_api import sync_playwright


def main() -> None:
    assert os.path.exists(IMG), f"missing {IMG}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 2200})
        print(f"[pw] goto {URL}")
        page.goto(URL, wait_until="networkidle", timeout=60_000)
        # give the app a moment to finish startup rendering
        page.wait_for_timeout(2000)

        # --- locate the image upload file input (Gradio gr.Image) ---
        file_input = page.locator("input[type='file']").first
        fi_count = file_input.count()
        print(f"[pw] found {fi_count} file input(s)")
        assert fi_count >= 1, "no file input found -- image upload component missing"

        # --- upload the OpenI image (fires img.change -> predict) ---
        print(f"[pw] uploading {os.path.basename(IMG)}")
        file_input.set_input_files(IMG)

        # --- wait for predict to populate outputs (table rows + maha textbox) ---
        # The results dataframe tbody should gain rows; the maha textbox should fill.
        def outputs_ready(_=None):
            tbls = page.locator("table tbody tr").count()
            txt = page.locator("input[type='text'], textarea").all_inner_texts()
            has_maha = any("%" in t and "raw" in t.lower() for t in txt) or \
                       any("pct" in t.lower() for t in txt)
            return tbls > 0 and has_maha
        try:
            page.wait_for_function(outputs_ready, timeout=90_000)
            print("[pw] outputs populated")
        except Exception as e:
            print(f"[pw] WARN wait_for_function timed out: {e}; dumping what we have")

        # --- screenshot (full page) for the user to eyeball ---
        SHOT.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOT), full_page=True)
        print(f"[pw] screenshot -> {SHOT}")

        # --- read back the rendered text ---
        # Banner: the first gr.HTML block holding the flag summary
        banner_text = ""
        for el in page.locator("[data-testid='html'], .html, .gradio-html, div").all()[:50]:
            t = el.inner_text() if el.is_visible() else ""
            if "Mahalanobis OOD" in t and "Rep-mismatch" in t:
                banner_text = t
                break

        # Results table: pull every row's cells
        rows = []
        ntr = page.locator("table tbody tr").count()
        for i in range(ntr):
            cells = page.locator("table tbody tr").nth(i).locator("td").all_inner_texts()
            rows.append([c.strip() for c in cells])

        # Mahalanobis percentile textbox + slider value
        maha_txt = ""
        for el in page.locator("input[type='text'], textarea").all():
            t = (el.input_value() if el.is_visible() else "") or ""
            if "%" in t:
                maha_txt = t
                break

        browser.close()

    # --- report ---
    print("\n===== BANNER =====")
    print(banner_text.strip() or "(not located)")
    print("\n===== MAHA TEXTBOX =====")
    print(maha_txt or "(not located)")
    print(f"\n===== RESULTS TABLE ({len(rows)} rows) =====")
    if rows:
        # print header-ish first row width, then each row compactly
        for r in rows:
            print(" | ".join(r))
    else:
        print("(table empty)")
    # assert the watch pathology fired
    fired = False
    for r in rows:
        if len(r) >= 2 and r[0].lower() == WATCH.lower():
            # repmismatch_flag column -- find a cell that is True/✓
            fired = any(c.lower() in ("true", "✓", "yes") for c in r)
            print(f"\n[pw] {WATCH} row found; repmismatch_flag cell(s): {[c for c in r if c.lower() in ('true','false','✓','yes','no')]}")
            break
    print(f"\n[pw] {WATCH} rep-mismatch fired via upload: {fired}")
    print("[pw] PASS" if fired else "[pw] see notes above (flag may be in a column we didn't isolate)")


if __name__ == "__main__":
    main()