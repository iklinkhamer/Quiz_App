#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:52:12 2025

@author: Ilse Klinkhamer
"""

import pandas as pd
import re, json, sys
from pathlib import Path

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "anatomy_physiology_mcqs.csv"

def stem_from_csv(path: str) -> str:
    s = Path(path).stem
    m = re.search(r"(ch\d{2}_[A-Za-z0-9_]+)$", s)
    return m.group(1) if m else s

def pick_info_col(df):
    for c in ["info_key", "topic", "concept"]:
        if c in df.columns:
            return c
    return None

# naive keyword → info_key mapping you can extend
HEURISTICS = [
    (r"\bglomerul|filtrat", "renal_filtration"),
    (r"\breabsorb", "tubular_reabsorption"),
    (r"\bsecret(ion|e)", "tubular_secretion"),
    (r"\bcountercurrent|henle", "countercurrent_multiplier"),
    (r"\bADH|vasopressin|aquaporin", "adh_regulation"),
    (r"\brenin|RAAS|angiotensin|aldoster", "raas"),
]

def guess_key(text: str) -> str | None:
    t = text.lower()
    for pat, key in HEURISTICS:
        if re.search(pat, t):
            return key
    return None

def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    info_col = pick_info_col(df)
    q_col = "question" if "question" in df.columns else None
    if not q_col:
        raise SystemExit("CSV must contain a 'question' column")

    keys = set()
    if info_col:
        keys |= {str(x).strip() for x in df[info_col].dropna().astype(str) if str(x).strip()}
    else:
        # try to infer from question text
        for q in df[q_col].astype(str):
            k = guess_key(q)
            if k: keys.add(k)

    if not keys:
        print("No info keys found/inferred. Add an 'info_key' column or tweak HEURISTICS.")
        return

    # Build skeleton
    micro = {}
    for k in sorted(keys):
        title = k.replace("_"," ").title()
        micro[k] = {
            "title": f"{title} (Heads-up)",
            "markdown": f"- Core idea bullets for **{title}**\n- Keep this 3–5 lines\n- Focus on what’s needed to answer the upcoming questions",
            "image": "",
            "min_repeat_minutes": 10
        }

    out = {
        "title": stem_from_csv(CSV_PATH).replace("_", " ").title(),
        "reading": "",
        "summary_bullets": [],
        "estimated_read_time_sec": 300,
        "micro_notes": micro,
        "source_refs": []
    }

    out_path = Path(CSV_PATH).with_suffix(".reading.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
