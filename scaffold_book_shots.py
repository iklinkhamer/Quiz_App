#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 23:07:48 2025

@author: Ilse Klinkhamer
"""

#!/usr/bin/env python3
import os, re
import pandas as pd
from pathlib import Path

CSV_PATH = "anatomy_physiology.mcqs.csv"   # set your path if different
ROOT = Path("book_shots")

CHAPTER_COLS = ["chapter_wiki","chapter","chapters","wiki_chapter","chapter_title","chapter_questions"]
SECTION_COLS = ["section_wiki","section","sections","subchapter","subchapters","subsection","subsections","wiki_section","section_questions"]

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+","_", str(s)).strip("_").lower()

def pick(df, cands):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in cols_lower: return cols_lower[c.lower()]
    return None

def split_multi(v):
    if pd.isna(v): return []
    t = str(v).strip()
    if not t or t.lower() in {"section not found","not found","none"}: return []
    parts = re.split(r"[;,]", t)
    return [p.strip() for p in parts if p.strip() and p.lower() not in {"section not found","not found"}]

df = pd.read_csv(CSV_PATH, engine="python")
ch_col = pick(df, CHAPTER_COLS)
sec_col = pick(df, SECTION_COLS)

if not ch_col or not sec_col:
    raise SystemExit("Couldn't find chapter/section columns in CSV.")

made = 0
for _, row in df.iterrows():
    chs = split_multi(row.get(ch_col, ""))
    secs = split_multi(row.get(sec_col, ""))
    for ch in chs:
        for sec in secs:
            path = ROOT / safe_name(ch) / safe_name(sec)
            path.mkdir(parents=True, exist_ok=True)
            marker = path / "README_place_your_images_here.txt"
            if not marker.exists():
                marker.write_text(f"Drop screenshots for '{ch}' / '{sec}' in this folder.\n")
                made += 1

print(f"Scaffolded {made} section folders under ./{ROOT}")
