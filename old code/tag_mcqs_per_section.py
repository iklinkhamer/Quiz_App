#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file tagger: outputs only chapter, section, confidence.
Created on Thu Oct 16 15:48:32 2025

@author: Ilse Klinkhamer
"""

import csv, json, re, math, argparse
from collections import Counter

# -------------------------------
# Minimal Porter stemmer
# -------------------------------
def _cons(word, i):
    c = word[i]
    if c in "aeiou": return False
    if c == 'y':
        return False if i == 0 else (not _cons(word, i-1))
    return True
def _m(word):
    n = 0; i = 0; L = len(word)
    while True:
        if i >= L: return n
        if not _cons(word, i): break
        i += 1
    i += 1
    while True:
        while True:
            if i >= L: return n
            if _cons(word, i): break
            i += 1
        i += 1; n += 1
        while True:
            if i >= L: return n
            if not _cons(word, i): break
            i += 1
        i += 1
def _vowel_in_stem(word): return any(not _cons(word, i) for i in range(len(word)))
def _doublec(word): return len(word) >= 2 and word[-1] == word[-2] and _cons(word, len(word)-1)
def _cvc(word):
    if len(word) < 3: return False
    return _cons(word,-3) and (not _cons(word,-2)) and _cons(word,-1) and word[-1] not in "wxy"
def porter_stem(t):
    w=t.lower()
    if len(w)<3: return w
    if w.endswith("sses"): w=w[:-2]
    elif w.endswith("ies"): w=w[:-2]
    elif w.endswith("ss"): pass
    elif w.endswith("s"): w=w[:-1]
    flag=False
    if w.endswith("eed"):
        base=w[:-3];  w = w[:-1] if _m(base)>0 else w
    elif w.endswith("ed") and _vowel_in_stem(w[:-2]): w=w[:-2]; flag=True
    elif w.endswith("ing") and _vowel_in_stem(w[:-3]): w=w[:-3]; flag=True
    if flag:
        if w.endswith(("at","bl","iz")): w+="e"
        elif _doublec(w) and w[-1] not in "lsz": w=w[:-1]
        elif _m(w)==1 and _cvc(w): w+="e"
    if _vowel_in_stem(w) and w.endswith("y"): w=w[:-1]+"i"
    return w

# -------------------------------
# Text utils
# -------------------------------
PUNCT_RE = re.compile(r"[^\w\+\-/²³⁺⁻α-ωΑ-Ωµ·\.]")
WS_RE = re.compile(r"\s+")
def normalize(text): return WS_RE.sub(" ", PUNCT_RE.sub(" ", (text or "").lower())).strip()
def tokenize(text): return [t for t in normalize(text).split(" ") if t]
def stem_tokens(tokens): return [porter_stem(t) for t in tokens]
def ngrams(tokens, n=2): return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
def slug(s): return re.sub(r"[^a-z0-9]+","_", normalize(s))[:80].strip("_")

# Domain regexes that should always boost
REGEX_FEATURES = [
    re.compile(r"\b(ad[ht]|acth|gh|fsh|lh|tsh)\b", re.I),
    re.compile(r"\b(t3|t4|thyroxine|insulin|glucagon|aldosterone|oxytocin|adh)\b", re.I),
    re.compile(r"\b(co2|o2|h2o|na\+|k\+|ca2\+|hco3-|po4|so4)\b", re.I),
    re.compile(r"\b(alveol\w+|bronch\w+|diaphragm|haemoglobin|hemoglobin)\b", re.I),
    re.compile(r"\b(lymph|lacteal|thymus|spleen|node)\b", re.I),
    re.compile(r"\b(glomerul\w+|tubul\w+|nephron)\b", re.I),
    re.compile(r"\b(binomial|genus|species|kingdom|phylum|order|family|taxonomy|vertebrate|invertebrate)\b", re.I),
]

# -------------------------------
# Build CHAPTER and SECTION indices
# -------------------------------
def add_text(acc_tokens, phrase_boost, s, w=1.0):
    if not s: return
    toks = stem_tokens(tokenize(s))
    acc_tokens.extend((t,w) for t in toks)
    for bg in ngrams(toks,2): phrase_boost[bg] += w*0.5
    for tg in ngrams(toks,3): phrase_boost[tg] += w*0.7  # light trigram weight

def build_indices(chapters, extra_synonyms=None):
    chapter_index = {}
    section_index = {}  # key -> {meta..., keywords, phrases}

    for ch in chapters:
        ch_title = ch["title"]
        ch_tokens, ch_phr = [], Counter()
        add_text(ch_tokens, ch_phr, ch.get("title",""), 2.0)
        add_text(ch_tokens, ch_phr, ch.get("reading",""), 1.5)

        # Chapter-wide synonyms
        for syn in (extra_synonyms or {}).get(ch_title, []):
            add_text(ch_tokens, ch_phr, syn, 3.0)

        # Sections from micro_notes
        for sec_key, sec in (ch.get("micro_notes") or {}).items():
            sec_title = sec.get("title","").strip() or sec_key
            sec_id = f"{slug(ch_title.split(' - ')[0])}::{slug(sec_title)}"
            sec_tokens, sec_phr = [], Counter()

            add_text(sec_tokens, sec_phr, ch_title, 1.2)
            add_text(sec_tokens, sec_phr, sec_title, 3.0)
            add_text(sec_tokens, sec_phr, sec.get("markdown",""), 1.6)

            # Fold into chapter signals
            add_text(ch_tokens, ch_phr, sec_title, 1.2)
            add_text(ch_tokens, ch_phr, sec.get("markdown",""), 1.1)

            section_index[sec_id] = {
                "chapter_title": ch_title,
                "section_title": sec_title,
                "keywords": Counter({t:w for t,w in sec_tokens}),
                "phrases": sec_phr
            }

        # Sections from mini_micronotes (optional; lighter)
        for mini_key, mini in (ch.get("mini_micronotes") or {}).items():
            sec_title = mini.get("title","").strip() or mini_key
            sec_id = f"{slug(ch_title.split(' - ')[0])}::{slug(sec_title)}"
            sec_tokens, sec_phr = [], Counter()
            add_text(sec_tokens, sec_phr, ch_title, 1.0)
            add_text(sec_tokens, sec_phr, sec_title, 2.2)
            add_text(sec_tokens, sec_phr, mini.get("markdown",""), 1.4)
            section_index[sec_id] = {
                "chapter_title": ch_title,
                "section_title": sec_title,
                "keywords": Counter({t:w for t,w in sec_tokens}),
                "phrases": sec_phr
            }
            add_text(ch_tokens, ch_phr, sec_title, 1.0)
            add_text(ch_tokens, ch_phr, mini.get("markdown",""), 1.0)

        chapter_index[ch_title] = {
            "keywords": Counter({t:w for t,w in ch_tokens}),
            "phrases": ch_phr
        }

    return chapter_index, section_index

# -------------------------------
# Scoring
# -------------------------------
def score_text(qtext, topic):
    toks = stem_tokens(tokenize(qtext))
    bg = ngrams(toks,2) + ngrams(toks,3)
    kw, ph = topic["keywords"], topic["phrases"]
    tok_score = sum(kw.get(t,0.0) for t in toks)
    phrase_score = sum(ph.get(p,0.0) for p in bg)
    regex_bonus = sum(1.0 for rx in REGEX_FEATURES if rx.search(qtext))
    raw = tok_score + phrase_score + regex_bonus
    denom = math.log(10 + len(toks))
    return raw / denom

def rank_targets(qtext, index_dict):
    scores = []
    for key, topic in index_dict.items():
        s = score_text(qtext, topic)
        scores.append((key, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def tanh_norm(x): return math.tanh(x)

# -------------------------------
# Main
# -------------------------------
def main(chapters_path, mcq_csv_path, out_csv_path,
         min_ch_conf=0.25, min_sec_conf=0.30):

    chapters = json.load(open(chapters_path, "r", encoding="utf-8"))

    extra_synonyms = {
        "Chapter 16 - Endocrine System": ["pituitary","hypothalamus","thyroid","parathyroid","adrenal","pineal","pancreas","ovaries","testes","hormone","adh","oxytocin","prolactin","fsh","lh","gh","melatonin","thyroxine","aldosterone","insulin","testosterone","oestrogen","progesterone"],
        "Chapter 9 - Respiratory System": ["alveoli","bronchi","haemoglobin","hemoglobin","oxygen","carbon dioxide","ventilation","inspiration","expiration","diaphragm","acid-base"],
        "Chapter 10 - Lymphatic System": ["lymph","lymphatic","lacteal","lymph node","spleen","thymus","tissue fluid"],
        "Chapter 1 - Chemicals": ["ion","electrolyte","carbohydrate","lipid","protein","monosaccharide","disaccharide","polysaccharide","triglyceride","phospholipid","amino acid"],
        "Chapter 2 - Classification": ["binomial","genus","species","taxonomy","vertebrate","invertebrate","kingdom","phylum","class","order","family","primates","rodentia","carnivora","marsupial","monotreme","placental"]
    }

    ch_index, sec_index = build_indices(chapters, extra_synonyms)

    # Read MCQs
    with open(mcq_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Prepare minimal output rows
    output_rows = []

    for row in rows:
        # Build text corpus from the MCQ row
        qtext = " ".join([
            row.get("question",""),
            row.get("answerA",""), row.get("answerB",""),
            row.get("answerC",""), row.get("answerD","")
        ])

        # Chapter-level best
        ch_scores = rank_targets(qtext, ch_index) if ch_index else []
        if ch_scores:
            best_ch_key, ch_s = ch_scores[0]
            ch_conf = tanh_norm(ch_s)
            best_chapter = best_ch_key if ch_conf >= min_ch_conf else ""
        else:
            best_chapter, ch_conf = "", 0.0

        # Section-level best
        sec_scores = rank_targets(qtext, sec_index) if sec_index else []
        if sec_scores:
            best_sec_key, sec_s = sec_scores[0]
            sec_meta = sec_index[best_sec_key]
            sec_conf = tanh_norm(sec_s)
            best_section_title = sec_meta["section_title"] if sec_conf >= min_sec_conf else ""
            best_section_chapter = sec_meta["chapter_title"]
        else:
            best_sec_key, sec_conf = "", 0.0
            best_section_title, best_section_chapter = "", ""

        # Guardrail: prefer consistent section if we have a confident chapter
        if best_chapter and best_section_title:
            if best_section_chapter != best_chapter and ch_conf >= 0.50:
                # conflict → drop section
                best_section_title = ""
                sec_conf = 0.0

        # Placeholders & single confidence rule
        final_chapter = best_chapter if best_chapter else "Chapter not found"
        final_section = best_section_title if best_section_title else "Section not found"
        final_conf = sec_conf if best_section_title else (ch_conf if best_chapter else 0.0)

        output_rows.append({
            "chapter": final_chapter,
            "section": final_section,
            "confidence": f"{final_conf:.3f}"
        })

    # Write the single combined CSV (only three columns)
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["chapter", "section", "confidence"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(output_rows)

    print(f"Tagged MCQs → {out_csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", required=True, help="Path to chapters JSON (array).")
    ap.add_argument("--mcqs", required=True, help="Path to MCQ CSV.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--min_ch_conf", type=float, default=0.25)
    ap.add_argument("--min_sec_conf", type=float, default=0.30)
    args = ap.parse_args()
    main(args.chapters, args.mcqs, args.out, args.min_ch_conf, args.min_sec_conf)



"""
Run this in bash 

python tag_mcqs_per_section.py \
  --chapters anatomy_physiology_mcqs.reading.json \
  --mcqs anatomy_physiology_mcqs.csv \
  --out anatomy_physiology_mcqs_tagged_2.csv
  
"""