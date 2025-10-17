#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:12:34 2025

@author: Ilse Klinkhamer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuizzyBee by Ilse Klinkhamer
"""

import streamlit as st
import pandas as pd
import random
import json
import time
import re
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------- Page Setup ----------
st.set_page_config(page_title="🧠 QuizzyBee", layout="centered")

# ---------- Configuration ----------
SECTIONS_IMG_ROOT = "book_shots"  # folder with screenshots: book_shots/<chapter>/<section>/*.(png|jpg|jpeg|gif)

REQUIRED_HEADERS = {
    "question",
    # any of these for options
    "optionA", "answerA", "A",
    "optionB", "answerB", "B",
    "optionC", "answerC", "C",
    "optionD", "answerD", "D",
    # any of these for correct
    "answer", "correct", "correct answer", "Correct Answer",
}

# Candidates for chapter/section columns in your CSV
CHAPTER_COL_CANDIDATES = ["chapter", "chapters", "Chapter", "wiki_chapter", "chapter_title"]
SECTION_COL_CANDIDATES = ["section", "sections", "subchapter", "subchapters", "subsection", "subsections", "wiki_section"]

PROFILES_FILE = "quiz_profiles.json"
SETTINGS_FILE = "quiz_settings.json"
NO_PROFILE_LABEL = "No profile (don't save)"

# Pre-reading modes
PRE_READ_NONE = "Skip"
PRE_READ_SUMMARY = "Summary only"
PRE_READ_FULL = "Full reading"
PRE_READ_MODES = [PRE_READ_NONE, PRE_READ_SUMMARY, PRE_READ_FULL]

# Rotation & Spaced Practice Config
ACTIVE_POOL_SIZE = 10
GRADUATE_STREAK = 3
BASE_INTERVAL_MIN = 1
INTERVAL_GROWTH_FACTOR = 2.0

LETTERS = ["A", "B", "C", "D"]

# Exclude questions that lack chapter/section when filters are used
EXCLUDE_UNTAGGED_WHEN_FILTERING = True

# (Optional) If you want to exclude untagged ALWAYS (even with no filters), set this to True
EXCLUDE_UNTAGGED_ALWAYS = False



# ---------- Small helpers ----------
def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s)).strip("_").lower() if s is not None else ""


def list_question_sets(search_root="."):
    """Find CSVs recursively that look like MCQ sets (by header)."""
    candidates = glob.glob(os.path.join(search_root, "**/*.csv"), recursive=True)
    valid = []
    for path in candidates:
        try:
            df_head = pd.read_csv(path, nrows=0, engine="python")
            cols = set([c.strip() for c in df_head.columns])
            if any(c.lower() == "question" for c in cols) and len(cols & REQUIRED_HEADERS) >= 5:
                valid.append(path)
        except Exception:
            continue
    valid = sorted(valid, key=lambda p: (p.count(os.sep), p.lower()))
    return valid


def progress_path_for(csv_file: str, profile_name: Optional[str]):
    if not profile_name or profile_name == NO_PROFILE_LABEL:
        return None
    base = os.path.splitext(os.path.basename(csv_file))[0]
    safe_ds = safe_name(base)
    safe_prof = safe_name(profile_name)
    return f"quiz_progress__{safe_ds}__{safe_prof}.json"


# ---------- Profiles store ----------
def load_profiles() -> list[str]:
    try:
        with open(PROFILES_FILE) as f:
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
    except FileNotFoundError:
        save_profiles(["Ilse"])
        return ["Ilse"]
    except Exception:
        pass
    return ["Ilse"]


def save_profiles(names: list[str]):
    try:
        with open(PROFILES_FILE, "w") as f:
            json.dump(sorted(set(names), key=str.lower), f, indent=2)
    except Exception as e:
        st.warning(f"Couldn't save profiles: {e}")


def add_profile(name: str):
    name = name.strip()
    if not name:
        return
    profiles = load_profiles()
    if name not in profiles:
        profiles.append(name)
        save_profiles(profiles)


# ---------- App-wide settings ----------
def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return {}


def save_settings(last_profile: str, last_num_questions: int, pre_read_mode: str,
                  last_chapter: Optional[str], last_section: Optional[str]):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(
                {
                    "last_profile": last_profile,
                    "last_num_questions": int(last_num_questions),
                    "pre_read_mode": pre_read_mode,
                    "last_chapter": last_chapter,
                    "last_section": last_section,
                },
                f, indent=2
            )
    except Exception as e:
        st.warning(f"Couldn't save settings: {e}")


# ---------- App State Defaults ----------
def init_state():
    settings = load_settings()
    defaults = {
        "screen": "start",              # "start" | "reading" | "quiz" | "summary"
        "num_questions": settings.get("last_num_questions", 20),
        "questions_answered": 0,
        "correct_count": 0,
        "session_started_at": None,
        "quiz_data": None,
        "current_question_idx": None,
        # submission/feedback gating
        "awaiting_continue": False,
        "last_was_correct": None,
        "last_feedback": "",
        # datasets
        "available_sets": [],
        "selected_csv": None,
        "last_loaded_csv": None,
        # profiles
        "profiles": [],
        "selected_profile": settings.get("last_profile", NO_PROFILE_LABEL),
        "last_loaded_profile": None,
        "new_profile_name": "",
        # reset confirmation state
        "confirm_reset": False,
        # pre-reading
        "pre_read_mode": settings.get("pre_read_mode", PRE_READ_NONE),
        "reading_payload": None,        # JSON reading fallback
        "reading_stem": None,
        # taxonomy from CSV
        "chapter_col": None,
        "section_col": None,
        "chapters_list": [],
        "sections_by_chapter": {},      # {chapter: sorted list of sections}
        "selected_chapter": settings.get("last_chapter"),   # can be None or chapter string
        "selected_section": settings.get("last_section"),   # can be None or section string
        # raw df cache per file to avoid re-reading
        "df_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

st.title("🐝 QuizzyBee")
st.caption("Filter by chapter & section • See screenshots • Then drill only those questions")


# ---------- CSV helpers ----------
def _pick_first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in df.columns:
        normalized = c.strip()
        if normalized in candidates:
            return normalized
    # loose match ignoring case/spaces
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _split_multi(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    # split by comma or semicolon
    parts = re.split(r"[;,]", text)
    return [p.strip() for p in parts if p.strip()]


@st.cache_data
def read_df(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, sep=",", engine="python")


def detect_taxonomy(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], list[str], dict]:
    """Return (chapter_col, section_col, chapters_list, sections_by_chapter)."""
    chapter_col = _pick_first_present(df, CHAPTER_COL_CANDIDATES)
    section_col = _pick_first_present(df, SECTION_COL_CANDIDATES)

    chapters_list: list[str] = []
    sections_by_chapter: dict[str, list[str]] = {}

    if chapter_col:
        # collect unique chapters
        all_chapters = []
        for v in df[chapter_col].fillna(""):
            all_chapters.extend(_split_multi(v))
        chapters_list = sorted(sorted(set(all_chapters)), key=lambda s: s.lower())

    if chapter_col and section_col:
        # map chapter -> set(sections) from rows
        temp = {c: set() for c in chapters_list} if chapters_list else {}
        for _, row in df.iterrows():
            chs = _split_multi(row.get(chapter_col, ""))
            secs = _split_multi(row.get(section_col, ""))
            if not chs or not secs:
                continue
            for ch in chs:
                temp.setdefault(ch, set()).update(secs)
        sections_by_chapter = {ch: sorted(sorted(list(s)), key=lambda x: x.lower()) for ch, s in temp.items()}

    return chapter_col, section_col, chapters_list, sections_by_chapter


# ---------- Reading JSON fallback (your earlier flow) ----------
def extract_chapter_stem_from_csv(csv_path: str) -> Optional[str]:
    name = Path(csv_path).stem
    m = re.search(r"(ch\d{2}_[A-Za-z0-9_]+)$", name)
    if m:
        return m.group(1)
    m = re.search(r"(ch\d{2}_.+)", name)
    return m.group(1) if m else None


def find_reading_json_for_csv(csv_path: str) -> Optional[str]:
    p = Path(csv_path)
    stem = extract_chapter_stem_from_csv(csv_path)
    if not stem:
        return None
    candidates = [
        p.with_name(f"{stem}.reading.json"),
        p.with_name(f"{stem}.json"),
        p.with_name(stem) / "reading.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def load_reading_payload(csv_path: str):
    """
    Return (payload, support, stem)
    payload keys: title, reading, summary_bullets, estimated_read_time_sec
    support: "both" | "summary_only" | None
    """
    stem = extract_chapter_stem_from_csv(csv_path)
    json_path = find_reading_json_for_csv(csv_path)
    if not json_path:
        return None, None, stem
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        title = data.get("title") or stem.replace("_", " ")
        reading = (data.get("reading") or "").strip()
        summary_bullets = data.get("summary_bullets") or []
        payload = {
            "title": title,
            "reading": reading,
            "summary_bullets": summary_bullets,
            "estimated_read_time_sec": int(data.get("estimated_read_time_sec") or 60),
            "source_refs": data.get("source_refs", []),
        }
        if reading and summary_bullets:
            support = "both"
        elif summary_bullets:
            support = "summary_only"
        else:
            support = None
        return payload, support, stem
    except Exception as e:
        st.warning(f"Couldn't load reading JSON: {e}")
        return None, None, stem


# ---------- Load questions (now with chapter/section filter) ----------
def normalize_question_fields(q: dict):
    q.setdefault("status", "unseen")
    q.setdefault("streak", 0)
    q.setdefault("seen_count", 0)
    q.setdefault("interval", 1)
    q.setdefault("next_time", datetime.now().isoformat())
    return q


def _col_try(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    # case-insensitive fallback
    for n in names:
        for c in df.columns:
            if c.strip().lower() == n.strip().lower():
                return c
    return None


def _row_matches_filters(row, chapter_col, section_col, sel_chapter, sel_section) -> bool:
    """
    Return True if this row should be included given user's selections.
    Behavior:
      • If chapter/section filters are in use, rows missing either chapter OR section are excluded.
      • If no filters are in use, all rows are included (unless EXCLUDE_UNTAGGED_ALWAYS=True).
    """
    # Gather tags (as sets) if columns exist
    chs = set(_split_multi(row.get(chapter_col, ""))) if chapter_col else set()
    secs = set(_split_multi(row.get(section_col, ""))) if section_col else set()

    using_filters = bool(sel_chapter or sel_section)

    # Exclude untagged if filters are in use (so incomplete rows don’t sneak in)
    if using_filters and EXCLUDE_UNTAGGED_WHEN_FILTERING:
        if not chs or not secs:
            return False

    # Optional: exclude untagged globally, even without filters
    if not using_filters and EXCLUDE_UNTAGGED_ALWAYS:
        if not chs or not secs:
            return False

    # No filters at all → include (subject to optional global exclusion above)
    if not using_filters:
        return True

    # Only section selected (no chapter): include if section matches anywhere
    if sel_section and not sel_chapter:
        return sel_section in secs

    # Chapter selected (with or without section)
    if sel_chapter:
        if sel_chapter not in chs:
            return False
        if sel_section:
            return sel_section in secs
        return True

    return False



def build_questions_from_df(df: pd.DataFrame,
                            sel_chapter: Optional[str],
                            sel_section: Optional[str],
                            chapter_col: Optional[str],
                            section_col: Optional[str]) -> list[dict]:
    """Create internal question dicts, filtered by chapter/section selections."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    q_col = _col_try(df, "question", "Question")
    a_cols = [
        _col_try(df, "optionA", "answerA", "AnswerA", "A"),
        _col_try(df, "optionB", "answerB", "AnswerB", "B"),
        _col_try(df, "optionC", "answerC", "AnswerC", "C"),
        _col_try(df, "optionD", "answerD", "AnswerD", "D"),
    ]
    ans_col = _col_try(df, "answer", "correct", "Correct", "correct answer", "Correct Answer")

    if not q_col or not ans_col or any(c is None for c in a_cols):
        raise ValueError(
            "CSV is missing required columns. Expected: "
            "question, optionA, optionB, optionC, optionD, answer"
        )

    now_iso = datetime.now().isoformat()
    items: list[dict] = []

    for _, row in df.iterrows():
        if not _row_matches_filters(row, chapter_col, section_col, sel_chapter, sel_section):
            continue

        options = [str(row[a_cols[0]]), str(row[a_cols[1]]), str(row[a_cols[2]]), str(row[a_cols[3]])]
        ans_raw = str(row[ans_col]).strip()

        # Parse A-D or try to match full text
        m = re.match(r"^\s*([A-D])\b", ans_raw, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
        else:
            letter = None
            try:
                idx_guess = [o.lower().strip() for o in options].index(ans_raw.lower().strip())
                letter = LETTERS[idx_guess]
            except Exception:
                letter = "A"

        correct_idx = LETTERS.index(letter) if letter in LETTERS else 0

        items.append({
            "question": row[q_col],
            "options": options,
            "answer_letter": letter,
            "answer_idx": correct_idx,
            "interval": 1,
            "next_time": now_iso,
            "status": "unseen",
            "streak": 0,
            "seen_count": 0
        })

    return items


def ensure_quiz_data_loaded():
    """(Re)load quiz_data if dataset/profile/filters changed."""
    dataset_changed = (
        st.session_state.quiz_data is None or
        st.session_state.selected_csv != st.session_state.last_loaded_csv or
        st.session_state.selected_profile != st.session_state.last_loaded_profile or
        st.session_state.get("filters_signature") != (st.session_state.selected_chapter, st.session_state.selected_section)
    )

    if dataset_changed and st.session_state.selected_csv:
        df = read_df(st.session_state.selected_csv)

        # detect chapter/section columns and taxonomy
        chapter_col, section_col, chapters_list, sections_by_chapter = detect_taxonomy(df)
        st.session_state.chapter_col = chapter_col
        st.session_state.section_col = section_col
        st.session_state.chapters_list = chapters_list
        st.session_state.sections_by_chapter = sections_by_chapter

        # build filtered questions
        questions = build_questions_from_df(
            df,
            st.session_state.selected_chapter,
            st.session_state.selected_section,
            chapter_col,
            section_col
        )
        st.session_state.quiz_data = questions

        # normalize (in case of older saves)
        for q in st.session_state.quiz_data:
            normalize_question_fields(q)

        st.session_state.last_loaded_csv = st.session_state.selected_csv
        st.session_state.last_loaded_profile = st.session_state.selected_profile
        st.session_state.filters_signature = (st.session_state.selected_chapter, st.session_state.selected_section)

        # Merge saved schedule only if saving is enabled
        if st.session_state.selected_profile != NO_PROFILE_LABEL:
            load_progress(merge=True)


# ---------- Load/save/reset progress ----------
def save_progress():
    path = progress_path_for(st.session_state.selected_csv or "default", st.session_state.selected_profile)
    if path is None:
        return
    try:
        with open(path, "w") as f:
            json.dump(st.session_state.quiz_data, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Couldn't save progress to '{path}': {e}")


def load_progress(merge=False):
    path = progress_path_for(st.session_state.selected_csv or "default", st.session_state.selected_profile)
    if path is None:
        return
    try:
        with open(path) as f:
            saved = json.load(f)
        if merge and st.session_state.quiz_data:
            saved_map = {q["question"]: q for q in saved}
            for q in st.session_state.quiz_data:
                if q["question"] in saved_map:
                    s = saved_map[q["question"]]
                    q["interval"] = s.get("interval", q["interval"])
                    q["next_time"] = s.get("next_time", q["next_time"])
        else:
            st.session_state.quiz_data = saved
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"Couldn't load prior progress from '{path}': {e}")


def reset_progress_for_current_set():
    path = progress_path_for(st.session_state.selected_csv or "default", st.session_state.selected_profile)
    if path and os.path.exists(path):
        try:
            os.remove(path)
            st.success("Progress reset for this set.")
        except Exception as e:
            st.warning(f"Couldn't delete progress file: {e}")
    if st.session_state.quiz_data:
        now_iso = datetime.now().isoformat()
        for q in st.session_state.quiz_data:
            q["interval"] = 1
            q["next_time"] = now_iso


# ---------- Rotation helpers ----------
def ensure_active_pool():
    if not st.session_state.quiz_data:
        return
    for q in st.session_state.quiz_data:
        normalize_question_fields(q)

    active = [q for q in st.session_state.quiz_data if q["status"] == "active"]
    need = max(0, ACTIVE_POOL_SIZE - len(active))
    if need <= 0:
        return

    unseen = [q for q in st.session_state.quiz_data if q["status"] == "unseen"]
    for q in unseen[:need]:
        q["status"] = "active"
        q["interval"] = max(1, int(q.get("interval", 1)))
        q["next_time"] = datetime.now().isoformat()


def graduate_question(q: dict):
    q["status"] = "deferred"
    minutes = max(5, int(q.get("interval", 1)))
    q["next_time"] = (datetime.now() + timedelta(minutes=minutes)).isoformat()
    q["streak"] = 0


def active_or_deferred_due_indexes():
    now = datetime.now()
    data = st.session_state.quiz_data or []
    due = []
    for i, q in enumerate(data):
        if q.get("status") in ("active", "deferred"):
            try:
                if datetime.fromisoformat(q["next_time"]) <= now:
                    due.append(i)
            except Exception:
                due.append(i)
    return due


def pick_next_question_idx_rotation():
    data = st.session_state.quiz_data or []
    now = datetime.now()

    active_ids = [i for i, q in enumerate(data) if q.get("status") == "active"]
    deferred_ids = [i for i, q in enumerate(data) if q.get("status") == "deferred"]

    due = active_or_deferred_due_indexes()
    due_active = [i for i in due if i in active_ids]
    if due_active:
        return random.choice(due_active)

    due_deferred = [i for i in due if i in deferred_ids]
    if due_deferred:
        return random.choice(due_deferred)

    if active_ids:
        active_sorted = sorted(
            active_ids,
            key=lambda i: datetime.fromisoformat(data[i]["next_time"])
                          if "next_time" in data[i] else now
        )
        return active_sorted[0]

    ensure_active_pool()
    active_ids = [i for i, q in enumerate(st.session_state.quiz_data) if q.get("status") == "active"]
    if active_ids:
        return random.choice(active_ids)
    return None


# ---------- Due helpers ----------
def due_indexes():
    now = datetime.now()
    return [i for i, q in enumerate(st.session_state.quiz_data) if datetime.fromisoformat(q["next_time"]) <= now]


# ---------- Screenshots (reading) ----------
def find_section_images(chapter: Optional[str], section: Optional[str]) -> dict:
    """
    Returns a mapping {section_title: [image_paths]}.
    If only chapter is selected: gather all section folders under that chapter.
    If chapter+section: just that one.
    """
    result: dict[str, list[str]] = {}
    root = Path(SECTIONS_IMG_ROOT)

    if not chapter and not section:
        return result  # no target

    if chapter and section:
        ch_dir = root / safe_name(chapter)
        sec_dir = ch_dir / safe_name(section)
        imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
            imgs.extend(sorted(glob.glob(str(sec_dir / ext))))
        if imgs:
            result[section] = imgs
        return result

    # Only chapter chosen → find all section subfolders
    if chapter:
        ch_dir = root / safe_name(chapter)
        if not ch_dir.exists():
            return result
        # If we know declared sections, use that order; else list directories
        sections_known = st.session_state.sections_by_chapter.get(chapter, [])
        if sections_known:
            sections = sections_known
        else:
            # derive from folders
            sections = []
            for p in ch_dir.glob("*"):
                if p.is_dir():
                    sections.append(p.name.replace("_", " "))
            sections = sorted(sections, key=lambda s: s.lower())

        for sec in sections:
            sec_dir = ch_dir / safe_name(sec)
            imgs = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
                imgs.extend(sorted(glob.glob(str(sec_dir / ext))))
            if imgs:
                result[sec] = imgs

    return result


# ---------- UI: question rendering ----------
def show_question(idx):
    q = st.session_state.quiz_data[idx]
    opts_idx = list(range(4))
    def fmt(i): return f"{LETTERS[i]}. {q['options'][i]}"

    if st.session_state.awaiting_continue:
        st.radio("Choose an answer:", opts_idx, format_func=fmt, key=f"choice_q{idx}", disabled=True)
        if st.session_state.last_was_correct:
            st.success(st.session_state.last_feedback)
        else:
            st.error(st.session_state.last_feedback)

        if st.button("Continue ➜", type="primary", key=f"continue_q{idx}"):
            st.session_state.awaiting_continue = False
            st.session_state.last_was_correct = None
            st.session_state.last_feedback = ""
            finish_or_continue_session()
        return

    selected_idx = st.radio("Choose an answer:", opts_idx, format_func=fmt, key=f"choice_q{idx}")
    submitted = st.button("Submit", type="primary", key=f"submit_q{idx}")

    if submitted:
        q["seen_count"] = int(q.get("seen_count", 0)) + 1
        correct_idx = q["answer_idx"]
        correct_letter = LETTERS[correct_idx]
        correct_text = q["options"][correct_idx]

        if selected_idx == correct_idx:
            st.session_state.correct_count += 1
            q["interval"] = max(1, int(q.get("interval", 1) * INTERVAL_GROWTH_FACTOR))
            q["streak"] = int(q.get("streak", 0)) + 1
            st.session_state.last_was_correct = True
            st.session_state.last_feedback = "✅ Correct!"

            q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()

            if q.get("status") == "active" and q["streak"] >= GRADUATE_STREAK:
                graduate_question(q)
                ensure_active_pool()

            st.session_state.questions_answered += 1
            save_progress()

            st.success("✅ Correct!")
            time.sleep(0.6)

            st.session_state.awaiting_continue = False
            st.session_state.current_question_idx = None
            finish_or_continue_session()
            return
        else:
            q["interval"] = BASE_INTERVAL_MIN
            q["streak"] = 0
            st.session_state.last_was_correct = False
            st.session_state.last_feedback = f"❌ Wrong! The correct answer was {correct_letter}. {correct_text}"

            q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()
            if q.get("status") == "deferred":
                q["status"] = "active"

            st.session_state.questions_answered += 1
            save_progress()

            st.session_state.awaiting_continue = True
            st.rerun()


def finish_or_continue_session():
    n = st.session_state.num_questions
    if st.session_state.questions_answered >= n:
        st.session_state.screen = "summary"
    else:
        st.session_state.current_question_idx = None
    st.rerun()


# ---------- Screens ----------
def start_screen():
    st.markdown(
        "<div style='text-align:center; font-size:2.2rem; line-height:1.2'>"
        "🐝 <b>QuizzyBee</b> 🌸"
        "</div>",
        unsafe_allow_html=True
    )

    # Profiles & datasets
    st.session_state.profiles = load_profiles()
    st.session_state.available_sets = list_question_sets(".")
    if not st.session_state.available_sets:
        st.error("No CSV question sets found. Add a CSV with headers like: question, optionA..D, answer (+ chapter/section columns).")
        return

    if not st.session_state.selected_csv or st.session_state.selected_csv not in st.session_state.available_sets:
        st.session_state.selected_csv = st.session_state.available_sets[0]

    # Profile picker
    profile_options = [NO_PROFILE_LABEL] + st.session_state.profiles
    if st.session_state.selected_profile not in profile_options:
        st.session_state.selected_profile = NO_PROFILE_LABEL
    st.selectbox(
        "Profile",
        options=profile_options,
        index=profile_options.index(st.session_state.selected_profile),
        key="selected_profile",
        help="Choose a profile to save progress, or 'No profile' to practice without saving."
    )

    # Add a new profile
    with st.expander("Add a new profile"):
        st.text_input("New profile name", key="new_profile_name", placeholder="e.g., Ilse, Student A")
        cols = st.columns([1, 1, 4])
        if cols[0].button("Add profile"):
            name = st.session_state.new_profile_name.strip()
            if not name:
                st.warning("Please enter a name.")
            elif name == NO_PROFILE_LABEL:
                st.warning("That name is reserved. Pick another.")
            else:
                add_profile(name)
                st.session_state.new_profile_name = ""
                st.success(f"Profile '{name}' added.")
                st.rerun()

    # Dataset picker
    st.selectbox(
        "Choose a question set (CSV)",
        options=st.session_state.available_sets,
        index=st.session_state.available_sets.index(st.session_state.selected_csv),
        key="selected_csv"
    )

    # Load df and detect taxonomy for this CSV to populate pickers
    try:
        df = read_df(st.session_state.selected_csv)
        chapter_col, section_col, chapters_list, sections_by_chapter = detect_taxonomy(df)
        st.session_state.chapter_col = chapter_col
        st.session_state.section_col = section_col
        st.session_state.chapters_list = chapters_list
        st.session_state.sections_by_chapter = sections_by_chapter
    except Exception as e:
        st.warning(f"Couldn't inspect CSV: {e}")

    # --- Targeting: Chapter / Section pickers ---
    if st.session_state.chapters_list:
        # choose chapter (or none)
        chapter_choices = ["— All chapters —"] + st.session_state.chapters_list
        default_ch_idx = 0
        if st.session_state.selected_chapter and st.session_state.selected_chapter in st.session_state.chapters_list:
            default_ch_idx = chapter_choices.index(st.session_state.selected_chapter)
        st.session_state.selected_chapter = st.selectbox(
            "Target a Chapter (optional)",
            options=chapter_choices,
            index=default_ch_idx,
        )
        if st.session_state.selected_chapter == "— All chapters —":
            st.session_state.selected_chapter = None

        # section depends on chapter
        if st.session_state.selected_chapter:
            sections = st.session_state.sections_by_chapter.get(st.session_state.selected_chapter, [])
            section_choices = ["— All sections in this chapter —"] + sections if sections else ["— No sections found —"]
            default_sec_idx = 0
            if st.session_state.selected_section and sections and st.session_state.selected_section in sections:
                default_sec_idx = section_choices.index(st.session_state.selected_section)
            st.session_state.selected_section = st.selectbox(
                "Target a Section (optional)",
                options=section_choices,
                index=min(default_sec_idx, len(section_choices) - 1),
                help="If you leave this as 'All', you'll see all sections of the selected chapter."
            )
            if st.session_state.selected_section and st.session_state.selected_section.startswith("—"):
                st.session_state.selected_section = None
        else:
            # no chapter → let user optionally pick any section (rare; only if df has a section column)
            if st.session_state.section_col:
                # derive all sections globally
                all_secs = []
                for v in df[st.session_state.section_col].fillna(""):
                    all_secs.extend(_split_multi(v))
                all_secs = sorted(sorted(set(all_secs)), key=lambda s: s.lower())
                sec_choices = ["— Any section —"] + all_secs
                default_any_sec_idx = 0
                if st.session_state.selected_section and st.session_state.selected_section in all_secs:
                    default_any_sec_idx = sec_choices.index(st.session_state.selected_section)
                st.session_state.selected_section = st.selectbox(
                    "Target a Section (without picking a chapter)",
                    options=sec_choices,
                    index=default_any_sec_idx
                )
                if st.session_state.selected_section == "— Any section —":
                    st.session_state.selected_section = None
    else:
        st.info("This CSV doesn't declare chapters/sections. You'll drill the whole set.")

    # Pre-reading preference
    st.session_state.pre_read_mode = st.radio(
        "Before the questions, would you like to read…",
        PRE_READ_MODES,
        index=PRE_READ_MODES.index(st.session_state.pre_read_mode),
        horizontal=True,
        help="Choose screenshots first (if available), or fallback to JSON text."
    )

    # Build/refresh quiz data with current filters
    ensure_quiz_data_loaded()

    # Info about due items
    currently_due = len(due_indexes()) if st.session_state.quiz_data else 0
    # show how many questions are in the filtered set
    total_filtered = len(st.session_state.quiz_data or [])
    st.info(
        f"Profile: **{st.session_state.selected_profile}**  •  "
        f"Filtered set size: **{total_filtered}**  •  "
        f"Due now: **{currently_due}**"
    )

    # Reset with confirmation (only if saving)
    if st.session_state.selected_profile != NO_PROFILE_LABEL and total_filtered:
        if not st.session_state.confirm_reset:
            if st.button("Reset progress for this set"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("Are you sure? This will erase saved intervals & due times for this profile and set.")
            c1, c2 = st.columns(2)
            if c1.button("✅ Yes, reset now"):
                reset_progress_for_current_set()
                st.session_state.confirm_reset = False
                st.rerun()
            if c2.button("❌ Cancel"):
                st.info("Reset canceled.")
                st.session_state.confirm_reset = False
                st.rerun()

    # Session size
    st.session_state.num_questions = st.number_input(
        "Questions per session",
        min_value=1,
        max_value=100,
        value=st.session_state.num_questions,
        step=1
    )

    # Persist defaults (including last chosen chapter/section)
    save_settings(
        last_profile=st.session_state.selected_profile,
        last_num_questions=st.session_state.num_questions,
        pre_read_mode=st.session_state.pre_read_mode,
        last_chapter=st.session_state.selected_chapter,
        last_section=st.session_state.selected_section,
    )

    if st.button("Start Session ▶", type="primary"):
        st.session_state.questions_answered = 0
        st.session_state.correct_count = 0
        st.session_state.session_started_at = datetime.now().isoformat()
        st.session_state.current_question_idx = None
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""

        # Prefer screenshot reading for selected chapter/section
        st.session_state.screenshot_map = find_section_images(st.session_state.selected_chapter,
                                                              st.session_state.selected_section)

        # also prepare JSON reading fallback (old behavior)
        payload, support, stem = load_reading_payload(st.session_state.selected_csv)
        st.session_state.reading_payload = payload
        st.session_state.reading_stem = stem

        want_reading = (st.session_state.pre_read_mode != PRE_READ_NONE)
        have_shots = bool(st.session_state.screenshot_map)

        # route to reading if user wants it and we have screenshots OR valid JSON fallback
        go_reading = False
        if want_reading and have_shots:
            go_reading = True
        elif want_reading and payload and (support == "both" or (support == "summary_only" and st.session_state.pre_read_mode == PRE_READ_SUMMARY)):
            go_reading = True

        st.session_state.screen = "reading" if go_reading else "quiz"

        ensure_quiz_data_loaded()
        ensure_active_pool()
        st.rerun()


def reading_screen():
    """Show screenshots for selected (chapter[/section]) OR fallback to JSON reading."""
    # 1) Prefer screenshots
    shots = st.session_state.get("screenshot_map") or {}
    if shots:
        # Title
        title_parts = []
        if st.session_state.selected_chapter:
            title_parts.append(st.session_state.selected_chapter)
        if st.session_state.selected_section:
            title_parts.append(st.session_state.selected_section)
        st.header(" • ".join(title_parts) if title_parts else "Reading")

        # Render sections with images
        for sec_title, img_paths in shots.items():
            st.subheader(sec_title)
            for p in img_paths:
                st.image(p, use_column_width=True)
            st.markdown("---")

        if st.button("Start Quiz ▶", type="primary"):
            st.session_state.screen = "quiz"
            st.rerun()
        return

    # 2) Fallback to JSON reading payload (old flow)
    payload = st.session_state.reading_payload
    stem = st.session_state.reading_stem or "Reading"
    if not payload:
        st.info("No screenshots or reading material found. Starting the quiz.")
        st.session_state.screen = "quiz"
        st.rerun()
        return

    st.header(payload.get("title", stem).replace("_", " "))

    mode = st.session_state.pre_read_mode
    if mode == PRE_READ_FULL and payload.get("reading"):
        est = payload.get("estimated_read_time_sec", 60)
        st.caption(f"Estimated read time ~ {max(1, int(est/60))} min")
        st.write(payload["reading"])
        if payload.get("summary_bullets"):
            with st.expander("Quick summary"):
                for b in payload["summary_bullets"]:
                    st.markdown(f"- {b}")
    elif mode == PRE_READ_SUMMARY and payload.get("summary_bullets"):
        st.subheader("Summary")
        for b in payload["summary_bullets"]:
            st.markdown(f"- {b}")
        if payload.get("reading"):
            with st.expander("Read the full section"):
                st.write(payload["reading"])
    else:
        st.info("This chapter has no matching reading/summary. Starting the quiz.")
        st.session_state.screen = "quiz"
        st.rerun()
        return

    if st.button("Start Quiz ▶", type="primary"):
        st.session_state.screen = "quiz"
        st.rerun()


def quiz_screen():
    ensure_quiz_data_loaded()
    if not st.session_state.quiz_data:
        st.error("No questions loaded for this selection. Try choosing a different chapter/section or CSV.")
        if st.button("Back to start"):
            st.session_state.screen = "start"
            st.rerun()
        return

    n = st.session_state.num_questions
    answered = st.session_state.questions_answered
    st.progress(answered / n if n else 0.0)
    # Show current target
    tgt = []
    if st.session_state.selected_chapter:
        tgt.append(st.session_state.selected_chapter)
    if st.session_state.selected_section:
        tgt.append(st.session_state.selected_section)
    target_str = " • ".join(tgt) if tgt else "All"
    st.caption(
        f"Question {min(answered + 1, n)} of {n}  •  "
        f"Target: {target_str}  •  "
        f"Profile: {st.session_state.selected_profile}"
    )

    if st.session_state.current_question_idx is None:
        idx = pick_next_question_idx_rotation()
        if idx is None:
            st.success("🎉 All due questions are up to date right now.")
            if answered > 0 and not st.session_state.awaiting_continue:
                st.write("That’s the end of your session for now.")
                st.session_state.screen = "summary"
                st.rerun()
            else:
                if st.button("Back to start"):
                    st.session_state.screen = "start"
                    st.rerun()
            return
        st.session_state.current_question_idx = idx

    idx = st.session_state.current_question_idx
    st.subheader(st.session_state.quiz_data[idx]["question"])
    show_question(idx)

    st.markdown("---")
    if st.button("🏠 Return to Start", type="secondary"):
        st.session_state.screen = "start"
        st.session_state.current_question_idx = None
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""
        st.rerun()


def summary_screen():
    total = st.session_state.questions_answered
    correct = st.session_state.correct_count
    st.header("Session Summary")
    if total > 0:
        accuracy = 100.0 * correct / total
        st.metric("Questions answered", total)
        st.metric("Correct answers", correct)
        st.metric("Accuracy", f"{accuracy:.1f}%")
    else:
        st.write("No questions were due this time. Nice and caught up!")

    due_count = len(due_indexes())
    tgt = []
    if st.session_state.selected_chapter:
        tgt.append(st.session_state.selected_chapter)
    if st.session_state.selected_section:
        tgt.append(st.session_state.selected_section)
    target_str = " • ".join(tgt) if tgt else "All"

    st.caption(
        f"Target: **{target_str}**  •  "
        f"Currently due for review: **{due_count}**  •  "
        f"Profile: **{st.session_state.selected_profile}**"
    )

    if st.button("Return to Start", type="primary"):
        st.session_state.screen = "start"
        st.session_state.current_question_idx = None
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""
        st.rerun()


# ---------- Router ----------
screen = st.session_state.screen
if screen == "start":
    start_screen()
elif screen == "reading":
    reading_screen()
elif screen == "quiz":
    quiz_screen()
else:
    summary_screen()

