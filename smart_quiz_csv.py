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
from pathlib import Path  # NEW

# ---------- Page Setup ----------
st.set_page_config(page_title="🧠 QuizzyBee", layout="centered")

# ---------- Helpers: files ----------
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

PROFILES_FILE = "quiz_profiles.json"
SETTINGS_FILE = "quiz_settings.json"   # persists defaults across runs
NO_PROFILE_LABEL = "No profile (don't save)"

# NEW: Pre-reading modes
PRE_READ_NONE = "Skip"
PRE_READ_SUMMARY = "Summary only"
PRE_READ_FULL = "Full reading"
PRE_READ_MODES = [PRE_READ_NONE, PRE_READ_SUMMARY, PRE_READ_FULL]

# --- Rotation & Spaced Practice Config (NEW) ---
ACTIVE_POOL_SIZE = 10          # how many questions to drill at once
GRADUATE_STREAK = 3            # correct-in-a-row needed to graduate a question
BASE_INTERVAL_MIN = 1          # minutes after a wrong answer (short retry)
INTERVAL_GROWTH_FACTOR = 2.0   # doubling on each correct

# --- Micro-notes throttle (NEW) ---
DEFAULT_INFO_REPEAT_MIN = 10   # minimum minutes before showing the same concept again in-session

# --- Quick-check helpers (section + classic) ---
TAGGED_CSV_PATH = "anatomy_physiology_mcqs_tagged.csv"   # <- set to your actual path

LETTERS = ["A", "B", "C", "D"]

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:80]

def section_id_from_titles(chapter_title: str, section_title: str) -> str:
    return f"{_slug(chapter_title.split(' - ')[0])}::{_slug(section_title)}"

@st.cache_data
def load_tagged_mcqs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "topic_confidence" in df.columns:
        df["topic_confidence"] = pd.to_numeric(df["topic_confidence"], errors="coerce").fillna(0.0)
    if "topic_section_confidence" in df.columns:
        df["topic_section_confidence"] = pd.to_numeric(df["topic_section_confidence"], errors="coerce").fillna(0.0)
    # Normalize question column name to 'question'
    if "Question" in df.columns and "question" not in df.columns:
        df = df.rename(columns={"Question": "question"})
    return df

def qc_pick_for_section(df: pd.DataFrame, chapter_title: str, section_title: str,
                        n: int = 5, min_conf: float = 0.55) -> list[str]:
    sid = section_id_from_titles(chapter_title, section_title)
    mask = (df.get("topic_section_id", "") == sid) & (df.get("topic_section_confidence", 0.0) >= min_conf)
    out = df.loc[mask].sort_values("topic_section_confidence", ascending=False)["question"].dropna().tolist()
    return out[:n]

def qc_pick_for_chapter(df: pd.DataFrame, chapter_title: str,
                        n: int = 5, min_conf: float = 0.50) -> list[str]:
    mask = (df.get("topic_chapter", "") == chapter_title) & (df.get("topic_confidence", 0.0) >= min_conf)
    out = df.loc[mask].sort_values("topic_confidence", ascending=False)["question"].dropna().tolist()
    return out[:n]

def get_tag_catalog(df: pd.DataFrame):
    """Return (chapters_list, sections_by_chapter: dict[chapter] -> list[section_title or id])."""
    chapters = []
    sections_by_ch = {}
    ch_col = "topic_chapter"
    sec_title_col = "topic_section_title" if "topic_section_title" in df.columns else None
    sec_id_col = "topic_section_id" if "topic_section_id" in df.columns else None

    if ch_col in df.columns:
        chapters = sorted([c for c in df[ch_col].dropna().unique().tolist() if str(c).strip()])
        for ch in chapters:
            mask = df[ch_col] == ch
            if sec_title_col and sec_title_col in df.columns:
                secs = sorted(set([s for s in df.loc[mask, sec_title_col].dropna().tolist() if str(s).strip()]))
            elif sec_id_col and sec_id_col in df.columns:
                secs = sorted(set([s for s in df.loc[mask, sec_id_col].dropna().tolist() if str(s).strip()]))
            else:
                secs = []
            sections_by_ch[ch] = secs
    return chapters, sections_by_ch

def build_qc_filter(df: pd.DataFrame, chapter: str | None, section: str | None,
                    mode: str, n: int, min_conf: float) -> list[str]:
    """
    Return list of question texts matching the chosen mode/targets.
    """
    ch_col = "topic_chapter"
    sec_title_col = "topic_section_title" if "topic_section_title" in df.columns else None
    sec_id_col = "topic_section_id" if "topic_section_id" in df.columns else None
    q_col = "question" if "question" in df.columns else ("Question" if "Question" in df.columns else None)
    conf_ch = "topic_confidence" if "topic_confidence" in df.columns else None
    conf_sec = "topic_section_confidence" if "topic_section_confidence" in df.columns else None
    if not q_col:
        return []

    df2 = df.copy()

    if chapter and ch_col in df2.columns:
        df2 = df2[df2[ch_col] == chapter]

    # Section Quick-Checks prefers section targeting
    if mode == "Section Quick-Checks" and section:
        if sec_title_col and (sec_title_col in df2.columns):
            df2 = df2[df2[sec_title_col] == section]
            if conf_sec and (conf_sec in df2.columns):
                df2 = df2[df2[conf_sec] >= float(min_conf)]
            sortcol = conf_sec or conf_ch
        elif sec_id_col and (sec_id_col in df2.columns):
            df2 = df2[df2[sec_id_col] == section]
            if conf_sec and (conf_sec in df2.columns):
                df2 = df2[df2[conf_sec] >= float(min_conf)]
            sortcol = conf_sec or conf_ch
        else:
            sortcol = conf_ch
            if conf_ch and (conf_ch in df2.columns):
                df2 = df2[df2[conf_ch] >= float(min_conf)]
    else:
        # Classic or no section specified → chapter-level or whole set
        sortcol = conf_ch
        if conf_ch and (conf_ch in df2.columns):
            df2 = df2[df2[conf_ch] >= max(0.45, float(min_conf) - 0.05)]

    if sortcol and (sortcol in df2.columns):
        df2 = df2.sort_values(sortcol, ascending=False)

    questions = [str(x) for x in df2[q_col].dropna().tolist()]
    return questions[: int(n)]

def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_").lower()

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

def progress_path_for(csv_file: str | None,
                      profile_name: str | None,
                      mode: str | None = None,
                      sel_chapter: str | None = None,
                      sel_section: str | None = None):
    """Unique progress file per dataset/profile AND per mode; for Section QC also per chapter/section."""
    if not profile_name or profile_name == NO_PROFILE_LABEL:
        return None
    mode_key = "classic" if mode == "Classic Quiz Mode" else ("section_qc" if mode == "Section Quick-Checks" else "classic")
    if mode_key == "classic":
        base = os.path.splitext(os.path.basename(csv_file or "default"))[0]
        safe_ds = safe_name(base)
        safe_prof = safe_name(profile_name)
        return f"quiz_progress__{mode_key}__{safe_ds}__{safe_prof}.json"
    else:
        # Section QC: save per chapter/section selection
        ch = safe_name(sel_chapter or "all_chapters")
        sec = safe_name(sel_section or "all_sections")
        safe_prof = safe_name(profile_name)
        return f"quiz_progress__{mode_key}__{ch}__{sec}__{safe_prof}.json"


def render_section(sec, chapter_payload):
    img_id = sec.get("image")
    img_ids = sec.get("images")
    fig_index = {f["figure_id"]: f for f in (chapter_payload.get("figures") or [])}

    def _show(fid):
        f = fig_index.get(fid)
        if not f: 
            return
        st.image(f["url"], caption=f.get("caption") or "", use_column_width=True)

    if img_id:
        _show(img_id)
    if img_ids:
        for fid in img_ids:
            _show(fid)

    st.markdown(sec.get("markdown",""))


# ---------- Profiles store ----------
def load_profiles() -> list[str]:
    """Persisted list of profile display names. Create with ['Ilse'] on first run."""
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
    return ["Ilse"]  # fallback

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
    """Return persisted settings."""
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

def save_settings(last_profile: str, last_num_questions: int, pre_read_mode: str):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(
                {
                    "last_profile": last_profile,
                    "last_num_questions": int(last_num_questions),
                    "pre_read_mode": pre_read_mode,
                },
                f, indent=2
            )
    except Exception as e:
        st.warning(f"Couldn't save settings: {e}")

# ---------- App State Defaults ----------
def init_state():
    settings = load_settings()

    defaults = {
        "screen": "start",              # "start" | "reading" | "info" | "quiz" | "summary"
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
        # NEW: pre-reading
        "pre_read_mode": settings.get("pre_read_mode", PRE_READ_NONE),
        "reading_payload": None,        # loaded reading JSON dict
        "reading_stem": None,           # e.g. "ch01_Organisation_of_the_Body"
        # NEW: interspersed info state
        "pending_info_for_idx": None,   # which question index to show after info
        "shown_info_keys": [],          # concepts shown this session
        "last_info_key": None,          # last concept shown (for change detection)
        "last_info_shown_at": {},       # {info_key: iso_str} throttle per concept
        # --- Quick-check mode ---
        "quiz_mode": "Classic Quiz Mode",      # or "Section Quick-Checks"
        "qc_num_qs": 5,
        "qc_min_conf": 0.55,
        "qc_fallback_to_chapter": True,
        "qc_filter_questions": None,           # set[str]: only serve these question texts if not None
        "current_chapter_title": None,         # (from reading payload when present)
        "current_section_title": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Global title (cute!)
st.title("🐝 QuizzyBee")
st.caption("Questions return sooner if you miss them!")

# ---------- Load questions from CSV ----------
@st.cache_data
def load_questions(csv_file="anatomy_physiology_mcqs.csv"):
    df = pd.read_csv(csv_file, sep=",", engine="python")
    df.columns = df.columns.str.strip()

    def col_try(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    q_col = col_try("question", "Question")
    a_cols = [
        col_try("optionA", "answerA", "AnswerA", "A"),
        col_try("optionB", "answerB", "AnswerB", "B"),
        col_try("optionC", "answerC", "AnswerC", "C"),
        col_try("optionD", "answerD", "AnswerD", "D"),
    ]
    ans_col = col_try("answer", "correct", "Correct", "correct answer", "Correct Answer")
    info_col = col_try("info_key", "topic", "concept")  # NEW: optional per-question concept key

    if not q_col or not ans_col or any(c is None for c in a_cols):
        raise ValueError(
            f"CSV '{csv_file}' is missing required columns. Expected headers like: "
            "question, optionA, optionB, optionC, optionD, answer"
        )

    now_iso = datetime.now().isoformat()
    questions_list = []
    for _, row in df.iterrows():
        options = [str(row[a_cols[0]]), str(row[a_cols[1]]), str(row[a_cols[2]]), str(row[a_cols[3]])]
        ans_raw = str(row[ans_col]).strip()

        # Parse letter A-D, or fall back to matching full option text
        letter = None
        m = re.match(r"^\s*([A-D])\b", ans_raw, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
        elif ans_raw:
            try:
                idx_guess = [o.lower().strip() for o in options].index(ans_raw.lower().strip())
                letter = LETTERS[idx_guess]
            except ValueError:
                letter = "A"

        correct_idx = LETTERS.index(letter) if letter in LETTERS else 0
        info_key = str(row[info_col]).strip() if info_col and pd.notna(row[info_col]) else None

        questions_list.append({
            "question": row[q_col],
            "options": options,
            "answer_letter": letter,
            "answer_idx": correct_idx,
            "interval": 1,
            "next_time": now_iso,
            "status": "unseen",          # unseen | active | deferred
            "streak": 0,                 # consecutive correct answers
            "seen_count": 0,             # total times shown (nice for debugging)
            "info_key": info_key,        # NEW
        })
    return questions_list

# ---------- Rotation helpers (NEW) ----------
def normalize_question_fields(q: dict):
    """Add defaults for older progress files."""
    q.setdefault("status", "unseen")
    q.setdefault("streak", 0)
    q.setdefault("seen_count", 0)
    q.setdefault("interval", 1)
    q.setdefault("next_time", datetime.now().isoformat())
    q.setdefault("info_key", None)
    return q

def ensure_active_pool():
    """
    Promote unseen questions into 'active' until we have ACTIVE_POOL_SIZE active items,
    respecting the optional quick-check filter.
    """
    if not st.session_state.quiz_data:
        return

    for q in st.session_state.quiz_data:
        normalize_question_fields(q)

    flt = st.session_state.get("qc_filter_questions")

    def allowed(q: dict) -> bool:
        return (not flt) or (str(q.get("question","")) in flt)

    active = [q for q in st.session_state.quiz_data if q.get("status") == "active" and allowed(q)]
    need = max(0, ACTIVE_POOL_SIZE - len(active))
    if need <= 0:
        return

    unseen = [q for q in st.session_state.quiz_data if q.get("status") == "unseen" and allowed(q)]
    for q in unseen[:need]:
        q["status"] = "active"
        q["next_time"] = datetime.now().isoformat()

def graduate_question(q: dict):
    """
    Move a well-known question out of the active pool and schedule it further out.
    It will come back later when it's due.
    """
    q["status"] = "deferred"
    minutes = max(5, int(q.get("interval", 1)))  # at least 5 minutes
    q["next_time"] = (datetime.now() + timedelta(minutes=minutes)).isoformat()
    q["streak"] = 0  # optional: reset streak once it's graduated

def active_or_deferred_due_indexes():
    now = datetime.now()
    data = st.session_state.quiz_data or []
    due = []
    for i, q in enumerate(data):
        if q.get("status") in ("active", "deferred") and _qc_allow_index(i):
            try:
                if datetime.fromisoformat(q["next_time"]) <= now:
                    due.append(i)
            except Exception:
                due.append(i)
    return due

def active_or_deferred_due_count():
    now = datetime.now()
    data = st.session_state.quiz_data or []
    def due_time(q):
        try:
            return datetime.fromisoformat(q.get("next_time","")) <= now
        except Exception:
            return True
    return sum(1 for q in data if q.get("status") in ("active","deferred") and due_time(q))

def pick_next_question_idx_rotation():
    """
    Priority:
    1) Due 'active' questions (drill set)
    2) Due 'deferred' questions (spaced checks)
    3) If nothing due, pick the 'active' question with the earliest next_time
    """
    data = st.session_state.quiz_data or []
    now = datetime.now()

    active_ids = [i for i, q in enumerate(data) if q.get("status") == "active" and _qc_allow_index(i)]
    deferred_ids = [i for i, q in enumerate(data) if q.get("status") == "deferred" and _qc_allow_index(i)]

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

def _qc_allow_index(i: int) -> bool:
    """Return True if index i is allowed by the current quick-check filter (or no filter)."""
    flt = st.session_state.get("qc_filter_questions")
    if not flt:
        return True
    try:
        q = st.session_state.quiz_data[i]
        return str(q.get("question","")) in flt
    except Exception:
        return False

# ---------- Reading loader (NEW) ----------
def extract_chapter_stem_from_csv(csv_path: str) -> str | None:
    name = Path(csv_path).stem
    m = re.search(r"(ch\d{2}_[A-Za-z0-9_]+)$", name)
    if m:
        return m.group(1)
    m = re.search(r"(ch\d{2}_.+)", name)  # fallback catch-all
    return m.group(1) if m else None

def find_reading_json_for_csv(csv_path: str) -> str | None:
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

def load_reading_payload(csv_path: str) -> tuple[dict | None, str | None, str | None]:
    """
    Return (payload, mode_support, stem)

    Expected JSON keys (any optional):
      - title (str)
      - reading (str)
      - estimated_read_time_sec (int)
      - summary_bullets (list[str])
      - micro_notes (dict[str -> {title, markdown, image|images}])
      - figures (list[{figure_id, url, caption}])
      - sections (list[{title?, markdown?, image?, images?}])   # OPTIONAL chunked reading
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
        figures = data.get("figures") or []
        sections = data.get("sections") or None   # optional

        payload = {
            "title": title,
            "reading": reading,
            "summary_bullets": summary_bullets,
            "estimated_read_time_sec": int(data.get("estimated_read_time_sec") or 60),
            "source_refs": data.get("source_refs", []),
            "micro_notes": data.get("micro_notes", {}),
            "figures": figures,
            "sections": sections,
        }

        # Determine what we can show before quiz
        if (sections and len(sections) > 0) or (reading and summary_bullets):
            support = "both"
        elif summary_bullets:
            support = "summary_only"
        elif reading:
            support = "both"   # treat plain reading as showable
        else:
            support = None

        return payload, support, stem
    except Exception as e:
        st.warning(f"Couldn't load reading JSON: {e}")
        return None, None, stem


def ensure_quiz_data_loaded():
    """(Re)load quiz_data if dataset or profile selection changed."""
    dataset_changed = (
        st.session_state.quiz_data is None or
        st.session_state.selected_csv != st.session_state.last_loaded_csv or
        st.session_state.selected_profile != st.session_state.last_loaded_profile
    )
    if dataset_changed and st.session_state.selected_csv:
        st.session_state.quiz_data = load_questions(st.session_state.selected_csv)
        for q in st.session_state.quiz_data:
            normalize_question_fields(q)
        st.session_state.last_loaded_csv = st.session_state.selected_csv
        st.session_state.last_loaded_profile = st.session_state.selected_profile
        if st.session_state.selected_profile != NO_PROFILE_LABEL:
            load_progress(merge=True)

# ---------- Load/save/reset progress (per dataset & profile) ----------
def save_progress():
    path = progress_path_for(
        st.session_state.selected_csv,
        st.session_state.selected_profile,
        st.session_state.quiz_mode,
        st.session_state.current_chapter_title,
        st.session_state.current_section_title,
    )
    if path is None:
        return
    try:
        with open(path, "w") as f:
            json.dump(st.session_state.quiz_data, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Couldn't save progress to '{path}': {e}")


def load_progress(merge=False):
    path = progress_path_for(
        st.session_state.selected_csv,
        st.session_state.selected_profile,
        st.session_state.quiz_mode,
        st.session_state.current_chapter_title,
        st.session_state.current_section_title,
    )
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
                    q["status"]  = s.get("status", q.get("status", "unseen"))
                    q["streak"]  = s.get("streak", q.get("streak", 0))
                    q["seen_count"] = s.get("seen_count", q.get("seen_count", 0))
                    if not q.get("info_key"):
                        q["info_key"] = s.get("info_key")
        else:
            st.session_state.quiz_data = saved
    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"Couldn't load prior progress from '{path}': {e}")


def reset_progress_for_current_set():
    path = progress_path_for(
        st.session_state.selected_csv,
        st.session_state.selected_profile,
        st.session_state.quiz_mode,
        st.session_state.current_chapter_title,
        st.session_state.current_section_title,
    )
    if path and os.path.exists(path):
        try:
            os.remove(path)
            st.success("Progress reset for this mode/selection.")
        except Exception as e:
            st.warning(f"Couldn't delete progress file: {e}")
    if st.session_state.quiz_data:
        now_iso = datetime.now().isoformat()
        for q in st.session_state.quiz_data:
            q["interval"] = 1
            q["next_time"] = now_iso
            q["status"] = "unseen"
            q["streak"] = 0
            q["seen_count"] = 0


# ---------- Quiz helpers ----------
def due_indexes():
    now = datetime.now()
    return [i for i, q in enumerate(st.session_state.quiz_data) if datetime.fromisoformat(q["next_time"]) <= now]

def pick_next_question_idx():
    due = due_indexes()
    if not due:
        return None
    return random.choice(due)

def finish_or_continue_session():
    n = st.session_state.num_questions
    if st.session_state.questions_answered >= n:
        st.session_state.screen = "summary"
    else:
        st.session_state.current_question_idx = None
    st.rerun()

def show_question(idx):
    q = st.session_state.quiz_data[idx]
    opts_idx = list(range(4))
    def fmt(i): return f"{LETTERS[i]}. {q['options'][i]}"

    if st.session_state.awaiting_continue:
        st.radio("Choose an answer:", opts_idx, format_func=fmt,
                 key=f"choice_q{idx}", disabled=True)
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
            # ✅ correct
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
            time.sleep(0.7)

            st.session_state.awaiting_continue = False
            st.session_state.current_question_idx = None
            finish_or_continue_session()
            return
        else:
            # ❌ incorrect
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

# ---------- Interspersed Info Screen (NEW) ----------
def info_screen():
    idx = st.session_state.pending_info_for_idx
    if idx is None:
        st.session_state.screen = "quiz"
        st.rerun()
        return

    if not st.session_state.quiz_data or idx >= len(st.session_state.quiz_data):
        st.session_state.screen = "quiz"
        st.rerun()
        return

    q = st.session_state.quiz_data[idx]
    info_key = q.get("info_key")
    payload = st.session_state.reading_payload or {}
    notes_all = payload.get("micro_notes") or {}
    notes = notes_all.get(info_key, {}) if info_key else {}

    display_title = notes.get("title") or (f"Heads-up: {info_key.replace('_',' ').title()}" if info_key else "Heads-up")
    body_md = notes.get("markdown") or "_No notes available for this concept._"

    st.header(display_title)

    # Prefer figure-id rendering if image(s) match known figure_ids
    fig_ids = None
    if "images" in notes and isinstance(notes["images"], list):
        fig_ids = [x for x in notes["images"] if isinstance(x, str)]
    elif "image" in notes and isinstance(notes["image"], str):
        fig_ids = [notes["image"]]

    used_render_section = False
    if fig_ids:
        # Build figure index to check if any id matches
        fig_index = {f.get("figure_id"): f for f in (payload.get("figures") or []) if f.get("figure_id")}
        if any(fid in fig_index for fid in fig_ids):
            sec_stub = {"markdown": body_md}
            # keep only valid figure ids
            valid_fids = [fid for fid in fig_ids if fid in fig_index]
            if len(valid_fids) == 1:
                sec_stub["image"] = valid_fids[0]
            elif len(valid_fids) > 1:
                sec_stub["images"] = valid_fids
            render_section(sec_stub, payload)
            used_render_section = True

    if not used_render_section:
        # Fallback: raw image url/path or no image at all
        raw_img = notes.get("image")
        if raw_img:
            try:
                st.image(raw_img, use_column_width=True)
            except Exception:
                st.caption("*(Image could not be loaded)*")
        st.markdown(body_md)

    c1, c2 = st.columns(2)
    if c1.button("Start question ▶", type="primary"):
        now_iso = datetime.now().isoformat()
        if info_key:
            if info_key not in st.session_state.shown_info_keys:
                st.session_state.shown_info_keys.append(info_key)
            st.session_state.last_info_key = info_key
            st.session_state.last_info_shown_at[info_key] = now_iso

        st.session_state.screen = "quiz"
        st.session_state.pending_info_for_idx = None
        st.rerun()
    if c2.button("Skip info and continue"):
        st.session_state.screen = "quiz"
        st.session_state.pending_info_for_idx = None
        st.rerun()


# ---------- Screens ----------
def start_screen():
    # Cute header (start screen only)
    st.markdown(
        "<div style='text-align:center; font-size:2.2rem; line-height:1.2'>"
        "🐝 <b>QuizzyBee</b> 🌸"
        "</div>",
        unsafe_allow_html=True
    )

    # Load profiles & datasets
    st.session_state.profiles = load_profiles()
    st.session_state.available_sets = list_question_sets(".")
    if not st.session_state.available_sets:
        st.error("No CSV question sets found in this repository. Add a CSV with headers like: question, optionA..D, answer.")
        return

    # ----- Quiz Type & Targeting (moved to top) -----
    st.markdown("### Quiz Type & Targeting")
    colA, colB, colC = st.columns([1.1, 1, 1])
    with colA:
        st.session_state.quiz_mode = st.radio(
            "Mode",
            options=["Section Quick-Checks", "Classic Quiz Mode"],
            index=0 if st.session_state.quiz_mode == "Section Quick-Checks" else 1,
            help="Section = only questions tagged to a subchapter; Classic = ordinary chapter or entire set."
        )
    with colB:
        if st.session_state.quiz_mode == "Section Quick-Checks":
            st.session_state.qc_num_qs = st.number_input("How many questions?", min_value=3, max_value=50,
                                                         value=int(st.session_state.qc_num_qs), step=1)
        else:
            # Placeholder for spacing
            st.empty()
    with colC:
        if st.session_state.quiz_mode == "Section Quick-Checks":
            st.session_state.qc_min_conf = st.slider("Min. confidence", 0.30, 0.90,
                                                     float(st.session_state.qc_min_conf), 0.05)
        else:
            st.empty()

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

    # Dataset picker (Classic mode needs this; Section QC often uses the tagged CSV too)
    st.selectbox(
        "Choose a question set (CSV)",
        options=st.session_state.available_sets,
        index=st.session_state.available_sets.index(st.session_state.selected_csv) if st.session_state.selected_csv in st.session_state.available_sets else 0,
        key="selected_csv"
    )

    # Try to load tagged MCQs for chapter/subchapter pickers
    try:
        df_tag = load_tagged_mcqs(TAGGED_CSV_PATH)
    except Exception:
        df_tag = None

    if st.session_state.quiz_mode == "Section Quick-Checks":
        if df_tag is None:
            st.error("Tagged MCQs file not available, section/chapter targeting disabled.")
            return

        chapters, sections_by_ch = get_tag_catalog(df_tag)
        sel_chapter = st.selectbox("Chapter", chapters)
        sec_opts = sections_by_ch.get(sel_chapter, [])
        sel_section = st.selectbox("Subchapter (optional)", ["<All subchapters>"] + sec_opts)
        if sel_section == "<All subchapters>":
            sel_section = None

        st.session_state.qc_fallback_to_chapter = st.checkbox(
            "Fallback to chapter if section is sparse",
            value=bool(st.session_state.qc_fallback_to_chapter)
        )

        # Build quick-check filter preview (but don't commit to state until Start)
        preview_qs = build_qc_filter(
            df=df_tag,
            chapter=sel_chapter,
            section=sel_section,
            mode="Section Quick-Checks",
            n=int(st.session_state.qc_num_qs),
            min_conf=float(st.session_state.qc_min_conf),
        )
        if st.session_state.qc_fallback_to_chapter and sel_section and len(preview_qs) < int(st.session_state.qc_num_qs):
            topup = build_qc_filter(
                df=df_tag,
                chapter=sel_chapter,
                section=None,
                mode="Classic Quiz Mode",
                n=int(st.session_state.qc_num_qs) - len(preview_qs),
                min_conf=max(0.45, float(st.session_state.qc_min_conf) - 0.1),
            )
            preview_qs.extend([q for q in topup if q not in preview_qs])

        st.caption(f"Will drill **{min(len(preview_qs), int(st.session_state.qc_num_qs))}** question(s) this session.")

        # Due count here should reflect the filtered subset with current saved intervals
        ensure_quiz_data_loaded()
        # Temporarily set the filter to compute due, then revert
        old_filter = st.session_state.get("qc_filter_questions")
        st.session_state.qc_filter_questions = set(map(str, preview_qs)) if preview_qs else None
        ensure_active_pool()
        due_count = active_or_deferred_due_count()
        # revert
        st.session_state.qc_filter_questions = old_filter

        st.info(
            f"Profile: **{st.session_state.selected_profile}**  •  "
            f"Due in this selection: **{due_count}**"
        )

        # Pre-reading choice is forced in Section QC, but expose an override if you want:
        st.session_state.pre_read_mode = st.radio(
            "Before the questions, show…",
            [PRE_READ_SUMMARY, PRE_READ_FULL],
            index=0,
            horizontal=True,
            help="In Section Quick-Checks, you'll see the targeted section's info before questions."
        )

    else:
        # Classic Mode controls
        st.session_state.pre_read_mode = st.radio(
            "Before the questions, would you like to read…",
            PRE_READ_MODES,
            index=PRE_READ_MODES.index(st.session_state.pre_read_mode),
            horizontal=True,
        )

        # Load/refresh quiz data if selection or profile changed
        ensure_quiz_data_loaded()

        # Session size (Classic mode only)
        st.session_state.num_questions = st.number_input(
            "Questions per session",
            min_value=1,
            max_value=100,
            value=int(st.session_state.num_questions),
            step=1
        )

        due_count = active_or_deferred_due_count()
        st.info(
            f"Profile: **{st.session_state.selected_profile}**  •  "
            f"Due in this set: **{due_count}**"
        )

    # SAVE defaults so they stick next time
    save_settings(
        last_profile=st.session_state.selected_profile,
        last_num_questions=st.session_state.num_questions,
        pre_read_mode=st.session_state.pre_read_mode,
    )

    # Start button
    if st.button("Start Session ▶", type="primary"):
        # Reset per-session counters/state
        st.session_state.questions_answered = 0
        st.session_state.correct_count = 0
        st.session_state.session_started_at = datetime.now().isoformat()
        st.session_state.current_question_idx = None
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""
        st.session_state.pending_info_for_idx = None
        st.session_state.shown_info_keys = []
        st.session_state.last_info_key = None
        st.session_state.last_info_shown_at = {}

        # (Re)load quiz data for the chosen CSV/profile
        ensure_quiz_data_loaded()

        # Build (and COMMIT) quick-check filter if needed
        sel_chapter = None
        sel_section = None
        selected_qs = []
        if st.session_state.quiz_mode == "Section Quick-Checks" and df_tag is not None:
            chapters, sections_by_ch = get_tag_catalog(df_tag)
            # Retrieve current selections from widgets
            sel_chapter = st.session_state.get("Chapter")
            # Above line may not work depending on Streamlit's internal keys; safer to reuse local vars:
            # But we already computed preview_qs just above, so:
            selected_qs = preview_qs
            # Save these selections for reading & save keying
            # (We can rebuild from df_tag; here use last computed suiting values)
            # In case you need exact values, store them earlier into session_state.

        # Apply/clear filter
        st.session_state.qc_filter_questions = set(map(str, selected_qs)) if selected_qs else None

        # Fill active pool AFTER setting the filter
        ensure_active_pool()

        # ---------- session size by mode ----------
        if st.session_state.quiz_mode == "Section Quick-Checks":
            st.session_state.num_questions = min(int(st.session_state.qc_num_qs), len(selected_qs))

        # Decide first screen (and set chapter/section for reading targeting) — handled in the global code we inserted earlier
        # We'll set these from the last computed selection vars if available
        if st.session_state.quiz_mode == "Section Quick-Checks" and df_tag is not None:
            # capture the actual chosen chapter/section from widgets
            chapters, sections_by_ch = get_tag_catalog(df_tag)
            # Re-read current values safely:
            # we saved them above in local vars sel_chapter, sel_section, so reuse:
            st.session_state.current_chapter_title = sel_chapter
            st.session_state.current_section_title = sel_section

        # Decide first screen + reading payload & rerun
        payload, support, stem = load_reading_payload(st.session_state.selected_csv)
        st.session_state.reading_payload = payload
        st.session_state.reading_stem = stem

        if st.session_state.quiz_mode == "Section Quick-Checks":
            if payload and (payload.get("summary_bullets") or payload.get("reading") or payload.get("sections")):
                st.session_state.pre_read_mode = PRE_READ_SUMMARY if payload.get("summary_bullets") else PRE_READ_FULL
                st.session_state.screen = "reading"
            else:
                st.session_state.screen = "quiz"
        else:
            want_reading = (st.session_state.pre_read_mode != PRE_READ_NONE)
            if want_reading and payload and (support == "both" or (support == "summary_only" and st.session_state.pre_read_mode == PRE_READ_SUMMARY)):
                st.session_state.screen = "reading"
            else:
                st.session_state.screen = "quiz"

        st.rerun()


def reading_screen():
    """Show reading/summary; in Section QC, prefer only the targeted section."""
    payload = st.session_state.reading_payload
    stem = st.session_state.reading_stem or "Reading"
    if not payload:
        st.info("No reading material found. Starting the quiz.")
        st.session_state.screen = "quiz"
        st.rerun()
        return

    st.header(payload.get("title", stem).replace("_", " "))
    mode = st.session_state.pre_read_mode
    est = max(1, int(payload.get("estimated_read_time_sec", 60) / 60))
    st.caption(f"Estimated read time ~ {est} min")

    sections = payload.get("sections") or []
    has_sections = len(sections) > 0

    target_section = st.session_state.current_section_title if st.session_state.quiz_mode == "Section Quick-Checks" else None
    targeted_sections = []
    if target_section and has_sections:
        targeted_sections = [s for s in sections if (s.get("title") == target_section)]
        if not targeted_sections:
            # fallback: try case-insensitive match
            targeted_sections = [s for s in sections if s.get("title","").strip().lower() == target_section.strip().lower()]

    def _render_sections(sec_list):
        for sec in sec_list:
            if sec.get("title"):
                st.subheader(sec["title"])
            render_section(sec, payload)
            st.markdown("---")

    if st.session_state.quiz_mode == "Section Quick-Checks" and targeted_sections:
        # Always render the targeted section regardless of summary/full choice
        _render_sections(targeted_sections)
        # Show quick summary bullets too if available
        if payload.get("summary_bullets"):
            with st.expander("Quick chapter summary"):
                for b in payload["summary_bullets"]:
                    st.markdown(f"- {b}")
    else:
        # Classic behavior
        if mode == PRE_READ_FULL and payload.get("reading"):
            st.write(payload["reading"])
            if payload.get("summary_bullets"):
                with st.expander("Quick summary"):
                    for b in payload["summary_bullets"]:
                        st.markdown(f"- {b}")
            if has_sections:
                with st.expander("See sectioned version"):
                    _render_sections(sections)
        elif mode == PRE_READ_SUMMARY and payload.get("summary_bullets"):
            st.subheader("Summary")
            for b in payload["summary_bullets"]:
                st.markdown(f"- {b}")
            if has_sections or payload.get("reading"):
                with st.expander("Read the full section"):
                    if has_sections:
                        _render_sections(sections)
                    else:
                        st.write(payload["reading"])
        else:
            st.info("No reading/summary available. Starting the quiz.")
            st.session_state.screen = "quiz"
            st.rerun()
            return

    st.markdown("---")
    if st.button("Start Quiz ▶", type="primary"):
        st.session_state.screen = "quiz"
        st.rerun()



def quiz_screen():
    ensure_quiz_data_loaded()
    if not st.session_state.quiz_data:
        st.error("No questions loaded for this set.")
        if st.button("Back to start"):
            st.session_state.screen = "start"
            st.rerun()
        return

    # Progress header
    n = st.session_state.num_questions
    answered = st.session_state.questions_answered
    st.progress(answered / n if n else 0.0)
    st.caption(
        f"Question {min(answered + 1, n)} of {n}  •  "
        f"Profile: {st.session_state.selected_profile}"
    )

    # Keep the same question until Continue after Submit
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
    q = st.session_state.quiz_data[idx]
    info_key = q.get("info_key")

    # Decide whether to show a micro-info screen (NEW)
    should_show_info = False
    payload = st.session_state.reading_payload or {}
    notes = (payload.get("micro_notes") or {})
    has_note = info_key and (info_key in notes)

    if has_note:
        not_shown_before = info_key not in st.session_state.shown_info_keys
        concept_changed = (st.session_state.last_info_key != info_key)
        last_shown_iso = st.session_state.last_info_shown_at.get(info_key)
        too_recent = False
        min_repeat = int(notes[info_key].get("min_repeat_minutes") or DEFAULT_INFO_REPEAT_MIN)
        if last_shown_iso:
            try:
                last_dt = datetime.fromisoformat(last_shown_iso)
                too_recent = (datetime.now() - last_dt) < timedelta(minutes=min_repeat)
            except Exception:
                too_recent = False
        should_show_info = (not_shown_before or concept_changed) and (not too_recent)

    if should_show_info:
        st.session_state.pending_info_for_idx = idx
        st.session_state.screen = "info"
        st.rerun()
        return

    # Regular question render
    st.subheader(q["question"])
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

    due_count = active_or_deferred_due_count()
    st.caption(
        f"Currently due for review: **{due_count}** question(s).  •  "
        f"Profile: {st.session_state.selected_profile}"
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
elif screen == "reading":  # NEW
    reading_screen()
elif screen == "info":     # NEW
    info_screen()
elif screen == "quiz":
    quiz_screen()
else:
    summary_screen()


