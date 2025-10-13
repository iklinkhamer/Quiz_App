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

def progress_path_for(csv_file: str, profile_name: str | None):
    """Unique progress file per dataset & profile; None if practice-without-profile."""
    if not profile_name or profile_name == NO_PROFILE_LABEL:
        return None
    base = os.path.splitext(os.path.basename(csv_file))[0]
    safe_ds = safe_name(base)
    safe_prof = safe_name(profile_name)
    return f"quiz_progress__{safe_ds}__{safe_prof}.json"

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
        # NEW: pre-reading
        "pre_read_mode": settings.get("pre_read_mode", PRE_READ_NONE),
        "reading_payload": None,        # loaded reading JSON dict
        "reading_stem": None,           # e.g. "ch01_Organisation_of_the_Body"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Global title (cute!)
st.title("🐝 QuizzyBee")
st.caption("Questions return sooner if you miss them!")

LETTERS = ["A", "B", "C", "D"]

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

        questions_list.append({
            "question": row[q_col],
            "options": options,
            "answer_letter": letter,
            "answer_idx": correct_idx,
            "interval": 1,
            "next_time": now_iso
        })
    return questions_list

# ---------- Reading loader (NEW) ----------
def extract_chapter_stem_from_csv(csv_path: str) -> str | None:
    """
    Try to get a stem like 'ch01_Organisation_of_the_Body' from the CSV filename.
    Works with names like:
      mcqs_ch01_Organisation_of_the_Body.csv
      ch01_Organisation_of_the_Body.csv
    """
    name = Path(csv_path).stem
    m = re.search(r"(ch\d{2}_[A-Za-z0-9_]+)$", name)
    if m:
        return m.group(1)
    m = re.search(r"(ch\d{2}_.+)", name)  # fallback catch-all
    return m.group(1) if m else None

def find_reading_json_for_csv(csv_path: str) -> str | None:
    """
    Look for a reading file near the CSV with the same stem:
      <dir>/<stem>.reading.json
      <dir>/<stem>.json
    Also supports a nested layout:
      <dir>/<stem>/reading.json
    """
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
    'payload' has keys: title, reading, summary_bullets, estimated_read_time_sec.
    'mode_support' is one of: "both", "summary_only" (if reading missing), or None if not found.
    """
    stem = extract_chapter_stem_from_csv(csv_path)
    json_path = find_reading_json_for_csv(csv_path)
    if not json_path:
        return None, None, stem
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Normalize minimal shape
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

def ensure_quiz_data_loaded():
    """(Re)load quiz_data if dataset or profile selection changed."""
    dataset_changed = (
        st.session_state.quiz_data is None or
        st.session_state.selected_csv != st.session_state.last_loaded_csv or
        st.session_state.selected_profile != st.session_state.last_loaded_profile
    )
    if dataset_changed and st.session_state.selected_csv:
        st.session_state.quiz_data = load_questions(st.session_state.selected_csv)
        st.session_state.last_loaded_csv = st.session_state.selected_csv
        st.session_state.last_loaded_profile = st.session_state.selected_profile
        # Merge saved SRS schedule only if we are in a saving profile
        if st.session_state.selected_profile != NO_PROFILE_LABEL:
            load_progress(merge=True)

# ---------- Load/save/reset progress (per dataset & profile) ----------
def save_progress():
    path = progress_path_for(st.session_state.selected_csv or "default", st.session_state.selected_profile)
    if path is None:
        return  # no-save mode
    try:
        with open(path, "w") as f:
            json.dump(st.session_state.quiz_data, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Couldn't save progress to '{path}': {e}")

def load_progress(merge=False):
    path = progress_path_for(st.session_state.selected_csv or "default", st.session_state.selected_profile)
    if path is None:
        return  # no-save mode
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
    """Delete the progress file for the current (dataset, profile)."""
    path = progress_path_for(st.session_state.selected_csv or "default", st.session_state.selected_profile)
    if path and os.path.exists(path):
        try:
            os.remove(path)
            st.success("Progress reset for this set.")
        except Exception as e:
            st.warning(f"Couldn't delete progress file: {e}")
    # also reset in-memory intervals to defaults
    if st.session_state.quiz_data:
        now_iso = datetime.now().isoformat()
        for q in st.session_state.quiz_data:
            q["interval"] = 1
            q["next_time"] = now_iso

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
    """Call after pressing Continue or auto-advancing."""
    n = st.session_state.num_questions
    if st.session_state.questions_answered >= n:
        st.session_state.screen = "summary"
    else:
        st.session_state.current_question_idx = None  # pick a fresh due question
    st.rerun()

def show_question(idx):
    q = st.session_state.quiz_data[idx]
    opts_idx = list(range(4))
    def fmt(i): return f"{LETTERS[i]}. {q['options'][i]}"

    # If we're awaiting the user's Continue click, lock inputs and just show feedback.
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

    # Normal interactive state before Submit
    selected_idx = st.radio("Choose an answer:", opts_idx, format_func=fmt, key=f"choice_q{idx}")
    submitted = st.button("Submit", type="primary", key=f"submit_q{idx}")

    if submitted:
        correct_idx = q["answer_idx"]
        correct_letter = LETTERS[correct_idx]
        correct_text = q["options"][correct_idx]

        if selected_idx == correct_idx:
            # ✅ correct → brief green feedback, then auto-advance
            st.session_state.correct_count += 1
            q["interval"] = max(1, int(q["interval"] * 2))
            st.session_state.last_was_correct = True
            st.session_state.last_feedback = "✅ Correct!"

            q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()
            st.session_state.questions_answered += 1
            save_progress()

            st.success("✅ Correct!")
            time.sleep(0.7)

            st.session_state.awaiting_continue = False
            st.session_state.current_question_idx = None
            finish_or_continue_session()
            return
        else:
            # ❌ incorrect → pause and show correction until Continue
            q["interval"] = 1
            st.session_state.last_was_correct = False
            st.session_state.last_feedback = f"❌ Wrong! The correct answer was {correct_letter}. {correct_text}"

            q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()
            st.session_state.questions_answered += 1
            save_progress()

            st.session_state.awaiting_continue = True
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

    # Default dataset selection
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

    # NEW: Pre-reading selection (persisted)
    st.session_state.pre_read_mode = st.radio(
        "Before the questions, would you like to read…",
        PRE_READ_MODES,
        index=PRE_READ_MODES.index(st.session_state.pre_read_mode),
        horizontal=True,
        help="You can choose to read the full section, a short summary, or skip reading."
    )

    # Load/refresh quiz data if selection or profile changed
    ensure_quiz_data_loaded()

    # Info about due items for this set (for this profile if saving)
    currently_due = len(due_indexes()) if st.session_state.quiz_data else 0
    st.info(
        f"Profile: **{st.session_state.selected_profile}**  •  "
        f"Due in this set: **{currently_due}**"
    )

    # Reset progress (only if saving with a profile) with confirmation
    if st.session_state.selected_profile != NO_PROFILE_LABEL:
        if not st.session_state.confirm_reset:
            if st.button("Reset progress for this set"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("Are you sure? This will erase all saved intervals and due times for this profile and set.")
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

    # SAVE defaults so they stick next time
    save_settings(
        last_profile=st.session_state.selected_profile,
        last_num_questions=st.session_state.num_questions,
        pre_read_mode=st.session_state.pre_read_mode,
    )

    if st.button("Start Session ▶", type="primary"):
        st.session_state.questions_answered = 0
        st.session_state.correct_count = 0
        st.session_state.session_started_at = datetime.now().isoformat()
        st.session_state.current_question_idx = None
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""

        # NEW: If reading is requested and available, go to reading screen; else quiz
        payload, support, stem = load_reading_payload(st.session_state.selected_csv)
        st.session_state.reading_payload = payload
        st.session_state.reading_stem = stem
        want_reading = (st.session_state.pre_read_mode != PRE_READ_NONE)

        if want_reading and payload and (support == "both" or (support == "summary_only" and st.session_state.pre_read_mode == PRE_READ_SUMMARY)):
            st.session_state.screen = "reading"
        else:
            st.session_state.screen = "quiz"

        st.rerun()

def reading_screen():
    """NEW: Show reading or summary, then proceed to quiz."""
    payload = st.session_state.reading_payload
    stem = st.session_state.reading_stem or "Reading"
    if not payload:
        st.info("No reading material found for this chapter. Starting the quiz.")
        st.session_state.screen = "quiz"
        st.rerun()
        return

    st.header(payload.get("title", stem).replace("_", " "))

    mode = st.session_state.pre_read_mode
    if mode == PRE_READ_FULL and payload.get("reading"):
        # Full reading
        est = payload.get("estimated_read_time_sec", 60)
        st.caption(f"Estimated read time ~ {max(1, int(est/60))} min")
        st.write(payload["reading"])
        if payload.get("summary_bullets"):
            with st.expander("Quick summary"):
                for b in payload["summary_bullets"]:
                    st.markdown(f"- {b}")
    elif mode == PRE_READ_SUMMARY and payload.get("summary_bullets"):
        # Summary only
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
        idx = pick_next_question_idx()
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
    
    # allow returning to start during quiz
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
elif screen == "quiz":
    quiz_screen()
else:
    summary_screen()

