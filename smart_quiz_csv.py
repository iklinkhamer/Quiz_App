#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:12:34 2025

@author: Ilse Klinkhamer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Quiz with Start Screen & Sessions
- FIX: hold current question steady until Submit is pressed
- FIX: accept CSV answers as letters A/B/C/D (robust parsing)
"""

import streamlit as st
import pandas as pd
import random
import json
import time
import re
from datetime import datetime, timedelta

# ---------- Page Setup ----------
st.set_page_config(page_title="🧠 Smart Quiz", layout="centered")

# ---------- App State Defaults ----------
def init_state():
    defaults = {
        "screen": "start",              # "start" | "quiz" | "summary"
        "num_questions": 10,            # setting chosen on start screen
        "questions_answered": 0,        # session counter
        "correct_count": 0,             # session counter
        "session_started_at": None,     # ISO timestamp for summary
        "quiz_data": None,              # spaced repetition store
        "current_question_idx": None,   # <-- FIX: persist currently shown question
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.title("🧠 Smart Quiz")
st.caption("Questions return sooner if you miss them!")

LETTERS = ["A", "B", "C", "D"]

# ---------- Load questions from CSV ----------
@st.cache_data
def load_questions(csv_file="anatomy_physiology_mcqs.csv"):
    df = pd.read_csv(csv_file, sep=",", engine="python")
    df.columns = df.columns.str.strip()

    # Flexible column resolver
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
            "CSV is missing required columns. Expected headers like: "
            "question, optionA, optionB, optionC, optionD, answer"
        )

    now_iso = datetime.now().isoformat()
    questions_list = []
    for _, row in df.iterrows():
        options = [str(row[a_cols[0]]), str(row[a_cols[1]]), str(row[a_cols[2]]), str(row[a_cols[3]])]
        ans_raw = str(row[ans_col]).strip()

        # Robustly parse answer letter (A/B/C/D) or fall back to matching option text
        letter = None
        m = re.match(r"^\s*([A-D])\b", ans_raw, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
        elif ans_raw:
            # try to match by option text (case-insensitive)
            try:
                idx_guess = [o.lower().strip() for o in options].index(ans_raw.lower().strip())
                letter = LETTERS[idx_guess]
            except ValueError:
                letter = "A"  # final fallback

        correct_idx = LETTERS.index(letter) if letter in LETTERS else 0

        questions_list.append({
            "question": row[q_col],
            "options": options,
            "answer_letter": letter,         # store as letter
            "answer_idx": correct_idx,       # store as index for fast compare
            "interval": 1,                   # minutes until next review
            "next_time": now_iso             # eligible immediately
        })
    return questions_list

def ensure_quiz_data_loaded():
    if st.session_state.quiz_data is None:
        st.session_state.quiz_data = load_questions()
        load_progress(merge=True)

# ---------- Load/save progress (spaced repetition schedule) ----------
def save_progress():
    try:
        with open("quiz_progress.json", "w") as f:
            json.dump(st.session_state.quiz_data, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Couldn't save progress: {e}")

def load_progress(merge=False):
    try:
        with open("quiz_progress.json") as f:
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
        st.warning(f"Couldn't load prior progress: {e}")

# ---------- Helpers ----------
def due_indexes():
    now = datetime.now()
    return [i for i, q in enumerate(st.session_state.quiz_data) if datetime.fromisoformat(q["next_time"]) <= now]

def pick_next_question_idx():
    due = due_indexes()
    if not due:
        return None
    return random.choice(due)

def show_question(idx):
    q = st.session_state.quiz_data[idx]

    # Radio returns the INDEX (0..3); we format label as "A. text"
    opts_idx = list(range(4))
    def fmt(i): return f"{LETTERS[i]}. {q['options'][i]}"
    choice_key = f"choice_q{idx}"   # stable key while this exact question is shown
    selected_idx = st.radio("Choose an answer:", opts_idx, format_func=fmt, key=choice_key)

    submitted = st.button("Submit", type="primary", key=f"submit_q{idx}")
    if submitted:
        correct_idx = q["answer_idx"]
        correct_letter = LETTERS[correct_idx]
        correct_text = q["options"][correct_idx]

        if selected_idx == correct_idx:
            st.success("✅ Correct!")
            st.session_state.correct_count += 1
            q["interval"] = max(1, int(q["interval"] * 2))   # grow interval when correct
        else:
            st.error(f"❌ Wrong! The correct answer was {correct_letter}. {correct_text}")
            q["interval"] = 1                                # reset if wrong

        q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()
        st.session_state.questions_answered += 1
        save_progress()

        # Clear current question so the next rerun picks a NEW one (or ends)
        st.session_state.current_question_idx = None

        # Move along within this session
        if st.session_state.questions_answered >= st.session_state.num_questions:
            st.session_state.screen = "summary"
        # brief pause so feedback is visible before rerun
        time.sleep(0.5)
        st.rerun()

# ---------- Screens ----------
def start_screen():
    ensure_quiz_data_loaded()

    st.header("Start a practice session")
    st.write("Set how many questions you'd like to answer in this session. Questions you miss will return sooner.")

    currently_due = len(due_indexes())
    st.info(f"Questions currently due: **{currently_due}**")

    st.session_state.num_questions = st.number_input(
        "Questions per session",
        min_value=1,
        max_value=100,
        value=st.session_state.num_questions,
        step=1
    )

    if st.button("Start Session", type="primary"):
        st.session_state.questions_answered = 0
        st.session_state.correct_count = 0
        st.session_state.session_started_at = datetime.now().isoformat()
        st.session_state.current_question_idx = None
        st.session_state.screen = "quiz"
        st.rerun()

def quiz_screen():
    ensure_quiz_data_loaded()

    # Session progress header
    n = st.session_state.num_questions
    answered = st.session_state.questions_answered
    st.progress(answered / n if n else 0.0)
    st.caption(f"Question {answered + 1} of {n}")

    # Keep the same question until Submit
    if st.session_state.current_question_idx is None:
        idx = pick_next_question_idx()
        if idx is None:
            st.success("🎉 All due questions are up to date right now.")
            if answered > 0:
                st.write("That’s the end of your session for now.")
                st.session_state.screen = "summary"
                st.rerun()
            else:
                if st.button("Back to start"):
                    st.session_state.screen = "start"
                    st.rerun()
            return
        st.session_state.current_question_idx = idx

    # Show the persistent current question
    idx = st.session_state.current_question_idx
    q = st.session_state.quiz_data[idx]
    st.subheader(q["question"])
    show_question(idx)

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
    st.caption(f"Currently due for review: **{due_count}** question(s).")

    if st.button("Return to Start", type="primary"):
        st.session_state.screen = "start"
        st.session_state.current_question_idx = None
        st.rerun()

# ---------- Router ----------
screen = st.session_state.screen
if screen == "start":
    start_screen()
elif screen == "quiz":
    quiz_screen()
else:
    summary_screen()

