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
Created on Fri Oct 10 14:12:34 2025

@author: Ilse Klinkhamer
"""

import streamlit as st
import pandas as pd
import random
import json
import time
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
        "last_question_idx": None,      # index of the most recent asked question
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.title("🧠 Smart Quiz")
st.caption("Questions return sooner if you miss them!")

# ---------- Load questions from CSV ----------
@st.cache_data
def load_questions(csv_file="anatomy_physiology_mcqs.csv"):
    df = pd.read_csv(csv_file, sep=",", engine="python")
    df.columns = df.columns.str.strip()

    # Try to be forgiving about column names
    # Preferred: question, optionA..D, answer
    # Fallbacks from other exports: answerA..D, correct / correct answer
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

    questions_list = []
    now_iso = datetime.now().isoformat()
    for _, row in df.iterrows():
        options = [row[a_cols[0]], row[a_cols[1]], row[a_cols[2]], row[a_cols[3]]]
        questions_list.append({
            "question": row[q_col],
            "options": options,
            "answer": row[ans_col],
            "interval": 1,              # minutes until next review
            "next_time": now_iso        # eligible immediately
        })
    return questions_list

def ensure_quiz_data_loaded():
    if st.session_state.quiz_data is None:
        # First time: load from CSV
        st.session_state.quiz_data = load_questions()
        # Try to merge any prior progress
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
            # Merge by (question text) identity; keep current CSV questions, but
            # restore known intervals/next_time from saved store when matched.
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
def due_questions():
    now = datetime.now()
    return [i for i, q in enumerate(st.session_state.quiz_data) if datetime.fromisoformat(q["next_time"]) <= now]

def pick_next_question_idx():
    due = due_questions()
    if not due:
        return None
    return random.choice(due)

def ask_question(idx):
    q = st.session_state.quiz_data[idx]
    st.subheader(q["question"])
    choice = st.radio("Choose an answer:", q["options"], key=f"choice_{idx}_{st.session_state.questions_answered}")
    submitted = st.button("Submit", type="primary")
    if submitted:
        if choice == q["answer"]:
            st.success("✅ Correct!")
            st.session_state.correct_count += 1
            q["interval"] = max(1, int(q["interval"] * 2))   # grow interval when correct
        else:
            st.error(f"❌ Wrong! The correct answer was {q['answer']}.")
            q["interval"] = 1                                # reset if wrong

        q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()
        st.session_state.questions_answered += 1
        st.session_state.last_question_idx = idx
        save_progress()

        # Move along within this session
        if st.session_state.questions_answered >= st.session_state.num_questions:
            st.session_state.screen = "summary"
            st.rerun()
        else:
            # continue to next question (if any due)
            time.sleep(0.5)
            st.rerun()

# ---------- Screens ----------
def start_screen():
    ensure_quiz_data_loaded()

    st.header("Start a practice session")
    st.write("Set how many questions you'd like to answer in this session. Questions you miss will return sooner.")

    # Count currently due
    currently_due = len(due_questions())
    st.info(f"Questions currently due: **{currently_due}**")

    st.session_state.num_questions = st.number_input(
        "Questions per session",
        min_value=1,
        max_value=100,
        value=st.session_state.num_questions,
        step=1
    )

    start_btn = st.button("Start Session", type="primary")
    if start_btn:
        st.session_state.questions_answered = 0
        st.session_state.correct_count = 0
        st.session_state.session_started_at = datetime.now().isoformat()

        # If nothing is due, let them know; still allow starting (it will end immediately)
        st.session_state.screen = "quiz"
        st.rerun()

def quiz_screen():
    ensure_quiz_data_loaded()

    # Session progress header
    n = st.session_state.num_questions
    answered = st.session_state.questions_answered
    st.progress(answered / n if n else 0.0)
    st.caption(f"Question {answered + 1} of {n}")

    idx = pick_next_question_idx()
    if idx is None:
        st.success("🎉 All due questions are up to date right now.")
        if st.session_state.questions_answered > 0:
            st.write("That’s the end of your session for now.")
            st.session_state.screen = "summary"
            st.rerun()
        else:
            # Nothing due at all and session hasn't asked anything
            if st.button("Back to start"):
                st.session_state.screen = "start"
                st.rerun()
        return

    ask_question(idx)

def summary_screen():
    total = st.session_state.questions_answered
    correct = st.session_state.correct_count
    if total > 0:
        accuracy = 100.0 * correct / total
        st.header("Session Summary")
        st.metric("Questions answered", total)
        st.metric("Correct answers", correct)
        st.metric("Accuracy", f"{accuracy:.1f}%")
    else:
        st.header("Session Summary")
        st.write("No questions were due this time. Nice and caught up!")

    # Quick next-due preview
    due_count = len(due_questions())
    st.caption(f"Currently due for review: **{due_count}** question(s).")

    if st.button("Return to Start", type="primary"):
        st.session_state.screen = "start"
        st.rerun()

# ---------- Router ----------
screen = st.session_state.screen
if screen == "start":
    start_screen()
elif screen == "quiz":
    quiz_screen()
else:
    summary_screen()
