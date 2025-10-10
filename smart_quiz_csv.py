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
- NEW: after Submit, show correction and wait for a Continue click
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
        "num_questions": 10,
        "questions_answered": 0,
        "correct_count": 0,
        "session_started_at": None,
        "quiz_data": None,
        "current_question_idx": None,   # persist currently shown question
        # NEW: submission/feedback gating
        "awaiting_continue": False,
        "last_was_correct": None,
        "last_feedback": "",            # text shown after submit
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

def ensure_quiz_data_loaded():
    if st.session_state.quiz_data is None:
        st.session_state.quiz_data = load_questions()
        load_progress(merge=True)

# ---------- Load/save progress ----------
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

def finish_or_continue_session():
    """Call after pressing Continue."""
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
        # Disabled radio echoing their last choice (kept by Streamlit via the key)
        st.radio("Choose an answer:", opts_idx, format_func=fmt,
                 key=f"choice_q{idx}", disabled=True)

        # Feedback panel
        if st.session_state.last_was_correct:
            st.success(st.session_state.last_feedback)
        else:
            st.error(st.session_state.last_feedback)

        # Continue button to move on
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
            # ✅ correct → show green feedback briefly, then auto-advance
            st.session_state.correct_count += 1
            q["interval"] = max(1, int(q["interval"] * 2))
            st.session_state.last_was_correct = True
            st.session_state.last_feedback = "✅ Correct!"
    
            # update spaced repetition data
            q["next_time"] = (datetime.now() + timedelta(minutes=q["interval"])).isoformat()
            st.session_state.questions_answered += 1
            save_progress()
    
            # show feedback briefly before moving on
            st.success("✅ Correct!")
            time.sleep(1.2)  # <-- short pause for feedback visibility
    
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
    
            # now pause: wait for Continue
            st.session_state.awaiting_continue = True
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
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""
        st.session_state.screen = "quiz"
        st.rerun()

def quiz_screen():
    ensure_quiz_data_loaded()

    # Progress header
    n = st.session_state.num_questions
    answered = st.session_state.questions_answered
    st.progress(answered / n if n else 0.0)
    st.caption(f"Question {min(answered + 1, n)} of {n}")

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
        st.session_state.awaiting_continue = False
        st.session_state.last_was_correct = None
        st.session_state.last_feedback = ""
        st.rerun()

# ---------- Router ----------
screen = st.session_state.screen
if screen == "start":
    start_screen()
elif screen == "quiz":
    quiz_screen()
else:
    summary_screen()

