#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:12:34 2025

@author: Ilse Klinkhamer
"""

import streamlit as st
import pandas as pd
import random
import json
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="🧠 Smart Quiz", layout="centered")

st.title("🧠 Smart Quiz")
st.caption("Questions return sooner if you miss them!")

# ---- Load questions from CSV ----
@st.cache_data
def load_questions(csv_file="questions.csv"):
    df = pd.read_csv(csv_file, sep=",")
    df.columns = df.columns.str.strip()
    questions_list = []
    for _, row in df.iterrows():
        questions_list.append({
            "question": row["question"],
            "options": [row["optionA"], row["optionB"], row["optionC"], row["optionD"]],
            "answer": row["answer"],
            "interval": 1,
            "next_time": datetime.now().isoformat()
        })
    return questions_list

# ---- Load/save state ----
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = load_questions()

def save_progress():
    with open("quiz_progress.json", "w") as f:
        json.dump(st.session_state.quiz_data, f, indent=2, default=str)

def load_progress():
    try:
        with open("quiz_progress.json") as f:
            st.session_state.quiz_data = json.load(f)
    except FileNotFoundError:
        pass

load_progress()

# ---- Pick next question ----
def get_next_question():
    now = datetime.now()
    due = [q for q in st.session_state.quiz_data if datetime.fromisoformat(q["next_time"]) <= now]
    if not due:
        st.success("🎉 All questions are up to date! Come back later.")
        st.stop()
    return random.choice(due)

# ---- Main quiz logic ----
question = get_next_question()
st.subheader(question["question"])
choice = st.radio("Choose an answer:", question["options"])

if st.button("Submit"):
    if choice == question["answer"]:
        st.success("✅ Correct!")
        question["interval"] *= 2  # delay increases if correct
    else:
        st.error(f"❌ Wrong! The correct answer was {question['answer']}.")
        question["interval"] = 1  # reset interval if wrong

    # Update next appearance time
    question["next_time"] = (datetime.now() + timedelta(minutes=question["interval"])).isoformat()
    save_progress()
    time.sleep(1)
    st.rerun()
