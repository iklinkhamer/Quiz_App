#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:27:52 2025

@author: no1
"""
import pandas as pd
import random
import json
import time
from datetime import datetime, timedelta

def load_questions(csv_file="questions.csv"):
    df = pd.read_csv(csv_file, sep=";")
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

questions = load_questions()
a = 1