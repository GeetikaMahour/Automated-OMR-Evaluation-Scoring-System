import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from PIL import Image

# -------------------------
# CREATE FOLDERS
# -------------------------
os.makedirs("./answer_keys", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# -------------------------
# LOAD ANSWER KEYS
# -------------------------
def load_answer_key(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

answer_key_A = load_answer_key("./answer_keys/setA.json")
answer_key_B = load_answer_key("./answer_keys/setB.json")

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def detect_bubbles(thresh_img):
    contours,_ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours=[]
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        area=w*h
        aspect=w/float(h)
        if 400<area<2500 and 0.8<=aspect<=1.2:
            bubble_contours.append(cnt)
    bubble_contours=sorted(bubble_contours,key=lambda c: (cv2.boundingRect(c)[1],cv2.boundingRect(c)[0]))
    return bubble_contours

def map_bubbles_to_answers(bubbles, thresh_img, options=4):
    answers = {}
    question_num = 1
    bubbles = sorted(bubbles, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    for i in range(0, len(bubbles), options):
        question_bubbles = bubbles[i:i + options]
        
        filled_intensities = []
        for cnt in question_bubbles:
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            intensity = cv2.mean(thresh_img, mask=mask)[0]
            filled_intensities.append(intensity)
            
        if filled_intensities:
            min_intensity = min(filled_intensities)
            if min_intensity < 200:
                selected_index = filled_intensities.index(min_intensity)
                answers[f"Q{question_num}"] = chr(65 + selected_index)
            else:
                answers[f"Q{question_num}"] = "-"
        
        question_num += 1
        
    return answers

def score_student(student_answers, answer_key):
    if not answer_key:
        return {"total": 0}
        
    scores = {}
    total = 0
    for subject, q_dict in answer_key.items():
        sub_score = 0
        for q, correct_ans in q_dict.items():
            student_ans = student_answers.get(q, "-")
            if "," in correct_ans:
                if student_ans in correct_ans.split(","):
                    sub_score += 1
            elif student_ans == correct_ans:
                sub_score += 1
        scores[subject] = sub_score
        total += sub_score
    scores["total"] = total
    return scores

# -------------------------
# STREAMLIT UI (Stylized)
# -------------------------

# Inject custom CSS to create the dashboard look
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0d122b;
        color: #f0f4f8;
        font-family: Arial, sans-serif;
    }

    /* Main container and content */
    .main .block-container {
        padding: 5rem 10rem;
        display: flex;
        flex-direction: column;
        align-items: center; /* Center everything horizontally */
    }

    /* Center the main title */
    h1 {
        text-align: center;
        width: 100%;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Set text color for labels */
    .st-emotion-cache-1g08969 label, .st-emotion-cache-13ln4j7 p {
        color: #aeb4bc !important;
    }
    
    /* Dropdown color and text (More aggressive selectors) */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
        background-color: #FFD700 !important; /* Golden color */
        color: #0d122b !important; /* Dark text for visibility */
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        fill: #0d122b !important;
    }

    /* Dropdown list when open */
    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: #1a233b !important;
        border: 1px solid #FFD700 !important;
    }
    div[data-baseweb="popover"] div[role="option"] {
        color: #f0f4f8 !important;
    }
    div[data-baseweb="popover"] div[role="option"]:hover {
        background-color: #2e3e5c !important;
    }


    /* File Uploader and Buttons */
    .st-emotion-cache-1m4j18p.e1nzilvr2 {
        color: #f0f4f8;
    }
    .st-emotion-cache-4oy39w.e1nzilvr4 { /* Download button */
        color: #0d122b;
        background-color: #ffcc00; /* Accent color */
        font-weight: bold;
    }
    .st-emotion-cache-4oy39w.e1nzilvr4:hover {
        background-color: #e5b800;
    }

    /* Success message */
    .st-emotion-cache-1b2q7o7.e1gf0gq52 {
        background-color: #1a233b;
        color: #aeb4bc;
    }

    /* Dataframe Styling */
    .st-emotion-cache-1q4wz14 {
        background-color: #1a233b;
        border-radius: 0.5rem;
        border: 1px solid #2e3e5c;
    }
    .st-emotion-cache-1q4wz14 table {
        color: #f0f4f8;
    }
    .st-emotion-cache-1q4wz14 th {
        color: #aeb4bc;
    }
    .st-emotion-cache-1q4wz14 tbody tr:hover {
        background-color: #2e3e5c;
    }
    
</style>
""", unsafe_allow_html=True)


# The UI structure
st.header("Automated OMR Evaluation & Scoring System")
st.write("Upload OMR sheets to automatically detect and score answers.")

set_option = st.selectbox(
    "Select the question paper set:",
    ("Set A", "Set B")
)

uploaded_files = st.file_uploader(
    "Upload OMR images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if set_option == "Set A":
        current_answer_key = answer_key_A
    else:
        current_answer_key = answer_key_B
        
    if current_answer_key is None:
        st.error(f"Could not find the answer key for {set_option}. Please ensure setB.json is in the answer_keys folder.")
    else:
        all_results = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert("RGB")
            img_cv = np.array(img)[:, :, ::-1].copy()
            thresh = preprocess_image(img_cv)
            bubbles = detect_bubbles(thresh)
            student_answers = map_bubbles_to_answers(bubbles, thresh)
            scores = score_student(student_answers, current_answer_key)
            
            result = {"student_id": uploaded_file.name.split(".")[0]}
            result.update(student_answers)
            result.update(scores)
            
            all_results.append(result)
        
        df_results = pd.DataFrame(all_results)
        st.success("✅ Scoring completed!")
        st.dataframe(df_results)
        
        csv_file = "./results/omr_scored_results.csv"
        df_results.to_csv(csv_file, index=False)
        st.download_button(
            "Download CSV",
            data=open(csv_file, "rb"),
            file_name="omr_scored_results.csv"
        )
