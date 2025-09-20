# Automated OMR Evaluation & Scoring System

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-orange)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-brightgreen)](https://opencv.org/)

Automated evaluation and scoring of OMR (Optical Mark Recognition) sheets using *Python, **OpenCV, and **Streamlit*. This tool helps evaluators quickly score exams, generate per-subject scores, and export results, reducing manual effort and errors.

---

## 🔹 Features

- Detects filled bubbles in OMR sheets accurately using OpenCV
- Scores answers per subject and calculates total marks
- Generates downloadable CSV results
- Evaluator-friendly web interface with Streamlit
- Supports multiple sheet versions
- Maintains audit trail of detected answers

---

## 🛠 Tech Stack

- *Python 3.x*  
- *OpenCV* – Image processing and bubble detection  
- *NumPy & Pandas* – Data manipulation and scoring  
- *Streamlit* – Web dashboard for evaluation  
- *Pillow* – Image handling  

---

## 📂 Project Structure

Automated_OMR_Evaluation_Scoring_System/
│
├─ data/ # Sample OMR images for testing
├─ results/ # Generated CSV output (ignored in GitHub)
├─ answer_keys/ # Answer key JSON file
│ └─ setA.json
├─ omr_dashboard.py # Main Streamlit application
├─ README.md # Project description
├─ .gitignore # Files/folders to ignore in Git
## ⚡ How to Run

1. *Install dependencies*

```bash
pip install streamlit opencv-python-headless numpy pandas pillow
Place OMR images in the data/ folder.

Run the Streamlit app

bash
Copy code
streamlit run omr_dashboard.py
Upload images via the web interface to:

Detect answers

Score each student per subject

Download results as CSV

📝 Notes
results/ folder stores output CSV files

answer_keys/setA.json contains the correct answers for scoring

Only a few sample images need to be uploaded to GitHub (real exam sheets can be excluded)
