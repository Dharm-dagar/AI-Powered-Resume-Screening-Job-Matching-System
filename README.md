# 🤖 AI-Powered Resume Screening & Job Matching System

An intelligent web-based system that automates the screening of resumes and intelligently matches candidates to relevant job postings using advanced Natural Language Processing (NLP) techniques. Built with a scalable frontend in **React.js** and a powerful backend powered by **FastAPI** and **machine learning models**.

---

## 🚀 Features

- 📄 Upload resumes (PDF/DOCX)
- 📊 Extract and analyze candidate skills using NLP
- 🧠 Match resumes with job descriptions using similarity scoring
- 📝 Admin/HR dashboard to view candidate-job fit scores
- ⚡ FastAPI-powered backend for ML inference and data handling
- 🧾 JSON-based API for communication between frontend and backend

---

## 🧠 AI & ML Capabilities

- ✅ Resume Parsing (text extraction & cleaning)
- ✅ Named Entity Recognition (Skills, Education, Experience)
- ✅ Cosine Similarity for job matching
- ✅ TF-IDF / BERT vectorization (optional, customizable)
- ✅ Skill gap detection and match percentage calculation

---

## 🛠️ Tech Stack

| Category      | Tech Stack                     |
|---------------|--------------------------------|
| Frontend      | React.js, Tailwind CSS         |
| Backend       | FastAPI, Python                |
| ML/NLP        | scikit-learn, spaCy, NLTK      |
| Database (optional) | MongoDB / SQLite (for job/resume storage) |
| State Management | Zustand / Redux (Frontend)   |

---

## 📂 Folder Structure

AI-Powered-Resume-Screening-Job-Matching-System/
├── backend/
│ ├── main.py
│ ├── model/
│ └── utils/
├── frontend/
│ ├── src/
│ ├── components/
│ └── pages/
├── public/
├── .gitignore
├── README.md
└── package.json


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Dharm-dagar/AI-Powered-Resume-Screening-Job-Matching-System.git
cd AI-Powered-Resume-Screening-Job-Matching-System

cd backend
pip install -r requirements.txt
uvicorn main:app --reload

cd frontend
npm install
npm run dev

