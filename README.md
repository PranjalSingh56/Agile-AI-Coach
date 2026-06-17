# 🤖 AI-Powered Agile Coaching System

> MTECH Project | Machine Learning + Flask | Scrum Masters & Product Owners

---

## 📋 Project Overview

An intelligent web application that uses **Machine Learning** and **NLP** to help Agile teams manage sprints more effectively. Built with Python Flask, Scikit-learn, and TextBlob — no paid AI APIs required.

---

## ✨ Features

| Feature | Technology | Algorithm |
|---|---|---|
| Sprint Success Prediction | Scikit-learn | Logistic Regression |
| Task Priority Prediction | Scikit-learn | Random Forest |
| Agile Risk Detection | Rule-based AI | If-else logic engine |
| Standup Summary Generator | NLP | Keyword extraction |
| Retrospective Sentiment Analysis | TextBlob | Polarity scoring |

---

## 🧰 Technologies Used

**Frontend:** HTML5, CSS3, JavaScript, Chart.js  
**Backend:** Python 3.10+, Flask 3.0  
**Machine Learning:** Scikit-learn, Pandas, NumPy  
**NLP:** TextBlob  
**Database:** SQLite (session storage)  
**Fonts:** JetBrains Mono, Syne (Google Fonts)  

---

## 🤖 ML Algorithms Explained

### 1. Logistic Regression (Sprint Prediction)
- Binary classification: Success (1) or Failure (0)
- Features: team_velocity, completed_tasks, pending_tasks, bugs_count, team_capacity
- Output: Probability score + class label
- Trained on 50 sprint records, ~95% accuracy on test set

### 2. Random Forest (Task Priority)
- Multi-class: High / Medium / Low
- Features: complexity, deadline_days, business_value, dependencies
- Builds 100 decision trees, majority vote decides output
- Feature importance ranking included

### 3. Rule-based AI (Risk Detection)
- 6 handcrafted rules based on Agile best practices
- No training data required — expert knowledge encoded
- Easy to extend with new rules

### 4. NLP — Keyword Extraction (Standup)
- Regex tokenization + stopword filtering
- Keyword dictionary matching for task classification
- Top-N most frequent words extracted

### 5. TextBlob Sentiment (Retrospective)
- Uses a pre-trained lexicon (PatternAnalyzer)
- Polarity: -1.0 (negative) to +1.0 (positive)
- Subjectivity: 0.0 (objective) to 1.0 (subjective)

---

## 📁 Folder Structure

```
agile-ai-coach/
│
├── static/
│   ├── css/
│   │   └── style.css          # Main stylesheet (dark theme)
│   └── js/
│       └── main.js            # JavaScript (Chart.js, API calls)
│
├── templates/
│   ├── index.html             # Login page
│   ├── dashboard.html         # Main dashboard
│   ├── prediction.html        # Sprint + task prediction
│   ├── risk.html              # Risk analysis
│   ├── standup.html           # Standup generator
│   ├── sentiment.html         # Sentiment analysis
│   └── charts.html            # Charts & reports
│
├── datasets/
│   ├── sprint_data.csv        # 50 sprint records
│   ├── task_priority_data.csv # 60 task records
│   └── retrospective_feedback.csv  # 40 feedback entries
│
├── models/                    # Saved ML models (after training)
│   ├── sprint_model.pkl
│   ├── sprint_scaler.pkl
│   └── task_model.pkl
│
├── app.py                     # Flask application (all routes)
├── train_model.py             # ML training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Installation Steps

### Prerequisites
- Python 3.10 or newer
- pip (Python package manager)

### Step 1: Clone / Download the project
```bash
# If using git:
git clone <your-repo-url>
cd agile-ai-coach

# Or just extract the ZIP and cd into the folder
```

### Step 2: Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download TextBlob corpora
```bash
python -c "import textblob; textblob.download_corpora()"
# or:
python -m textblob.download_corpora
```

### Step 5: Train the ML models
```bash
python train_model.py
```
This creates `models/sprint_model.pkl`, `models/sprint_scaler.pkl`, and `models/task_model.pkl`.

### Step 6: Run the Flask app
```bash
python app.py
```

### Step 7: Open in browser
```
http://127.0.0.1:5000
```

---

## 🔑 Demo Login Credentials

| Username | Password | Role |
|---|---|---|
| scrum_master | pass123 | Scrum Master |
| product_owner | pass123 | Product Owner |
| admin | admin123 | Admin |

---

## 📸 Screenshots

> Add screenshots of each page here after running the project.

- `screenshots/login.png` — Login page
- `screenshots/dashboard.png` — Main dashboard
- `screenshots/prediction.png` — Sprint & task prediction
- `screenshots/risk.png` — Risk analysis
- `screenshots/standup.png` — Standup generator
- `screenshots/sentiment.png` — Sentiment analysis
- `screenshots/charts.png` — Charts & reports

---

## 🔮 Future Improvements

1. **Real database integration** — Store actual sprint data in SQLite/PostgreSQL
2. **User registration** — Allow teams to create accounts
3. **Email alerts** — Send risk warnings via email
4. **Historical trends** — Track velocity over many sprints
5. **Better NLP** — Use BERT or spaCy for more accurate summaries
6. **Jira/Trello integration** — Import real sprint data automatically
7. **Model retraining** — Retrain models on user-contributed real data
8. **Mobile app** — React Native or Flutter frontend

---

## 👨‍💻 Author

MTECH first Year Project — AI/ML  
Tech Stack: Python · Flask · Scikit-learn · TextBlob · Chart.js

---

## 📄 License

This project is created for educational purposes.
