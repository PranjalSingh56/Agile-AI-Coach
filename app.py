"""
app.py
======
Flask Backend for AI-Powered Agile Coaching System
Routes: Login, Dashboard, Sprint Prediction, Task Priority,
        Risk Analysis, Standup Summary, Sentiment Analysis
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle, os, re
import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import Counter

app = Flask(__name__)
app.secret_key = "agile_ai_coach_secret_2024"

# ─── Load ML Models ───────────────────────────────────────────────────────────
MODELS_DIR = "models"

def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

sprint_model  = load_model("sprint_model.pkl")
sprint_scaler = load_model("sprint_scaler.pkl")
task_model    = load_model("task_model.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Redirect root to login page."""
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Simple login with hardcoded demo credentials."""
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        # Demo credentials — easy to explain in viva
        valid_users = {
            "scrum_master": "pass123",
            "product_owner": "pass123",
            "admin": "admin123",
        }
        if valid_users.get(username) == password:
            session["user"]     = username
            session["role"]     = username.replace("_", " ").title()
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password. Try: scrum_master / pass123"
    return render_template("index.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def login_required(f):
    """Simple decorator to protect routes."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard")
@login_required
def dashboard():
    """Main dashboard with summary stats."""
    stats = {
        "active_sprints"  : 3,
        "total_tasks"     : 47,
        "completed_tasks" : 29,
        "open_risks"      : 4,
        "team_velocity"   : 52,
        "sprint_health"   : "Good",
    }
    return render_template("dashboard.html",
                           user=session["user"],
                           role=session["role"],
                           stats=stats)


# ══════════════════════════════════════════════════════════════════════════════
# SPRINT SUCCESS PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    """Sprint success prediction using Logistic Regression."""
    result = None
    error  = None

    if request.method == "POST":
        try:
            # Collect form data
            velocity   = float(request.form["team_velocity"])
            completed  = float(request.form["completed_tasks"])
            pending    = float(request.form["pending_tasks"])
            bugs       = float(request.form["bugs_count"])
            capacity   = float(request.form["team_capacity"])

            if sprint_model is None or sprint_scaler is None:
                error = "Model not trained yet. Run: python train_model.py"
            else:
                features       = np.array([[velocity, completed, pending, bugs, capacity]])
                features_scaled = sprint_scaler.transform(features)
                prediction_val = sprint_model.predict(features_scaled)[0]
                probability    = sprint_model.predict_proba(features_scaled)[0]

                result = {
                    "prediction"   : "SUCCESS ✓" if prediction_val == 1 else "FAILURE ✗",
                    "success_class": "success" if prediction_val == 1 else "danger",
                    "confidence"   : f"{max(probability)*100:.1f}%",
                    "success_prob" : f"{probability[1]*100:.1f}%",
                    "failure_prob" : f"{probability[0]*100:.1f}%",
                    "inputs"       : {
                        "Team Velocity"   : velocity,
                        "Completed Tasks" : completed,
                        "Pending Tasks"   : pending,
                        "Bugs Count"      : bugs,
                        "Team Capacity %" : capacity,
                    },
                }
        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template("prediction.html",
                           user=session["user"],
                           role=session["role"],
                           result=result,
                           error=error)


# ══════════════════════════════════════════════════════════════════════════════
# TASK PRIORITY PREDICTION  (API endpoint called by JS)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/task_priority", methods=["POST"])
@login_required
def task_priority():
    """Predict task priority: High / Medium / Low."""
    try:
        data         = request.get_json()
        complexity   = float(data["complexity"])
        deadline     = float(data["deadline_days"])
        biz_value    = float(data["business_value"])
        dependencies = float(data["dependencies"])

        if task_model is None:
            return jsonify({"error": "Model not trained. Run train_model.py"}), 500

        features = pd.DataFrame([[complexity, deadline, biz_value, dependencies]],
                        columns=["complexity", "deadline_days", 
                                 "business_value", "dependencies"])
        pred     = task_model.predict(features)[0]
        proba    = task_model.predict_proba(features)[0]
        # features   = np.array([[complexity, deadline, biz_value, dependencies]])
        # pred       = task_model.predict(features)[0]
        # proba      = task_model.predict_proba(features)[0]
        labels     = {0: "High", 1: "Medium", 2: "Low"}
        colors     = {0: "danger", 1: "warning", 2: "success"}

        return jsonify({
            "priority"   : labels[pred],
            "color"      : colors[pred],
            "confidence" : f"{max(proba)*100:.1f}%",
            "probabilities": {
                "High"  : f"{proba[0]*100:.1f}%",
                "Medium": f"{proba[1]*100:.1f}%",
                "Low"   : f"{proba[2]*100:.1f}%",
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# RISK ANALYSIS  (Rule-based AI)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/risk", methods=["GET", "POST"])
@login_required
def risk():
    """Rule-based Agile risk detection engine."""
    risks = []

    if request.method == "POST":
        delayed_tasks  = int(request.form.get("delayed_tasks", 0))
        sprint_load    = int(request.form.get("sprint_load", 0))
        bugs_count     = int(request.form.get("bugs_count", 0))
        velocity       = int(request.form.get("team_velocity", 0))
        capacity_used  = int(request.form.get("capacity_used", 0))
        days_remaining = int(request.form.get("days_remaining", 0))

        # ── Rule 1: Delayed tasks ────────────────────────────────────────────
        if delayed_tasks > 5:
            risks.append({
                "level"  : "HIGH",
                "color"  : "danger",
                "icon"   : "⚠️",
                "title"  : "Critical Task Delays",
                "message": f"{delayed_tasks} tasks are delayed. Sprint goal is at risk.",
                "action" : "Hold an emergency standup and re-prioritize backlog immediately.",
            })
        elif delayed_tasks > 2:
            risks.append({
                "level"  : "MEDIUM",
                "color"  : "warning",
                "icon"   : "🔶",
                "title"  : "Task Delays Detected",
                "message": f"{delayed_tasks} tasks are behind schedule.",
                "action" : "Review blockers and reassign tasks if needed.",
            })

        # ── Rule 2: Overloaded sprint ─────────────────────────────────────────
        if sprint_load > 40:
            risks.append({
                "level"  : "HIGH",
                "color"  : "danger",
                "icon"   : "🔥",
                "title"  : "Sprint Overloaded",
                "message": f"{sprint_load} story points in sprint — team may burn out.",
                "action" : "Remove low-priority items. Keep sprint under 35 story points.",
            })
        elif sprint_load > 30:
            risks.append({
                "level"  : "MEDIUM",
                "color"  : "warning",
                "icon"   : "📦",
                "title"  : "High Sprint Load",
                "message": f"{sprint_load} story points may stretch team capacity.",
                "action" : "Monitor daily velocity and adjust scope if necessary.",
            })

        # ── Rule 3: High bug count ────────────────────────────────────────────
        if bugs_count > 10:
            risks.append({
                "level"  : "HIGH",
                "color"  : "danger",
                "icon"   : "🐛",
                "title"  : "Critical Bug Count",
                "message": f"{bugs_count} open bugs threaten release quality.",
                "action" : "Allocate 50% of capacity to bug fixing this sprint.",
            })
        elif bugs_count > 5:
            risks.append({
                "level"  : "MEDIUM",
                "color"  : "warning",
                "icon"   : "🔍",
                "title"  : "Elevated Bug Count",
                "message": f"{bugs_count} bugs need attention before sprint review.",
                "action" : "Schedule dedicated bug-bash session with QA team.",
            })

        # ── Rule 4: Low team velocity ─────────────────────────────────────────
        if velocity < 20:
            risks.append({
                "level"  : "HIGH",
                "color"  : "danger",
                "icon"   : "📉",
                "title"  : "Very Low Team Velocity",
                "message": f"Velocity of {velocity} is critically low.",
                "action" : "Investigate impediments. Consider team capacity review.",
            })
        elif velocity < 35:
            risks.append({
                "level"  : "LOW",
                "color"  : "info",
                "icon"   : "⬇️",
                "title"  : "Below-Average Velocity",
                "message": f"Velocity of {velocity} is below team average.",
                "action" : "Check for hidden blockers or skill gaps.",
            })

        # ── Rule 5: Over capacity ─────────────────────────────────────────────
        if capacity_used > 90:
            risks.append({
                "level"  : "HIGH",
                "color"  : "danger",
                "icon"   : "👥",
                "title"  : "Team at Full Capacity",
                "message": f"{capacity_used}% capacity used — no buffer for surprises.",
                "action" : "Remove scope or request additional resources.",
            })

        # ── Rule 6: Sprint deadline pressure ─────────────────────────────────
        if days_remaining <= 2 and delayed_tasks > 0:
            risks.append({
                "level"  : "HIGH",
                "color"  : "danger",
                "icon"   : "⏰",
                "title"  : "Sprint Deadline Pressure",
                "message": f"Only {days_remaining} days left with {delayed_tasks} delayed tasks.",
                "action" : "Escalate to Product Owner. Move incomplete items to backlog.",
            })

        if not risks:
            risks.append({
                "level"  : "NONE",
                "color"  : "success",
                "icon"   : "✅",
                "title"  : "No Significant Risks",
                "message": "Sprint is progressing well. Keep up the good work!",
                "action" : "Continue daily standups and maintain current velocity.",
            })

    return render_template("risk.html",
                           user=session["user"],
                           role=session["role"],
                           risks=risks,
                           submitted=(request.method == "POST"))


# ══════════════════════════════════════════════════════════════════════════════
# STANDUP SUMMARY GENERATOR  (NLP: keyword extraction)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/standup", methods=["GET", "POST"])
@login_required
def standup():
    """Generate standup summary from team updates using keyword extraction."""
    summary = None

    # Keywords for classification
    DONE_KEYWORDS      = ["completed", "finished", "done", "fixed", "resolved",
                          "merged", "shipped", "deployed", "closed", "delivered"]
    INPROGRESS_KEYWORDS= ["working", "in progress", "started", "developing",
                          "implementing", "coding", "testing", "reviewing"]
    BLOCKER_KEYWORDS   = ["blocked", "blocker", "issue", "problem", "stuck",
                          "waiting", "dependency", "cannot", "unable", "failed"]

    if request.method == "POST":
        updates_raw = request.form.get("team_updates", "")
        lines       = [l.strip() for l in updates_raw.strip().splitlines() if l.strip()]

        completed   = []
        in_progress = []
        blockers    = []

        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in BLOCKER_KEYWORDS):
                blockers.append(line)
            elif any(kw in lower for kw in DONE_KEYWORDS):
                completed.append(line)
            elif any(kw in lower for kw in INPROGRESS_KEYWORDS):
                in_progress.append(line)
            else:
                in_progress.append(line)  # default bucket

        # Extract most common words for topic summary
        all_words  = re.findall(r"\b[a-zA-Z]{4,}\b", updates_raw.lower())
        stop_words = {"that", "this", "with", "have", "been", "will", "from",
                      "they", "their", "team", "also", "some", "more", "than",
                      "working", "completed", "started", "waiting", "blocked"}
        filtered   = [w for w in all_words if w not in stop_words]
        top_topics = [w for w, _ in Counter(filtered).most_common(8)]

        summary = {
            "completed"  : completed,
            "in_progress": in_progress,
            "blockers"   : blockers,
            "top_topics" : top_topics,
            "total_lines": len(lines),
        }

    return render_template("standup.html",
                           user=session["user"],
                           role=session["role"],
                           summary=summary)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS  (TextBlob NLP)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/sentiment", methods=["GET", "POST"])
@login_required
def sentiment():
    """Analyze retrospective feedback sentiment using TextBlob."""
    results  = None
    bulk_res = None

    if request.method == "POST":
        action = request.form.get("action", "single")

        if action == "single":
            feedback = request.form.get("feedback", "")
            if feedback.strip():
                blob      = TextBlob(feedback)
                polarity  = blob.sentiment.polarity       # -1 to +1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1

                if polarity > 0.1:
                    label, color, icon = "Positive", "success", "😊"
                elif polarity < -0.1:
                    label, color, icon = "Negative", "danger",  "😞"
                else:
                    label, color, icon = "Neutral",  "warning", "😐"

                results = {
                    "feedback"     : feedback,
                    "label"        : label,
                    "color"        : color,
                    "icon"         : icon,
                    "polarity"     : round(polarity, 3),
                    "subjectivity" : round(subjectivity, 3),
                    "polarity_pct" : round((polarity + 1) / 2 * 100, 1),
                }

        elif action == "bulk":
            # Analyze CSV dataset
            df = pd.read_csv("datasets/retrospective_feedback.csv")
            sentiments = []
            for _, row in df.iterrows():
                blob   = TextBlob(str(row["feedback"]))
                p      = blob.sentiment.polarity
                if p > 0.1:
                    label = "Positive"
                elif p < -0.1:
                    label = "Negative"
                else:
                    label = "Neutral"
                sentiments.append(label)

            df["predicted_sentiment"] = sentiments
            counts = df["predicted_sentiment"].value_counts().to_dict()
            total  = len(df)
            bulk_res = {
                "total"    : total,
                "positive" : counts.get("Positive", 0),
                "neutral"  : counts.get("Neutral",  0),
                "negative" : counts.get("Negative", 0),
                "pos_pct"  : round(counts.get("Positive", 0) / total * 100, 1),
                "neu_pct"  : round(counts.get("Neutral",  0) / total * 100, 1),
                "neg_pct"  : round(counts.get("Negative", 0) / total * 100, 1),
                "samples"  : df[["feedback", "predicted_sentiment"]].head(8).to_dict("records"),
            }

    return render_template("sentiment.html",
                           user=session["user"],
                           role=session["role"],
                           results=results,
                           bulk_res=bulk_res)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS / REPORTS  (data for Chart.js)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/charts")
@login_required
def charts():
    return render_template("charts.html",
                           user=session["user"],
                           role=session["role"])


@app.route("/api/chart_data")
@login_required
def chart_data():
    """Return sample chart data as JSON for Chart.js."""
    data = {
        "velocity_trend": {
            "labels": ["Sprint 1", "Sprint 2", "Sprint 3",
                       "Sprint 4", "Sprint 5", "Sprint 6"],
            "data"  : [38, 42, 35, 50, 47, 55],
        },
        "task_status": {
            "labels": ["Done", "In Progress", "To Do", "Blocked"],
            "data"  : [29, 12, 5, 1],
            "colors": ["#22c55e", "#3b82f6", "#a855f7", "#ef4444"],
        },
        "bug_trend": {
            "labels": ["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5", "Wk 6"],
            "data"  : [12, 8, 15, 6, 4, 2],
        },
        "sentiment_dist": {
            "labels": ["Positive", "Neutral", "Negative"],
            "data"  : [22, 10, 8],
            "colors": ["#22c55e", "#f59e0b", "#ef4444"],
        },
        "sprint_success": {
            "labels": ["Sprint 1", "Sprint 2", "Sprint 3",
                       "Sprint 4", "Sprint 5"],
            "success": [1, 0, 1, 1, 1],
        },
    }
    return jsonify(data)


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  AI-Powered Agile Coaching System")
    print("  Flask dev server starting …")
    if sprint_model is None:
        print("\n  ⚠  ML models not found!")
        print("  Run: python train_model.py  first\n")
    print("═"*55 + "\n")
    app.run(debug=True, port=5000)
