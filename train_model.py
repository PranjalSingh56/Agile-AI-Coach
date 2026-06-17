"""
train_model.py
==============
ML Model Training Script for AI-Powered Agile Coaching System
- Trains Sprint Success Prediction model (Logistic Regression)
- Trains Task Priority Prediction model (Random Forest)
- Saves models using pickle for use in Flask app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASETS_DIR = "datasets"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SPRINT SUCCESS PREDICTION  (Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════════

def train_sprint_model():
    print("\n" + "="*55)
    print("  TRAINING: Sprint Success Prediction Model")
    print("="*55)

    # Load dataset
    df = pd.read_csv(f"{DATASETS_DIR}/sprint_data.csv")
    print(f"  Dataset shape : {df.shape}")
    print(f"  Columns       : {list(df.columns)}")

    # Features and target
    X = df[["team_velocity", "completed_tasks", "pending_tasks",
            "bugs_count", "team_capacity"]]
    y = df["sprint_success"]   # 1 = success, 0 = failure

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy      : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Failure", "Success"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model + scaler
    with open(f"{MODELS_DIR}/sprint_model.pkl",  "wb") as f:
        pickle.dump(model,  f)
    with open(f"{MODELS_DIR}/sprint_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  ✓ Model  saved → {MODELS_DIR}/sprint_model.pkl")
    print(f"  ✓ Scaler saved → {MODELS_DIR}/sprint_scaler.pkl")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TASK PRIORITY PREDICTION  (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

def train_task_model():
    print("\n" + "="*55)
    print("  TRAINING: Task Priority Prediction Model")
    print("="*55)

    # Load dataset
    df = pd.read_csv(f"{DATASETS_DIR}/task_priority_data.csv")
    print(f"  Dataset shape : {df.shape}")
    print(f"  Columns       : {list(df.columns)}")

    # Features and target
    X = df[["complexity", "deadline_days", "business_value", "dependencies"]]
    y = df["priority"]   # 0=High, 1=Medium, 2=Low

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Random Forest does not require scaling
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy      : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["High", "Medium", "Low"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importances
    fi = pd.Series(model.feature_importances_,
                   index=X.columns).sort_values(ascending=False)
    print("\n  Feature Importances:")
    for feat, imp in fi.items():
        print(f"    {feat:<20} {imp:.4f}")

    # Save model
    with open(f"{MODELS_DIR}/task_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"\n  ✓ Model saved → {MODELS_DIR}/task_model.pkl")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*55)
    print("  AI-Powered Agile Coaching System — Model Training")
    print("█"*55)

    sprint_acc = train_sprint_model()
    task_acc   = train_task_model()

    print("\n" + "="*55)
    print("  TRAINING SUMMARY")
    print("="*55)
    print(f"  Sprint Success Model Accuracy : {sprint_acc*100:.2f}%")
    print(f"  Task Priority Model Accuracy  : {task_acc*100:.2f}%")
    print("\n  All models trained and saved successfully!")
    print("  Run 'python app.py' to start the Flask server.\n")
