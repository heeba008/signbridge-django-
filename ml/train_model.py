"""
ml/train_model.py — SignBridge Model Training Script
=====================================================

Trains a scikit-learn classifier on ASL hand landmark data and saves
the best model to ml/sign_model.pkl (loaded by Django at runtime).

Pipeline:
  1. Load landmark CSV (or generate synthetic data for demo)
  2. Feature engineering via detection/predictor.extract_features()
  3. Train Random Forest + SVM — compare accuracy with cross-validation
  4. Save the best model as ml/sign_model.pkl

Dataset CSV format  (one row per sample):
  label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
  A,     0.51, 0.72, -0.03, ...

To collect real data:
  - Use collect_data.py (opens webcam, saves labelled landmarks)
  - Or use the Kaggle ASL Alphabet landmark dataset:
    https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Usage:
  cd signbridge_project          # project root
  python ml/train_model.py
  python ml/train_model.py --data ml/data/landmarks.csv --output ml/sign_model.pkl
"""

import os
import sys
import argparse
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import classification_report, confusion_matrix
from sklearn.pipeline        import Pipeline
import matplotlib.pyplot     as plt
import seaborn               as sns

# Make sure Django's detection app is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.predictor import extract_features, SIGN_DB

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("train")


# ── Synthetic data generator ───────────────────────────────────────────────────

def generate_synthetic_data(n_per_class: int = 200, noise: float = 0.03) -> pd.DataFrame:
    """
    Generates landmark data from geometric finger-state rules + Gaussian noise.
    Good for smoke-testing the pipeline. NOT a substitute for real webcam data.
    """
    logger.warning("No real dataset found — generating synthetic data for demo.")
    logger.warning("Collect real webcam samples for production accuracy.")

    SIGN_STATES = {
        "A": (0,0,0,0,0), "B": (0,1,1,1,1), "C": (0,0,0,0,0),
        "D": (0,1,0,0,0), "E": (0,0,0,0,0), "F": (1,0,1,1,1),
        "G": (1,1,0,0,0), "H": (0,1,1,0,0), "I": (0,0,0,0,1),
        "K": (1,1,1,0,0), "L": (1,1,0,0,0), "M": (0,0,0,0,0),
        "N": (0,0,0,0,0), "O": (0,0,0,0,0), "P": (1,1,0,0,0),
        "Q": (1,1,0,0,0), "R": (0,1,1,0,0), "S": (0,0,0,0,0),
        "T": (0,0,0,0,0), "U": (0,1,1,0,0), "V": (0,1,1,0,0),
        "W": (0,1,1,1,0), "X": (0,1,0,0,0), "Y": (1,0,0,0,1),
        "1": (0,1,0,0,0), "2": (0,1,1,0,0), "3": (1,1,1,0,0),
        "4": (0,1,1,1,1), "5": (1,1,1,1,1),
        "HELLO":     (1,1,1,1,1), "YES":   (0,0,0,0,0),
        "NO":        (0,1,1,0,0), "PLEASE":(1,1,1,1,1),
        "SORRY":     (0,0,0,0,0), "LOVE":  (1,1,0,0,1),
        "HELP":      (1,1,1,1,1), "STOP":  (1,1,1,1,1),
        "MORE":      (0,0,0,0,0), "THANK YOU": (0,1,1,1,1),
    }

    rows = []
    rng  = np.random.default_rng(42)

    for label, (th, ix, mi, ri, pk) in SIGN_STATES.items():
        for _ in range(n_per_class):
            lms = [{"x": 0.5, "y": 0.9, "z": 0.0}]  # wrist

            tx = 0.45 - th * 0.08
            for j in range(4):
                lms.append({
                    "x": tx + rng.normal(0, noise),
                    "y": 0.8 - j * 0.04 + rng.normal(0, noise),
                    "z": rng.normal(0, noise * 0.5),
                })

            for fi, ext in enumerate([ix, mi, ri, pk]):
                base_x = 0.40 + fi * 0.07
                for j in range(4):
                    y_drop = 0.03 if ext else 0.06
                    lms.append({
                        "x": base_x + rng.normal(0, noise),
                        "y": 0.75 - j * y_drop + rng.normal(0, noise),
                        "z": rng.normal(0, noise * 0.5),
                    })

            row = {"label": label}
            for i, lm in enumerate(lms):
                row[f"x{i}"] = lm["x"]
                row[f"y{i}"] = lm["y"]
                row[f"z{i}"] = lm["z"]
            rows.append(row)

    return pd.DataFrame(rows)


# ── Feature matrix builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    X, y = [], []
    for _, row in df.iterrows():
        landmarks = [{"x": row[f"x{i}"], "y": row[f"y{i}"], "z": row[f"z{i}"]}
                     for i in range(21)]
        X.append(extract_features(landmarks))
        y.append(row["label"])
    return np.array(X), np.array(y)


# ── Model definitions ──────────────────────────────────────────────────────────

def get_models():
    return {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
            ))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42,
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            ))
        ]),
    }


# ── Main training function ─────────────────────────────────────────────────────

def train(data_path: str, output_path: str) -> float:
    # Load data
    if os.path.exists(data_path):
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        df = generate_synthetic_data(n_per_class=200)

    logger.info(f"Dataset: {len(df)} samples, {df['label'].nunique()} classes")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    # Feature engineering
    logger.info("Extracting features…")
    X, y_raw = build_features(df)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    logger.info(f"Feature vector size: {X.shape[1]}")   # should be 83

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Train all models and compare
    models  = get_models()
    best_name, best_score, best_model = None, 0.0, None

    for name, model in models.items():
        logger.info(f"Training {name}…")
        model.fit(X_train, y_train)

        test_acc  = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

        logger.info(
            f"  {name}: test={test_acc:.3f}  "
            f"cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
        )

        if test_acc > best_score:
            best_score, best_name, best_model = test_acc, name, model

    logger.info(f"\n🏆 Best model: {best_name} (test accuracy: {best_score:.3f})")

    # Detailed classification report
    y_pred = best_model.predict(X_test)
    print("\n" + "=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=le.inverse_transform(sorted(set(y_test)))
    ))

    # Confusion matrix
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(output_path), "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    logger.info(f"Confusion matrix → {cm_path}")
    plt.close()

    # Save model
    payload = {
        "model":         best_model,
        "label_encoder": le,
        "model_name":    best_name,
        "test_accuracy": best_score,
        "feature_size":  X.shape[1],
        "classes":       list(le.classes_),
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Model saved → {output_path}")

    return best_score


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignBridge ASL classifier")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "data", "landmarks.csv"),
        help="Path to landmark CSV",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "sign_model.pkl"),
        help="Where to save the trained model",
    )
    args = parser.parse_args()

    acc = train(args.data, args.output)
    print(f"\n✅ Training complete. Final accuracy: {acc:.1%}")
