# ml/

This folder contains the scikit-learn model training pipeline.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Train Random Forest / SVM / GradientBoosting on landmark data |
| `sign_model.pkl` | Trained model (generated after running train_model.py) |
| `confusion_matrix.png` | Model evaluation chart (generated after training) |
| `data/landmarks.csv` | Your real landmark dataset (collect via webcam) |

## How to train

```bash
# From the project root (signbridge_project/)
python ml/train_model.py

# Or with your own CSV data:
python ml/train_model.py --data ml/data/landmarks.csv --output ml/sign_model.pkl
```

## CSV format

```
label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
A,     0.51, 0.72, -0.03, ...
B,     0.49, 0.68, -0.01, ...
```

One row = one hand sample. 21 landmarks × 3 coords = 63 columns + 1 label column.

## What happens at runtime

Django loads `ml/sign_model.pkl` once at startup (via `detection/predictor.py`).
If the file doesn't exist, the app falls back to the geometric rule-based classifier.
