"""
detection/predictor.py — SignBridge scikit-learn Predictor
===========================================================
Loads the trained .pkl model and exposes predict() to Django views.

Feature vector (83 dims):
  - 63  : normalised x/y/z for all 21 landmarks
  - 15  : pairwise distances between key fingertip/joint pairs
  -  5  : finger-extension ratios (tip-to-MCP / sum of segments)
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
WRIST       = 0
THUMB_TIP   = 4;  THUMB_IP  = 3;  THUMB_MCP = 2
INDEX_TIP   = 8;  INDEX_PIP = 6;  INDEX_MCP = 5
MIDDLE_TIP  = 12; MIDDLE_PIP= 10; MIDDLE_MCP= 9
RING_TIP    = 16; RING_PIP  = 14; RING_MCP  = 13
PINKY_TIP   = 20; PINKY_PIP = 18; PINKY_MCP = 17

DISTANCE_PAIRS = [
    (THUMB_TIP,  INDEX_TIP),
    (THUMB_TIP,  MIDDLE_TIP),
    (THUMB_TIP,  RING_TIP),
    (THUMB_TIP,  PINKY_TIP),
    (INDEX_TIP,  MIDDLE_TIP),
    (INDEX_TIP,  RING_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP,   PINKY_TIP),
    (INDEX_TIP,  PINKY_TIP),
    (THUMB_TIP,  WRIST),
    (INDEX_TIP,  WRIST),
    (MIDDLE_TIP, WRIST),
    (RING_TIP,   WRIST),
    (PINKY_TIP,  WRIST),
    (INDEX_MCP,  PINKY_MCP),
]

FINGER_JOINTS = [
    (THUMB_TIP,  THUMB_IP,   THUMB_MCP),
    (INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP,   RING_PIP,   RING_MCP),
    (PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
]

# ── ASL sign metadata (matches frontend ASL_SIGNS) ────────────────────────────
SIGN_DB: Dict[str, Dict] = {
    "A": {"meaning": "Letter A",    "category": "Alphabet", "emoji": "✊"},
    "B": {"meaning": "Letter B",    "category": "Alphabet", "emoji": "🖐️"},
    "C": {"meaning": "Letter C",    "category": "Alphabet", "emoji": "🤏"},
    "D": {"meaning": "Letter D",    "category": "Alphabet", "emoji": "👆"},
    "E": {"meaning": "Letter E",    "category": "Alphabet", "emoji": "✋"},
    "F": {"meaning": "Letter F",    "category": "Alphabet", "emoji": "👌"},
    "G": {"meaning": "Letter G",    "category": "Alphabet", "emoji": "👉"},
    "H": {"meaning": "Letter H",    "category": "Alphabet", "emoji": "✌️"},
    "I": {"meaning": "Letter I",    "category": "Alphabet", "emoji": "🤙"},
    "K": {"meaning": "Letter K",    "category": "Alphabet", "emoji": "✌️"},
    "L": {"meaning": "Letter L",    "category": "Alphabet", "emoji": "🤟"},
    "M": {"meaning": "Letter M",    "category": "Alphabet", "emoji": "✊"},
    "N": {"meaning": "Letter N",    "category": "Alphabet", "emoji": "✊"},
    "O": {"meaning": "Letter O",    "category": "Alphabet", "emoji": "👌"},
    "P": {"meaning": "Letter P",    "category": "Alphabet", "emoji": "👇"},
    "Q": {"meaning": "Letter Q",    "category": "Alphabet", "emoji": "👇"},
    "R": {"meaning": "Letter R",    "category": "Alphabet", "emoji": "✌️"},
    "S": {"meaning": "Letter S",    "category": "Alphabet", "emoji": "✊"},
    "T": {"meaning": "Letter T",    "category": "Alphabet", "emoji": "✊"},
    "U": {"meaning": "Letter U",    "category": "Alphabet", "emoji": "✌️"},
    "V": {"meaning": "Letter V",    "category": "Alphabet", "emoji": "✌️"},
    "W": {"meaning": "Letter W",    "category": "Alphabet", "emoji": "🖖"},
    "X": {"meaning": "Letter X",    "category": "Alphabet", "emoji": "☝️"},
    "Y": {"meaning": "Letter Y",    "category": "Alphabet", "emoji": "🤙"},
    "1": {"meaning": "Number 1",    "category": "Numbers",  "emoji": "☝️"},
    "2": {"meaning": "Number 2",    "category": "Numbers",  "emoji": "✌️"},
    "3": {"meaning": "Number 3",    "category": "Numbers",  "emoji": "🤟"},
    "4": {"meaning": "Number 4",    "category": "Numbers",  "emoji": "🖖"},
    "5": {"meaning": "Number 5",    "category": "Numbers",  "emoji": "🖐️"},
    "HELLO":     {"meaning": "Hello",     "category": "Phrases", "emoji": "👋"},
    "THANK YOU": {"meaning": "Thank You", "category": "Phrases", "emoji": "🙏"},
    "YES":       {"meaning": "Yes",       "category": "Phrases", "emoji": "✅"},
    "NO":        {"meaning": "No",        "category": "Phrases", "emoji": "❌"},
    "PLEASE":    {"meaning": "Please",    "category": "Phrases", "emoji": "🙏"},
    "SORRY":     {"meaning": "Sorry",     "category": "Phrases", "emoji": "😔"},
    "LOVE":      {"meaning": "I Love You","category": "Phrases", "emoji": "🤟"},
    "HELP":      {"meaning": "Help",      "category": "Phrases", "emoji": "🆘"},
    "STOP":      {"meaning": "Stop",      "category": "Phrases", "emoji": "✋"},
    "MORE":      {"meaning": "More",      "category": "Phrases", "emoji": "👐"},
}


# ── Feature engineering ────────────────────────────────────────────────────────

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def extract_features(landmarks: List[Dict]) -> np.ndarray:
    """
    Convert 21 raw MediaPipe landmarks → normalised 83-dim feature vector.

    Parameters
    ----------
    landmarks : list of 21 dicts with keys 'x', 'y', 'z'

    Returns
    -------
    np.ndarray of shape (83,)
    """
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks],
                   dtype=np.float32)          # (21, 3)

    # 1. Translate: wrist → origin
    pts = pts - pts[WRIST]

    # 2. Scale: divide by max distance wrist→any point
    scale = max(np.linalg.norm(pts, axis=1).max(), 1e-6)
    pts  /= scale

    # 3. Flatten → 63 normalised coords
    raw_features = pts.flatten()

    # 4. Pairwise distances (15 values)
    dist_features = np.array(
        [_dist(pts[i], pts[j]) for i, j in DISTANCE_PAIRS],
        dtype=np.float32
    )

    # 5. Finger extension ratios (5 values)
    ext_features = []
    for tip, pip, mcp in FINGER_JOINTS:
        tip_mcp = _dist(pts[tip], pts[mcp])
        tip_pip = _dist(pts[tip], pts[pip])
        pip_mcp = _dist(pts[pip], pts[mcp])
        ratio   = tip_mcp / (tip_pip + pip_mcp + 1e-6)
        ext_features.append(ratio)
    ext_features = np.array(ext_features, dtype=np.float32)

    return np.concatenate([raw_features, dist_features, ext_features])  # (83,)


# ── Predictor class ────────────────────────────────────────────────────────────

class SignPredictor:
    """
    Singleton-friendly wrapper around the trained scikit-learn model.
    Falls back to a geometric rule-based classifier if no .pkl exists yet.
    """

    # Django loads this once at startup via apps.py ready()
    _instance = None

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default: <project_root>/ml/sign_model.pkl
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base, "ml", "sign_model.pkl")

        self.sign_db    = SIGN_DB
        self.model_path = model_path

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
            self.model      = payload["model"]
            self.label_enc  = payload["label_encoder"]
            self.model_name = payload.get("model_name", "RandomForest")
            self.accuracy   = payload.get("test_accuracy", None)
            self._use_model = True
        else:
            self.model      = None
            self.label_enc  = None
            self.model_name = "GeometricFallback"
            self.accuracy   = None
            self._use_model = False

    def predict(self, landmarks: List[Dict]) -> Tuple[str, float]:
        """
        Parameters
        ----------
        landmarks : list of 21 dicts {'x': float, 'y': float, 'z': float}

        Returns
        -------
        (sign_label, confidence)  e.g. ("A", 0.94)
        """
        if self._use_model:
            return self._model_predict(landmarks)
        return self._geometric_predict(landmarks)

    def _model_predict(self, landmarks: List[Dict]) -> Tuple[str, float]:
        features = extract_features(landmarks).reshape(1, -1)
        proba    = self.model.predict_proba(features)[0]
        idx      = int(np.argmax(proba))
        label    = self.label_enc.inverse_transform([idx])[0]
        return label, float(proba[idx])

    def _geometric_predict(self, landmarks: List[Dict]) -> Tuple[str, float]:
        """Simple rule-based fallback (mirrors GestureClassifier in app.js)."""
        lm = landmarks

        def extended(tip, pip, mcp):
            return lm[tip]["y"] < lm[pip]["y"] < lm[mcp]["y"]

        thumb  = lm[4]["x"] < lm[3]["x"]
        index  = extended(8,  6,  5)
        middle = extended(12, 10, 9)
        ring   = extended(16, 14, 13)
        pinky  = extended(20, 18, 17)
        f = (thumb, index, middle, ring, pinky)

        RULES = {
            (False, False, False, False, False): ("A",    0.72),
            (False, True,  True,  True,  True ): ("B",   0.75),
            (True,  True,  False, False, False): ("L",   0.78),
            (True,  False, False, False, True ): ("Y",   0.77),
            (False, True,  False, False, False): ("1",   0.80),
            (False, True,  True,  False, False): ("2",   0.78),
            (True,  True,  True,  False, False): ("3",   0.75),
            (False, True,  True,  True,  False): ("4",   0.73),
            (True,  True,  True,  True,  True ): ("5",   0.80),
            (True,  True,  False, False, True ): ("LOVE",0.79),
        }
        return RULES.get(f, ("?", 0.30))


# Module-level singleton — imported by views.py
_predictor: SignPredictor = None


def get_predictor() -> SignPredictor:
    """Return the shared SignPredictor instance (lazy init)."""
    global _predictor
    if _predictor is None:
        _predictor = SignPredictor()
    return _predictor
