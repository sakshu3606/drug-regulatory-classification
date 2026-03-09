from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys, subprocess
import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings("ignore")

# ── Ensure feature-engine is installed (blocks until done) ─────────────────
try:
    import feature_engine
    print("✅ feature-engine ready:", feature_engine.__version__)
except ImportError:
    print("⏳ Installing feature-engine — please wait...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "feature-engine==1.6.2", "--quiet"],
        capture_output=True, text=True
    )
    try:
        import feature_engine
        print("✅ feature-engine installed:", feature_engine.__version__)
    except ImportError:
        print("❌ CRITICAL: feature-engine failed to install:", r.stderr[:200])

# ── TensorFlow optional (not available on all deployment platforms) ──────────
TF_AVAILABLE = False
try:
    import tensorflow as _tf
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except Exception:
    print("⚠️  TensorFlow not available — ANN model will be disabled")

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Each model pkl is a full sklearn Pipeline (preprocess + model).
# ANN pkl is a raw Keras model (needs separate preprocess pipeline).
# Decision Tree, SVM, XGBoost, ANN were trained with LabelEncoder on target
#   → they predict integers 0/1, which we decode back to class names.
# Random Forest was trained on ALL-LOWERCASE column names.
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY = {
    "Logistic Regression": {
        "file":        "logistic_model.pkl",
        "col_case":    "original",   # trained on original-case column names
        "label_enc":   False,        # target was string labels → predicts strings
    },
    "Decision Tree": {
        "file":        "decision_tree_model.pkl",
        "col_case":    "original",
        "label_enc":   True,         # target was LabelEncoded → predicts 0/1
    },
    "Random Forest": {
        "file":        "random_forest_model.pkl",
        "col_case":    "lower",      # trained on all-lowercase column names!
        "label_enc":   False,        # target was string labels
    },
    "KNN": {
        "file":        "knn_model.pkl",
        "col_case":    "original",
        "label_enc":   False,
    },
    "SVM": {
        "file":        "svm_model.pkl",
        "col_case":    "original",
        "label_enc":   True,
    },
    "XGBoost": {
        "file":        "xgboost_model.pkl",
        "col_case":    "original",
        "label_enc":   True,
    },
    "Deep Learning (ANN)": {
        "file":        "deep_learning_ANN_model.pkl",
        "col_case":    "original",
        "label_enc":   True,         # ANN trained on LabelEncoded target → predicts 0/1
        "is_ann":      True,
        # ANN used its OWN pipeline: StandardScaler + OneHotEncoder (no Winsorizer)
        # Save it by running the updated dl_ann.ipynb → preprocess_pipeline_ann.pkl
        "ann_pipe":    "preprocess_pipeline_ann.pkl",
    },
}

# LabelEncoder mapping for DT / SVM / XGBoost:
#   target was original-case strings → alphabetical: 0=Non-Regulated Drug, 1=Regulated Drug
LABEL_MAP = {0: "Non-Regulated Drug", 1: "Regulated Drug"}

# LabelEncoder mapping for ANN:
#   target was lowercased FIRST (y.str.lower()) before LabelEncoding
#   → alphabetical on lowercase: 0=non-regulated drug, 1=regulated drug
ANN_LABEL_MAP = {0: "Non-Regulated Drug", 1: "Regulated Drug"}

_cache = {}

# ── Original-case column names (as in the dataset for most models) ───────────
NUMERIC_ORIG = [
    "Dosage_mg", "Price_Per_Unit", "Production_Cost", "Marketing_Spend",
    "Clinical_Trial_Phase", "Side_Effect_Severity_Score", "Abuse_Potential_Score",
    "Prescription_Rate", "Hospital_Distribution_Percentage", "Pharmacy_Distribution_Percentage",
    "Annual_Sales_Volume", "Regulatory_Risk_Score", "Approval_Time_Months",
    "Patent_Duration_Years", "R&D_Investment_Million", "Competitor_Count",
    "Recall_History_Count", "Adverse_Event_Reports", "Insurance_Coverage_Percentage",
    "Export_Percentage", "Online_Sales_Percentage", "Brand_Reputation_Score",
    "Doctor_Recommendation_Rate",
]

CATEGORICAL_ORIG = [
    "Drug_Form", "Therapeutic_Class", "Manufacturing_Region",
    "Requires_Cold_Storage", "OTC_Flag", "High_Risk_Substance",
]

ALL_ORIG = NUMERIC_ORIG + CATEGORICAL_ORIG

# ── Lowercase column names (Random Forest trained on these) ──────────────────
NUMERIC_LOWER   = [c.lower() for c in NUMERIC_ORIG]
CATEGORICAL_LOWER = [c.lower() for c in CATEGORICAL_ORIG]
ALL_LOWER       = [c.lower() for c in ALL_ORIG]


def _load_pkl(path):
    import joblib, pickle

    errors = []
    for fn in [
        lambda p: joblib.load(p),
        lambda p: pickle.load(open(p, "rb")),
        lambda p: pickle.load(open(p, "rb"), encoding="latin-1"),
    ]:
        try:
            return fn(path)
        except Exception as e:
            errors.append(str(e))

    combined = " | ".join(errors)
    if any(x in combined for x in ["feature_engine", "Winsorizer", "No module", "cannot import"]):
        raise RuntimeError(
            f"PKL '{os.path.basename(path)}' needs feature-engine. "
            f"Run Cell 1 + Cell 2 in launch_pharmai.ipynb locally, "
            f"re-push all .pkl files to GitHub. Detail: {combined[:300]}"
        )
    raise RuntimeError(f"Cannot load '{os.path.basename(path)}': {combined[:300]}")


def load_model(name):
    if name in _cache:
        return _cache[name]
    cfg  = REGISTRY[name]

    # ANN requires TensorFlow — skip gracefully if not installed
    if cfg.get("is_ann") and not TF_AVAILABLE:
        raise RuntimeError(
            "Deep Learning (ANN) model requires TensorFlow which is not "
            "installed on this server. All other 6 models are available."
        )

    path = os.path.join(BASE, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{cfg['file']}' not found on server. "
            f"PKL files must be committed to GitHub to deploy on Render. "
            f"See README for instructions."
        )
    obj = _load_pkl(path)
    _cache[name] = obj
    return obj


def build_row(features: dict, col_case: str) -> pd.DataFrame:
    """
    Build a single-row DataFrame with correct column names for each model.
    col_case='original' → original case  |  col_case='lower' → all lowercase
    """
    norm = {k.lower(): v for k, v in features.items()}

    # alias: rd_investment_million → r&d_investment_million
    if "rd_investment_million" in norm and "r&d_investment_million" not in norm:
        norm["r&d_investment_million"] = norm["rd_investment_million"]

    numeric_cols    = NUMERIC_ORIG    if col_case == "original" else NUMERIC_LOWER
    categorical_cols = CATEGORICAL_ORIG if col_case == "original" else CATEGORICAL_LOWER
    all_cols        = ALL_ORIG        if col_case == "original" else ALL_LOWER
    numeric_set     = set(c.lower() for c in numeric_cols)

    row = {}
    for col in all_cols:
        val = norm.get(col.lower(), 0)
        if col.lower() in numeric_set:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
        row[col] = val

    return pd.DataFrame([row], columns=all_cols)


def decode_label(raw) -> str:
    """Convert integer label-encoded predictions back to class name strings."""
    try:
        idx = int(float(str(raw)))
        return LABEL_MAP.get(idx, str(raw))
    except Exception:
        return str(raw)


def _build_ann_pipeline(df: pd.DataFrame) -> np.ndarray:
    """
    Load the ANN's own saved preprocessing pipeline (preprocess_pipeline_ann.pkl).
    This pipeline uses StandardScaler + OneHotEncoder (NO Winsorizer) — different
    from the shared preprocess_pipeline.pkl used by other models.

    If preprocess_pipeline_ann.pkl doesn't exist yet:
      → Run the updated dl_ann.ipynb (all cells) to generate it.
    """
    # Primary: ANN's own pipeline (StandardScaler + OHE, no Winsorizer)
    ann_pipe_path = os.path.join(BASE, "preprocess_pipeline_ann.pkl")
    if os.path.exists(ann_pipe_path):
        try:
            pipe = _load_pkl(ann_pipe_path)
            X = pipe.transform(df)
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.array(X, dtype="float32")
        except Exception as e:
            raise RuntimeError(
                f"preprocess_pipeline_ann.pkl failed to transform: {e}. "
                f"Re-run dl_ann.ipynb to regenerate it."
            )

    # Fallback: shared pipeline (different scaler but same shape — better than nothing)
    shared_pipe_path = os.path.join(BASE, "preprocess_pipeline.pkl")
    if os.path.exists(shared_pipe_path):
        try:
            pipe = _load_pkl(shared_pipe_path)
            X = pipe.transform(df)
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.array(X, dtype="float32")
        except Exception:
            pass

    # Last resort: numeric only
    return df[NUMERIC_ORIG].values.astype("float32")


def _ann_predict(model, features: dict) -> dict:
    """
    Handle raw Keras ANN model prediction.
    ANN was trained with sparse_categorical_crossentropy + softmax output
    → output shape (1, num_classes), use argmax.
    ANN LabelEncoder ran on lowercased labels:
      0 = 'non-regulated drug' → 'Non-Regulated Drug'
      1 = 'regulated drug'     → 'Regulated Drug'
    """
    df  = build_row(features, "original")
    X   = _build_ann_pipeline(df)

    raw = model.predict(X, verbose=0)

    # softmax multi-class output → argmax
    if raw.ndim > 1 and raw.shape[1] > 1:
        probs = raw[0]
        idx   = int(np.argmax(probs))
        pred  = ANN_LABEL_MAP.get(idx, str(idx))
        proba = {ANN_LABEL_MAP.get(i, str(i)): round(float(probs[i]) * 100, 2)
                 for i in range(len(probs))}
    else:
        # sigmoid single output (fallback)
        prob_pos = float(np.clip(raw.flatten()[0], 0.0, 1.0))
        pred  = "Regulated Drug" if prob_pos >= 0.5 else "Non-Regulated Drug"
        proba = {
            "Regulated Drug":     round(prob_pos * 100, 2),
            "Non-Regulated Drug": round((1.0 - prob_pos) * 100, 2),
        }

    return {"prediction": pred, "probability": proba}


def predict_one(name: str, features: dict) -> dict:
    cfg   = REGISTRY[name]
    model = load_model(name)

    # ── ANN: raw Keras model ─────────────────────────────────────────────────
    if cfg.get("is_ann") and not hasattr(model, "named_steps"):
        return _ann_predict(model, features)

    # ── sklearn full pipeline ────────────────────────────────────────────────
    df       = build_row(features, cfg["col_case"])
    raw_pred = model.predict(df)
    raw_val  = raw_pred[0] if isinstance(raw_pred, (list, np.ndarray)) else raw_pred

    # Decode integer labels back to human-readable class names
    if cfg["label_enc"]:
        pred = decode_label(raw_val)
    else:
        pred = str(raw_val)

    # Build probability dict
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(df)[0]

            # Get class labels from pipeline's final estimator
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
            elif hasattr(model, "named_steps"):
                classes = list(list(model.named_steps.values())[-1].classes_)
            else:
                classes = list(np.unique(raw_pred))

            if cfg["label_enc"]:
                # Classes are integers → decode to strings
                class_names = [decode_label(c) for c in classes]
            else:
                class_names = [str(c) for c in classes]

            proba = {class_names[i]: round(float(p[i]) * 100, 2) for i in range(len(classes))}
        except Exception:
            proba = None

    return {"prediction": pred, "probability": proba}


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/models/status")
def model_status():
    status = {}
    for name, cfg in REGISTRY.items():
        status[name] = {
            "ready":      os.path.exists(os.path.join(BASE, cfg["file"])),
            "model_file": cfg["file"],
        }
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    body       = request.get_json(force=True)
    model_name = body.get("model", "Random Forest")
    features   = body.get("features", {})

    if model_name not in REGISTRY:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    try:
        result = predict_one(model_name, features)
        return jsonify({"model": model_name, **result})
    except Exception as e:
        full_trace = traceback.format_exc()
        print(f"PREDICT ERROR [{model_name}]:\n{full_trace}")
        return jsonify({"error": str(e), "trace": full_trace}), 500


@app.route("/predict/all", methods=["POST"])
def predict_all():
    body     = request.get_json(force=True)
    features = body.get("features", {})

    votes, models = {}, {}

    for name in REGISTRY:
        try:
            result       = predict_one(name, features)
            pred         = result["prediction"]
            votes[pred]  = votes.get(pred, 0) + 1
            models[name] = {"available": True, **result}
        except Exception as e:
            models[name] = {"available": False, "error": str(e)}

    if not votes:
        return jsonify({"error": "No models produced a prediction"}), 500

    ensemble    = max(votes, key=votes.get)
    total_voted = sum(votes.values())

    return jsonify({
        "ensemble":    ensemble,
        "confidence":  round((votes[ensemble] / total_voted) * 100, 2),
        "votes":       votes,
        "total_voted": total_voted,
        "models":      models,
    })



@app.route("/debug")
def debug():
    import platform
    info = {"python": sys.version, "platform": platform.platform(), "base": BASE}
    pkgs = {}
    for pkg in ["feature_engine", "sklearn", "xgboost", "joblib", "pandas", "numpy", "flask"]:
        try:
            m = __import__(pkg)
            pkgs[pkg] = getattr(m, "__version__", "ok")
        except ImportError as e:
            pkgs[pkg] = f"MISSING: {e}"
    info["packages"] = pkgs
    pkl_status = {}
    for name, cfg in REGISTRY.items():
        path = os.path.join(BASE, cfg["file"])
        if not os.path.exists(path):
            pkl_status[name] = "❌ FILE MISSING from GitHub"
        else:
            try:
                _load_pkl(path)
                pkl_status[name] = "✅ OK"
            except Exception as e:
                pkl_status[name] = f"❌ {str(e)[:200]}"
    info["pkl_files"] = pkl_status
    return jsonify(info)


if __name__ == "__main__":
    import logging, os
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 5050))
    host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    debug = not bool(os.environ.get("RENDER"))
    print(f"\n✅  PharmAI Flask server  →  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
