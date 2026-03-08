from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings("ignore")   # suppress sklearn version warnings

app = Flask(__name__)
CORS(app)  # Allow browser fetch from any origin

BASE = os.path.dirname(os.path.abspath(__file__))

REGISTRY = {
    "Logistic Regression":  ("logistic_model.pkl",          "preprocess_pipeline.pkl"),
    "Decision Tree":        ("decision_tree_model.pkl",      "preprocess_pipeline_dt.pkl"),
    "Random Forest":        ("random_forest_model.pkl",      "preprocess_pipeline_rf.pkl"),
    "KNN":                  ("knn_model.pkl",                "preprocess_pipeline.pkl"),
    "SVM":                  ("svm_model.pkl",                "preprocess_pipeline.pkl"),
    "XGBoost":              ("xgboost_model.pkl",            "preprocess_pipeline.pkl"),
    "Deep Learning (ANN)":  ("deep_learning_ANN_model.pkl",  "preprocess_pipeline.pkl"),
}

_cache = {}

NUMERIC = [
    "Dosage_mg", "Price_Per_Unit", "Production_Cost", "Marketing_Spend",
    "Clinical_Trial_Phase", "Side_Effect_Severity_Score", "Abuse_Potential_Score",
    "Prescription_Rate", "Hospital_Distribution_Percentage", "Pharmacy_Distribution_Percentage",
    "Annual_Sales_Volume", "Regulatory_Risk_Score", "Approval_Time_Months",
    "Patent_Duration_Years", "R&D_Investment_Million", "Competitor_Count",
    "Recall_History_Count", "Adverse_Event_Reports", "Insurance_Coverage_Percentage",
    "Export_Percentage", "Online_Sales_Percentage", "Brand_Reputation_Score",
    "Doctor_Recommendation_Rate",
]

CATEGORICAL = [
    "Drug_Form", "Therapeutic_Class", "Manufacturing_Region",
    "Requires_Cold_Storage", "OTC_Flag", "High_Risk_Substance",
]

ALL_FEATURES = NUMERIC + CATEGORICAL


def _load_pkl(path):
    """
    Try multiple deserialization strategies to handle pickle/joblib
    compatibility issues (e.g. STACK_GLOBAL requires str).
    """
    # Strategy 1: standard pickle
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # Strategy 2: joblib (handles numpy arrays better across versions)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass

    # Strategy 3: pickle with latin-1 encoding fix
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin-1")
    except Exception:
        pass

    # Strategy 4: pickle with bytes encoding
    try:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")
    except Exception as e:
        raise RuntimeError(
            f"Cannot load pkl file '{os.path.basename(path)}'. "
            f"Please run Cell 2 (Fix PKL) in the notebook to re-save it. "
            f"Original error: {e}"
        )


def load(name):
    if name in _cache:
        return _cache[name]

    mf, pf = REGISTRY[name]
    model_path = os.path.join(BASE, mf)
    pipe_path  = os.path.join(BASE, pf)

    if not os.path.exists(pipe_path):
        raise FileNotFoundError(f"Pipeline file not found: {pf}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {mf}")

    pipe = _load_pkl(pipe_path)

    if name == "Deep Learning (ANN)":
        # Try keras SavedModel first, then pkl
        try:
            import tensorflow as tf
            # If it's a directory (SavedModel format)
            if os.path.isdir(model_path.replace(".pkl", "")):
                model = tf.keras.models.load_model(model_path.replace(".pkl", ""))
            else:
                model = _load_pkl(model_path)
        except Exception:
            model = _load_pkl(model_path)
    else:
        model = _load_pkl(model_path)

    _cache[name] = (model, pipe)
    return model, pipe


def build_row(features: dict) -> pd.DataFrame:
    """
    Normalise the incoming feature dict and return a single-row DataFrame
    with exactly ALL_FEATURES columns in the correct order.
    """
    normalised = {}
    for k, v in features.items():
        normalised[k.lower()] = v

    # Accept alias "rd_investment_million" for "r&d_investment_million"
    if "rd_investment_million" in normalised and "r&d_investment_million" not in normalised:
        normalised["r&d_investment_million"] = normalised["rd_investment_million"]

    row = {}
    for col in ALL_FEATURES:
        val = normalised.get(col.lower(), 0)
        if col in NUMERIC:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
        row[col] = val

    return pd.DataFrame([row], columns=ALL_FEATURES)


def safe_transform(pipe, df: pd.DataFrame) -> np.ndarray:
    """
    Robustly transform the DataFrame through the pipeline.
    Handles ColumnTransformer deprecations and sparse/dense matrix differences.
    """
    # Strategy 1: transform with full DataFrame (preferred)
    try:
        X = pipe.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.array(X, dtype=float)
    except Exception:
        pass

    # Strategy 2: suppress feature_names_in_ mismatch by resetting it
    try:
        def _reset_feature_names(obj):
            if hasattr(obj, "feature_names_in_"):
                try:
                    obj.feature_names_in_ = np.array(ALL_FEATURES)
                except Exception:
                    pass
            if hasattr(obj, "steps"):
                for _, step in obj.steps:
                    _reset_feature_names(step)
            if hasattr(obj, "transformers"):
                for _, trans, _ in obj.transformers:
                    _reset_feature_names(trans)

        _reset_feature_names(pipe)
        X = pipe.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.array(X, dtype=float)
    except Exception:
        pass

    # Strategy 3: try passing as numpy array to skip column-name validation
    try:
        X = pipe.transform(df.values)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.array(X, dtype=float)
    except Exception:
        pass

    # Strategy 4: manually apply numeric + categorical transformers
    try:
        num_vals = df[NUMERIC].values.astype(float)
        cat_encoded = None

        def _get_transformers(obj):
            if hasattr(obj, "transformers"):
                return obj.transformers
            if hasattr(obj, "steps"):
                for _, step in obj.steps:
                    if hasattr(step, "transformers"):
                        return step.transformers
            return []

        for tname, trans, cols in _get_transformers(pipe):
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols:
                continue
            try:
                if tname in ("num", "numeric", "scaler"):
                    out = trans.transform(df[valid_cols])
                    if hasattr(out, "toarray"):
                        out = out.toarray()
                    num_vals = np.array(out, dtype=float)
                elif tname in ("cat", "categorical", "onehot"):
                    out = trans.transform(df[valid_cols])
                    if hasattr(out, "toarray"):
                        out = out.toarray()
                    cat_encoded = np.array(out, dtype=float)
            except Exception:
                pass

        if cat_encoded is not None:
            return np.hstack([num_vals, cat_encoded])
        return num_vals
    except Exception:
        pass

    # Strategy 5: last resort — raw numeric values only
    return df[NUMERIC].values.astype(float)


def predict_one(name: str, df: pd.DataFrame) -> dict:
    """Run inference for a single model. Returns dict with prediction + probability."""
    model, pipe = load(name)

    X = safe_transform(pipe, df)

    if name == "Deep Learning (ANN)":
        import tensorflow as tf
        X = np.array(X).astype("float32")

        raw = model.predict(X, verbose=0)

        if raw.ndim == 1 or raw.shape[1] == 1:
            prob_pos = float(raw.flatten()[0])
            prob_neg = 1.0 - prob_pos
            proba = {
                "Regulated Drug":     round(prob_pos * 100, 2),
                "Non-Regulated Drug": round(prob_neg * 100, 2),
            }
            pred = "Regulated Drug" if prob_pos >= 0.5 else "Non-Regulated Drug"
        else:
            try:
                classes = pipe.named_steps["label_encoder"].classes_
            except Exception:
                classes = ["Non-Regulated Drug", "Regulated Drug"]
            probs = raw[0]
            proba = {str(classes[i]): round(float(probs[i]) * 100, 2) for i in range(len(classes))}
            pred = str(classes[int(np.argmax(probs))])
    else:
        raw_pred = model.predict(X)
        pred = str(raw_pred[0] if isinstance(raw_pred, (list, np.ndarray)) else raw_pred)

        proba = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            classes = model.classes_
            proba = {str(classes[i]): round(float(p[i]) * 100, 2) for i in range(len(classes))}

    return {"prediction": pred, "probability": proba}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/models/status")
def model_status():
    status = {}
    for name, (model_file, pipe_file) in REGISTRY.items():
        model_path = os.path.join(BASE, model_file)
        pipe_path  = os.path.join(BASE, pipe_file)
        status[name] = {
            "ready":         os.path.exists(model_path) and os.path.exists(pipe_path),
            "model_file":    model_file,
            "pipeline_file": pipe_file,
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
        df     = build_row(features)
        result = predict_one(model_name, df)
        return jsonify({"model": model_name, **result})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/predict/all", methods=["POST"])
def predict_all():
    body     = request.get_json(force=True)
    features = body.get("features", {})

    try:
        df = build_row(features)
    except Exception as e:
        return jsonify({"error": f"Feature build failed: {e}"}), 400

    votes  = {}
    models = {}

    for name in REGISTRY:
        try:
            result = predict_one(name, df)
            pred   = result["prediction"]
            votes[pred] = votes.get(pred, 0) + 1
            models[name] = {"available": True, **result}
        except Exception as e:
            models[name] = {"available": False, "error": str(e)}

    if not votes:
        return jsonify({"error": "No models produced a prediction"}), 500

    ensemble    = max(votes, key=votes.get)
    total_voted = sum(votes.values())
    confidence  = round((votes[ensemble] / total_voted) * 100, 2)

    return jsonify({
        "ensemble":    ensemble,
        "confidence":  confidence,
        "votes":       votes,
        "total_voted": total_voted,
        "models":      models,
    })


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("\n✅  PharmAI Flask server  →  http://127.0.0.1:5050\n")
    app.run(port=5050, debug=True, use_reloader=False)
