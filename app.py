from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys, gc
import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings("ignore")

# ── TensorFlow: lazy import only (NOT at startup) ─────────────────────────────
# Importing TF at startup wastes ~300-400MB on Render's 512MB free tier.
# It is imported on-demand only when ANN is actually requested.
TF_AVAILABLE = None   # None = not yet checked; True/False after first check

def _check_tf():
    global TF_AVAILABLE
    if TF_AVAILABLE is not None:
        return TF_AVAILABLE
    try:
        import tensorflow as _tf  # noqa: F401
        TF_AVAILABLE = True
    except Exception:
        TF_AVAILABLE = False
    return TF_AVAILABLE


app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── SafeWinsorizer classes ────────────────────────────────────────────────────
from sklearn.base import BaseEstimator, TransformerMixin

class SafeWinsorizer(BaseEstimator, TransformerMixin):
    """Winsorizer wrapper for Decision Tree pipeline (original column names)."""
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        from feature_engine.outliers import Winsorizer
        X_c = X.copy()
        self.medians_ = {col: float(X_c[col].median()) for col in self.variables}
        for col in self.variables:
            X_c[col] = X_c[col].fillna(self.medians_[col])
        self.winsorizer_ = Winsorizer(
            capping_method="iqr", tail="both", fold=1.5, variables=self.variables
        )
        self.winsorizer_.fit(X_c)
        return self

    def transform(self, X):
        X_c = X.copy()
        for col in self.variables:
            X_c[col] = X_c[col].fillna(self.medians_.get(col, 0))
        result = self.winsorizer_.transform(X_c)
        for col in self.variables:
            if result[col].isna().any():
                result[col] = result[col].fillna(self.medians_.get(col, 0))
        return result


class SafeWinsorizerLower(BaseEstimator, TransformerMixin):
    """Winsorizer wrapper for Random Forest pipeline (lowercase column names)."""
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        from feature_engine.outliers import Winsorizer
        X_c = X.copy()
        self.medians_ = {col: float(X_c[col].median()) for col in self.variables}
        for col in self.variables:
            X_c[col] = X_c[col].fillna(self.medians_[col])
        self.winsorizer_ = Winsorizer(
            capping_method="iqr", tail="both", fold=1.5, variables=self.variables
        )
        self.winsorizer_.fit(X_c)
        return self

    def transform(self, X):
        X_c = X.copy()
        for col in self.variables:
            X_c[col] = X_c[col].fillna(self.medians_.get(col, 0))
        result = self.winsorizer_.transform(X_c)
        for col in self.variables:
            if result[col].isna().any():
                result[col] = result[col].fillna(self.medians_.get(col, 0))
        return result


# ── Model Registry ────────────────────────────────────────────────────────────
REGISTRY = {
    "Logistic Regression": {
        "file":      "logistic_model.pkl",
        "col_case":  "original",
        "label_enc": False,
    },
    "Decision Tree": {
        "file":      "decision_tree_model.pkl",
        "col_case":  "original",
        "label_enc": True,
    },
    "Random Forest": {
        "file":      "random_forest_model.pkl",
        "col_case":  "lower",
        "label_enc": False,
    },
    "KNN": {
        "file":      "knn_model.pkl",
        "col_case":  "original",
        "label_enc": False,
    },
    "SVM": {
        "file":      "svm_model.pkl",
        "col_case":  "original",
        "label_enc": True,
    },
    "XGBoost": {
        "file":      "xgboost_model.pkl",
        "col_case":  "original",
        "label_enc": True,
    },
    "Deep Learning (ANN)": {
        "file":      "deep_learning_ANN_model.pkl",
        "col_case":  "original",
        "label_enc": True,
        "is_ann":    True,
        "ann_pipe":  "preprocess_pipeline_ann.pkl",
    },
}

# ── MEMORY: Only ONE model is kept in memory at a time ───────────────────────
# Render free tier has 512MB. Keeping all 7 models loaded simultaneously
# causes OOM crashes. We use a single-slot LRU-1 cache instead.
_cache = {}          # { name: model_object }
_CACHE_LIMIT = 1     # max models in memory simultaneously (increase if upgraded)

LABEL_MAP     = {0: "Non-Regulated Drug", 1: "Regulated Drug"}
ANN_LABEL_MAP = {0: "Non-Regulated Drug", 1: "Regulated Drug"}

# ── Column definitions ────────────────────────────────────────────────────────
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

ALL_ORIG          = NUMERIC_ORIG + CATEGORICAL_ORIG
NUMERIC_LOWER     = [c.lower() for c in NUMERIC_ORIG]
CATEGORICAL_LOWER = [c.lower() for c in CATEGORICAL_ORIG]
ALL_LOWER         = [c.lower() for c in ALL_ORIG]
NUMERIC_SET_LOWER = set(c.lower() for c in NUMERIC_ORIG)


# ── PKL Loader ────────────────────────────────────────────────────────────────
def _load_pkl(path):
    import joblib, pickle
    try:
        return joblib.load(path)
    except Exception:
        pass
    try:
        import dill
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin-1")
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e4:
        raise RuntimeError(
            f"Cannot load '{os.path.basename(path)}' with any loader. "
            f"Last error: {str(e4)[:300]}"
        )


def load_model(name):
    """
    Load a model with a single-slot cache.
    When a new model is requested, the old one is evicted and gc.collect() is
    called so Python frees the RAM before loading the next model.
    This keeps peak memory use at ~1 model at a time instead of 7.
    """
    global _cache

    if name in _cache:
        return _cache[name]

    cfg = REGISTRY[name]

    # ANN requires TF — check lazily
    if cfg.get("is_ann") and not _check_tf():
        raise RuntimeError("ANN requires TensorFlow which is not installed.")

    path = os.path.join(BASE, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{cfg['file']}' not found on server.")

    # Evict old models from cache if at limit
    if len(_cache) >= _CACHE_LIMIT:
        evict_names = list(_cache.keys())
        for evict in evict_names:
            del _cache[evict]
        gc.collect()   # actually release the RAM

    obj = _load_pkl(path)
    _cache[name] = obj
    return obj


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_row(features: dict, col_case: str) -> pd.DataFrame:
    norm = {}
    for k, v in features.items():
        norm[k.lower().strip()] = v

    if "rd_investment_million" in norm and "r&d_investment_million" not in norm:
        norm["r&d_investment_million"] = norm["rd_investment_million"]

    all_cols = ALL_ORIG if col_case == "original" else ALL_LOWER

    row = {}
    for col in all_cols:
        key = col.lower()
        val = norm.get(key, None)
        if key in NUMERIC_SET_LOWER:
            if val is None or val == "":
                val = 0.0
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
        else:
            val = "" if val is None else str(val).strip()
        row[col] = val

    return pd.DataFrame([row], columns=all_cols)


def decode_label(raw) -> str:
    try:
        idx = int(float(str(raw)))
        return LABEL_MAP.get(idx, str(raw))
    except Exception:
        return str(raw)


def _get_final_estimator(model):
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def _build_ann_input(df: pd.DataFrame) -> np.ndarray:
    for pipe_file in ["preprocess_pipeline_ann.pkl", "preprocess_pipeline.pkl"]:
        pipe_path = os.path.join(BASE, pipe_file)
        if os.path.exists(pipe_path):
            try:
                pipe = _load_pkl(pipe_path)
                X = pipe.transform(df)
                if hasattr(X, "toarray"):
                    X = X.toarray()
                return np.array(X, dtype="float32")
            except Exception as e:
                print(f"⚠️ {pipe_file} failed: {e}")
    numeric_cols = [c for c in df.columns if c.lower() in NUMERIC_SET_LOWER]
    return df[numeric_cols].values.astype("float32")


def _ann_predict(model, features: dict) -> dict:
    df  = build_row(features, "original")
    X   = _build_ann_input(df)
    raw = model.predict(X, verbose=0)
    if raw.ndim > 1 and raw.shape[1] > 1:
        probs = raw[0]
        idx   = int(np.argmax(probs))
        pred  = ANN_LABEL_MAP.get(idx, str(idx))
        proba = {ANN_LABEL_MAP.get(i, str(i)): round(float(probs[i]) * 100, 2)
                 for i in range(len(probs))}
    else:
        prob_pos = float(np.clip(raw.flatten()[0], 0.0, 1.0))
        pred  = "Regulated Drug" if prob_pos >= 0.5 else "Non-Regulated Drug"
        proba = {"Regulated Drug": round(prob_pos * 100, 2),
                 "Non-Regulated Drug": round((1.0 - prob_pos) * 100, 2)}
    return {"prediction": pred, "probability": proba}


def predict_one(name: str, features: dict) -> dict:
    cfg   = REGISTRY[name]
    model = load_model(name)

    if cfg.get("is_ann") and not hasattr(model, "named_steps"):
        return _ann_predict(model, features)

    df       = build_row(features, cfg["col_case"])
    raw_pred = model.predict(df)
    raw_val  = raw_pred[0] if isinstance(raw_pred, (list, np.ndarray)) else raw_pred
    pred     = decode_label(raw_val) if cfg["label_enc"] else str(raw_val)

    proba = None
    final = _get_final_estimator(model)
    if hasattr(final, "predict_proba"):
        try:
            p           = model.predict_proba(df)[0]
            classes     = list(final.classes_) if hasattr(final, "classes_") else list(range(len(p)))
            class_names = [decode_label(c) for c in classes] if cfg["label_enc"] else [str(c) for c in classes]
            proba       = {class_names[i]: round(float(p[i]) * 100, 2) for i in range(len(p))}
        except Exception as e:
            print(f"⚠️ predict_proba failed for {name}: {e}")
            proba = None

    return {"prediction": pred, "probability": proba}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/health")
def health():
    # Lightweight: does NOT load any models
    return jsonify({"status": "ok"})


@app.route("/models/status")
def model_status():
    # MEMORY SAFE: only check file existence, do NOT load models
    status = {}
    for name, cfg in REGISTRY.items():
        path   = os.path.join(BASE, cfg["file"])
        exists = os.path.exists(path)
        status[name] = {
            "ready":      exists,
            "model_file": cfg["file"],
            "cached":     name in _cache,
        }
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    body = {}
    try:
        body       = request.get_json(force=True, silent=True) or {}
        model_name = body.get("model", "Random Forest")
        features   = body.get("features", {})

        if not features:
            return jsonify({"error": "No features provided"}), 400
        if model_name not in REGISTRY:
            return jsonify({"error": f"Unknown model: '{model_name}'"}), 400

        result = predict_one(model_name, features)
        return jsonify({"model": model_name, **result})

    except Exception as e:
        full_trace = traceback.format_exc()
        print(f"PREDICT ERROR [{body.get('model','?')}]:\n{full_trace}")
        return jsonify({"error": str(e), "trace": full_trace}), 200


@app.route("/predict/all", methods=["POST"])
def predict_all():
    """
    MEMORY-SAFE ensemble: runs models one-at-a-time and evicts each before
    loading the next. Skips ANN on free tier to avoid TF memory spike.
    """
    try:
        body     = request.get_json(force=True, silent=True) or {}
        features = body.get("features", {})

        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Skip ANN in ensemble by default to stay within 512MB
        skip_ann = not bool(os.environ.get("ENABLE_ANN_ENSEMBLE"))

        votes, models = {}, {}
        for name in REGISTRY:
            if skip_ann and REGISTRY[name].get("is_ann"):
                models[name] = {
                    "available": False,
                    "error": "ANN skipped in ensemble to conserve memory. "
                             "Use /predict with model='Deep Learning (ANN)' directly."
                }
                continue
            try:
                result       = predict_one(name, features)
                pred         = result["prediction"]
                votes[pred]  = votes.get(pred, 0) + 1
                models[name] = {"available": True, **result}
            except Exception as e:
                models[name] = {"available": False, "error": str(e)}
            finally:
                # Evict immediately after each prediction to free RAM
                if name in _cache:
                    del _cache[name]
                gc.collect()

        if not votes:
            return jsonify({"error": "No models predicted", "models": models}), 200

        ensemble    = max(votes, key=votes.get)
        total_voted = sum(votes.values())

        return jsonify({
            "ensemble":    ensemble,
            "confidence":  round((votes[ensemble] / total_voted) * 100, 2),
            "votes":       votes,
            "total_voted": total_voted,
            "models":      models,
        })

    except Exception as e:
        full_trace = traceback.format_exc()
        print(f"PREDICT/ALL ERROR:\n{full_trace}")
        return jsonify({"error": str(e), "trace": full_trace}), 200


@app.route("/debug")
def debug():
    import platform
    info = {
        "python":   sys.version,
        "platform": platform.platform(),
        "base":     BASE,
        "cache":    list(_cache.keys()),
    }

    pkgs = {}
    for pkg in ["feature_engine", "sklearn", "xgboost", "joblib",
                "pandas", "numpy", "flask", "dill"]:
        try:
            m = __import__(pkg)
            pkgs[pkg] = getattr(m, "__version__", "ok")
        except ImportError as e:
            pkgs[pkg] = f"MISSING: {e}"
    info["packages"] = pkgs

    # MEMORY SAFE: only check file existence, do NOT load models
    pkl_status = {}
    for name, cfg in REGISTRY.items():
        path = os.path.join(BASE, cfg["file"])
        if not os.path.exists(path):
            pkl_status[name] = "❌ FILE MISSING"
        else:
            size_kb = os.path.getsize(path) / 1024
            pkl_status[name] = f"✅ FILE OK ({size_kb:.1f} KB)"
    info["pkl_files"] = pkl_status

    return jsonify(info)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    port  = int(os.environ.get("PORT", 5050))
    host  = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    debug = not bool(os.environ.get("RENDER"))
    print(f"\n✅  PharmAI Flask server  →  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
