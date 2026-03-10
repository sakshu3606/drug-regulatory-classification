from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys
import pandas as pd
import numpy as np
import traceback
import warnings
import gc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY FIX 1: Do NOT auto-install packages at runtime.
#   Add all dependencies to requirements.txt instead and let Render install
#   them at build time.  Runtime pip installs waste ~50-100 MB of RAM.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import feature_engine
except ImportError:
    print("❌ feature-engine not installed. Add it to requirements.txt")

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
            capping_method="iqr", tail="both", fold=1.5,
            variables=self.variables
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
            capping_method="iqr", tail="both", fold=1.5,
            variables=self.variables
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
}

LABEL_MAP     = {0: "Non-Regulated Drug", 1: "Regulated Drug"}

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY FIX 3: Limit the in-memory model cache.
#   Keeping ALL 6 models loaded simultaneously (especially RF + XGBoost)
#   easily exceeds 512 MB.  We use an LRU-style cache that evicts the least
#   recently used model when the cap is reached, freeing memory via gc.collect().
# ─────────────────────────────────────────────────────────────────────────────
from collections import OrderedDict

# Max models kept in memory at once.  3 keeps us safely under 512 MB
# even with the larger sklearn models.
_MAX_CACHED = 3
_cache: OrderedDict = OrderedDict()


def _evict_if_needed():
    """Remove the oldest cached model when we're at capacity."""
    while len(_cache) >= _MAX_CACHED:
        evicted_name, _ = _cache.popitem(last=False)
        print(f"🗑  Evicted '{evicted_name}' from model cache to free RAM")
        gc.collect()


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
    """Load a model with LRU caching."""
    cfg = REGISTRY[name]


    if name in _cache:
        _cache.move_to_end(name)   # mark as recently used
        return _cache[name]

    path = os.path.join(BASE, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{cfg['file']}' not found on server.")

    _evict_if_needed()
    obj = _load_pkl(path)
    _cache[name] = obj
    _cache.move_to_end(name)
    return obj


# ── Helper utilities ──────────────────────────────────────────────────────────
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


def predict_one(name: str, features: dict) -> dict:
    cfg = REGISTRY[name]

    model    = load_model(name)
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

    return {"prediction": pred, "probability": proba}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY FIX 6: /models/status and /debug must NOT load every model.
#   They now only check whether the .pkl file EXISTS on disk.
#   Loading 7 models at once in a status check is what triggered the OOM spikes.
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/models/status")
def model_status():
    status = {}
    for name, cfg in REGISTRY.items():
        path = os.path.join(BASE, cfg["file"])
        status[name] = {
            "ready":      os.path.exists(path),
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
    """Run all models sequentially, evicting after each to stay under 512 MB."""
    try:
        body     = request.get_json(force=True, silent=True) or {}
        features = body.get("features", {})

        if not features:
            return jsonify({"error": "No features provided"}), 400

        votes, models = {}, {}

        # Run models sequentially with GC between each to stay under 512 MB
        ordered = list(REGISTRY.keys())

        for name in ordered:
            try:
                result       = predict_one(name, features)
                pred         = result["prediction"]
                votes[pred]  = votes.get(pred, 0) + 1
                models[name] = {"available": True, **result}
            except Exception as e:
                models[name] = {"available": False, "error": str(e)}
            finally:
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
        "python":    sys.version,
        "platform":  platform.platform(),
        "base":      BASE,
        "cache":     list(_cache.keys()),

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

    # Only check file existence — do NOT load models
    pkl_status = {}
    for name, cfg in REGISTRY.items():
        path = os.path.join(BASE, cfg["file"])
        pkl_status[name] = "✅ FILE EXISTS" if os.path.exists(path) else "❌ FILE MISSING"
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
