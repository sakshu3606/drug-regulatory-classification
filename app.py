from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, sys
import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings("ignore")

# ── Auto-install missing packages ─────────────────────────────────────────────
import subprocess

def _pip_install(pkg, quiet=True):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
        capture_output=True, text=True
    )
    return result.returncode == 0

# feature-engine
try:
    import feature_engine
    print("✅ feature-engine ready:", feature_engine.__version__)
except ImportError:
    print("⏳ Installing feature-engine...")
    if _pip_install("feature-engine==1.6.2"):
        try:
            import feature_engine
            print("✅ feature-engine installed:", feature_engine.__version__)
        except ImportError:
            print("❌ feature-engine failed to install")
    else:
        print("❌ feature-engine pip install failed")

# pyarrow  — inject a comprehensive stub so pkl files saved with pyarrow
#            can be unpickled without pyarrow installed.
try:
    import pyarrow
    print("✅ pyarrow ready:", pyarrow.__version__)
except ImportError:
    import types as _types, sys as _sys

    def _make_stub_class(name):
        """Return a stub class that accepts any args/kwargs and does nothing."""
        return type(name, (), {
            "__init__":      lambda self, *a, **k: None,
            "__reduce__":    lambda self: (self.__class__, ()),
            "__repr__":      lambda self: f"<{name} stub>",
        })

    # Build the main pyarrow module
    _pa = _types.ModuleType("pyarrow")
    _pa.__version__  = "0.0.0-stub"
    _pa.__spec__     = None

    # Stub classes that pandas / joblib reference during pickle load
    for _cls in ["Array", "ChunkedArray", "Table", "RecordBatch", "Schema",
                 "Field", "DataType", "DictionaryType", "ListType", "MapType",
                 "StructType", "UnionType", "TimestampType", "Time32Type",
                 "Time64Type", "FixedSizeBinaryType", "Decimal128Type",
                 "BaseExtensionType", "ExtensionType", "NativeFile",
                 "Buffer", "MemoryPool", "lib"]:
        setattr(_pa, _cls, _make_stub_class(_cls))

    # Stub functions
    for _fn in ["array", "chunked_array", "table", "record_batch", "schema",
                "field", "from_pandas", "concat_tables", "concat_arrays",
                "nulls", "zeros"]:
        setattr(_pa, _fn, lambda *a, **k: None)

    # Register sub-modules
    for _sub in ["lib", "types", "compat", "pandas_compat", "array",
                 "table", "schema", "ipc", "compute", "dataset",
                 "parquet", "csv", "json", "feather", "flight",
                 "gandiva", "plasma", "orc", "fs"]:
        _submod = _types.ModuleType(f"pyarrow.{_sub}")
        # copy all stub attrs into submodule too
        for _attr in dir(_pa):
            if not _attr.startswith("__"):
                setattr(_submod, _attr, getattr(_pa, _attr))
        _sys.modules[f"pyarrow.{_sub}"] = _submod
        setattr(_pa, _sub, _submod)

    _sys.modules["pyarrow"] = _pa
    print("✅ pyarrow comprehensive stub injected")

# ── TensorFlow optional ───────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    import tensorflow as _tf
    TF_AVAILABLE = True
    print("✅ TensorFlow available:", _tf.__version__)
except Exception:
    # Try installing a version compatible with the running Python
    import sys as _sys
    _pymajor = _sys.version_info.major
    _pyminor = _sys.version_info.minor
    print(f"⏳ TF not found, Python {_pymajor}.{_pyminor} — trying compatible version...")
    # TF version support: <=3.11 → 2.13.0 | 3.12 → 2.16.0 | 3.13+ → not supported yet
    if _pyminor <= 11:
        _tf_ver = "tensorflow==2.13.0"
    elif _pyminor == 12:
        _tf_ver = "tensorflow==2.16.2"
    else:
        _tf_ver = None  # Python 3.13/3.14 - no TF wheel exists yet

    if _tf_ver and _pip_install(_tf_ver):
        try:
            import tensorflow as _tf
            TF_AVAILABLE = True
            print(f"✅ TensorFlow installed: {_tf.__version__}")
        except Exception as _e:
            print(f"❌ TF import failed after install: {_e}")
    else:
        print("⚠️  TensorFlow not available — ANN model disabled (unsupported Python version)")

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

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

LABEL_MAP     = {0: "Non-Regulated Drug", 1: "Regulated Drug"}
ANN_LABEL_MAP = {0: "Non-Regulated Drug", 1: "Regulated Drug"}

_cache = {}

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

    # Try 1: joblib
    try:
        return joblib.load(path)
    except Exception:
        pass

    # Try 2: dill
    try:
        import dill
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception:
        pass

    # Try 3: pickle latin-1
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin-1")
    except Exception:
        pass

    # Try 4: plain pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e4:
        raise RuntimeError(
            f"Cannot load '{os.path.basename(path)}' with any loader. "
            f"Last error: {str(e4)[:300]}"
        )


def load_model(name):
    if name in _cache:
        return _cache[name]
    cfg  = REGISTRY[name]

    if cfg.get("is_ann") and not TF_AVAILABLE:
        raise RuntimeError("ANN requires TensorFlow which is not installed.")

    path = os.path.join(BASE, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{cfg['file']}' not found on server.")

    obj = _load_pkl(path)
    _cache[name] = obj
    return obj


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
    return jsonify({"status": "ok"})


@app.route("/models/status")
def model_status():
    status = {}
    for name, cfg in REGISTRY.items():
        path   = os.path.join(BASE, cfg["file"])
        exists = os.path.exists(path)
        entry  = {"ready": exists, "model_file": cfg["file"]}
        if exists:
            try:
                _load_pkl(path)
                entry["loaded"] = True
            except Exception as e:
                entry["loaded"] = False
                entry["error"]  = str(e)[:200]
        status[name] = entry
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
        # Always return 200 with error info so frontend shows the message
        return jsonify({"error": str(e), "trace": full_trace}), 200


@app.route("/predict/all", methods=["POST"])
def predict_all():
    try:
        body     = request.get_json(force=True, silent=True) or {}
        features = body.get("features", {})

        if not features:
            return jsonify({"error": "No features provided"}), 400

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
    info = {"python": sys.version, "platform": platform.platform(), "base": BASE}

    pkgs = {}
    for pkg in ["feature_engine", "sklearn", "xgboost", "joblib",
                "pandas", "numpy", "flask", "dill"]:
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
            pkl_status[name] = "❌ FILE MISSING"
        else:
            try:
                _load_pkl(path)
                pkl_status[name] = "✅ OK"
            except Exception as e:
                pkl_status[name] = f"❌ {str(e)[:200]}"
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
