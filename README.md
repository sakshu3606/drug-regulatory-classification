# 🧬 PharmAI — Drug Regulatory Intelligence Platform

> **Predict whether a pharmaceutical drug is Regulated or Non-Regulated using an ensemble of 7 Machine Learning models — with real-time classification, confidence scoring, and interactive analytics.**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-PharmAI-blue?style=for-the-badge)](https://pharmai-drug-classification-1.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red?style=flat-square)](https://xgboost.readthedocs.io/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?style=flat-square&logo=powerbi)](https://powerbi.microsoft.com/)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render)](https://render.com/)

---

## 📌 Project Overview

**PharmAI** is an end-to-end pharmaceutical data intelligence platform built to classify drugs as **Regulated** or **Non-Regulated** based on 29 clinical, financial, and distribution features. The platform combines rigorous data analysis, multiple trained ML models, ensemble voting logic, and an interactive web interface — enabling analysts and stakeholders to make fast, data-driven regulatory decisions.

---

## 🌐 Live Application

🔗 **[https://pharmai-drug-classification-1.onrender.com/](https://pharmai-drug-classification-1.onrender.com/)**

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Target Variable** | Drug Regulatory Status — `Regulated` / `Non-Regulated` |
| **Total Features** | 29 (23 Numeric + 6 Categorical) |
| **Key Features** | Dosage, Price Per Unit, Clinical Trial Phase, Side Effect Severity, Abuse Potential, Regulatory Risk Score, Prescription Rate, R&D Investment, Recall History, Adverse Event Reports, Therapeutic Class, Manufacturing Region, and more |

---

## 🤖 ML Models Trained

| Model | Notebook | Status |
|---|---|---|
| Logistic Regression | `logistic_regression.ipynb` | ✅ Deployed |
| Decision Tree | `decision_tree.ipynb` | ✅ Deployed |
| Random Forest | `random_forest.ipynb` | ✅ Deployed |
| K-Nearest Neighbors (KNN) | `knn_model.ipynb` | ✅ Deployed |
| Support Vector Machine (SVM) | `svm_model.ipynb` | ✅ Deployed |
| XGBoost | `xgboost.ipynb` | ✅ Deployed |
| Deep Learning (ANN) | `dl_ann.ipynb` | ✅ Deployed |

> **Ensemble Voting** — All 7 models can run simultaneously with majority vote classification for maximum prediction confidence.

---

## ⚙️ Project Workflow

```
Raw Dataset
    │
    ▼
Exploratory Data Analysis (EDA)
    │   exporation_data_analysis_EDA.ipynb
    ▼
Data Preprocessing & Feature Engineering
    │   preprocess_pipeline.ipynb → .pkl pipelines per model
    ▼
Model Training & Evaluation
    │   7 individual model notebooks → serialized .pkl files
    ▼
Web Application (Interactive UI)
    │   Single model prediction + Ensemble voting
    ▼
Power BI Dashboard
    │   Pharmaceutical Report.pbix
    ▼
Deployed on Render
    │   https://pharmai-drug-classification-1.onrender.com/
```

---

## 📁 Repository Structure

```
drug-regulatory-classification/
│
├── 📓 Notebooks
│   ├── exporation_data_analysis_EDA.ipynb     # EDA & feature insights
│   ├── preprocess_pipeline.ipynb              # Data cleaning & transformation
│   ├── logistic_regression.ipynb
│   ├── decision_tree.ipynb
│   ├── random_forest.ipynb
│   ├── knn_model.ipynb
│   ├── svm_model.ipynb
│   ├── xgboost.ipynb
│   └── dl_ann.ipynb                           # Deep Learning (ANN)
│
├── 🤖 Trained Models (.pkl)
│   ├── logistic_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── xgboost_model.pkl
│   └── deep_learning_ANN_model.pkl
│
├── 🔧 Preprocessing Pipelines (.pkl)
│   ├── preprocess_pipeline.pkl
│   ├── preprocess_pipeline_dt.pkl
│   ├── preprocess_pipeline_rf.pkl
│   └── preprocess_pipeline_ann.pkl
│
├── 📊 Analytics
│   └── Pharmaceutical Report.pbix             # Power BI Dashboard
│
├── 🌐 Application
│   ├── app.py                                  # Backend server
│   └── index.html                             # Frontend UI
│
├── 📦 Deployment
│   ├── requirements.txt
│   ├── Procfile
│   ├── render.yaml
│   └── runtime.txt
│
└── 📄 drug_regulatory_classification_dataset.csv
```

---

## 🔬 Features of the Platform

- **Single Model Mode** — Choose any one of 7 models to run classification individually
- **Ensemble Voting Mode** — All models predict simultaneously; majority vote determines final output
- **29-Feature Input Panel** — Numeric sliders + categorical selectors for full drug profile entry
- **Sample Data Loader** — One-click to populate a sample drug record for quick testing
- **Model Registry** — Live status of all serialized model files on the server
- **Prediction History** — Tracks all classification runs within the session

---

## 📈 Power BI Dashboard

The repository includes a **Pharmaceutical Report.pbix** Power BI dashboard for visual analytics on drug data, regulatory trends, distribution patterns, and market metrics — enabling stakeholder-ready data storytelling alongside the ML platform.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Analysis & EDA | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn, XGBoost |
| Deep Learning | TensorFlow / Keras (ANN) |
| Data Visualization | Power BI |
| Deployment | Render |

---

## 🚀 Getting Started (Local)

```bash
# 1. Clone the repository
git clone https://github.com/sakshu3606/drug-regulatory-classification.git
cd drug-regulatory-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

Then open `http://localhost:5000` in your browser.

---
