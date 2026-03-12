# PharmAI — Drug Regulatory Classification

> Predict whether a pharmaceutical drug is **Regulated** or **Non-Regulated** using a suite of 6 trained machine learning models with single-model precision and ensemble voting.


---

## Overview

PharmAI is an end-to-end drug intelligence platform built to support pharmaceutical compliance and data-driven regulatory decision-making. The system accepts 29 drug properties as input — spanning clinical, financial, distribution, and risk dimensions and returns a regulatory classification prediction using one or more trained ML models.

The platform is designed to assist analysts, researchers, and compliance teams in evaluating drug profiles quickly and accurately without manual review overhead.

---

## Features

- Classify drugs as **Regulated** or **Non-Regulated** based on 29 pharmaceutical features
- Choose between **6 individual ML models** or run **Ensemble Voting** across all models simultaneously
- Input 23 numeric features (dosage, pricing, clinical trial phase, abuse potential, etc.) and 6 categorical features (drug form, therapeutic class, manufacturing region, etc.)
- Real-time prediction with a live **Prediction History** tracker
- Model Registry panel displaying the status of all serialized `.pkl` models on the server
- Sample data loader for quick demonstration and testing

---

## Machine Learning Models

| Model | Type |
|---|---|
| Logistic Regression | Linear Classifier |
| Decision Tree | Tree-based Classifier |
| Random Forest | Ensemble — Bagging |
| K-Nearest Neighbors (KNN) | Instance-based Classifier |
| Support Vector Machine (SVM) | Margin-based Classifier |
| XGBoost | Ensemble — Boosting |

Each model was trained, evaluated, and serialized independently. The ensemble mode aggregates predictions from all 6 models using majority voting to produce a more robust final output.

---

## Dataset

- **File:** `drug_regulatory_classification_dataset.csv`
- **Features:** 29 total - 23 numeric, 6 categorical
- **Target:** Binary classification - `Regulated` / `Non-Regulated`

**Feature Categories:**

- *Clinical:* Clinical Trial Phase, Side Effect Severity, Adverse Event Reports, Abuse Potential, Recall History Count
- *Financial:* Price Per Unit, Production Cost, Marketing Spend, R&D Investment, Annual Sales Volume
- *Distribution:* Hospital Distribution %, Pharmacy Distribution %, Online Sales %, Export Percentage %, Insurance Coverage %
- *Regulatory:* Regulatory Risk Score, Prescription Rate, Approval Time, Patent Duration
- *Categorical:* Drug Form, Therapeutic Class, Manufacturing Region, Requires Cold Storage, OTC availability, High Risk Substance flag

---

## Project Structure

```
drug-regulatory-classification/
│
├── exporation_data_analysis_EDA.ipynb     # Exploratory Data Analysis
├── preprocess_pipeline.ipynb              # Data preprocessing and feature engineering
│
├── logistic_regression.ipynb              # Logistic Regression training
├── decision_tree.ipynb                    # Decision Tree training
├── random_forest.ipynb                    # Random Forest training
├── knn_model.ipynb                        # KNN training
├── svm_model.ipynb                        # SVM training
├── xgboost.ipynb                          # XGBoost training
│
├── logistic_model.pkl                     # Serialized Logistic Regression model
├── decision_tree_model.pkl                # Serialized Decision Tree model
├── random_forest_model.pkl                # Serialized Random Forest model
├── knn_model.pkl                          # Serialized KNN model
├── svm_model.pkl                          # Serialized SVM model
├── xgboost_model.pkl                      # Serialized XGBoost model
│
├── preprocess_pipeline.pkl                # Shared preprocessing pipeline
├── preprocess_pipeline_dt.pkl             # Decision Tree preprocessing pipeline
├── preprocess_pipeline_rf.pkl             # Random Forest preprocessing pipeline
│
├── drug_regulatory_classification_dataset.csv
├── Pharmaceutical Report.pbix             # Power BI Dashboard
├── app.py                                 # Application backend
├── index.html                             # Frontend interface
├── requirements.txt
├── render.yaml
└── Procfile
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Analysis & EDA | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Data Visualization | Power BI |
| Deployment | Render |

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/sakshu3606/drug-regulatory-classification.git
cd drug-regulatory-classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the application**
```bash
python app.py
```

**4. Open in browser**
```
http://localhost:5000
```

---

## Power BI Dashboard

The repository includes a `Pharmaceutical Report.pbix` Power BI dashboard that visualizes key metrics from the drug dataset including regulatory distribution, therapeutic class breakdowns, risk scoring trends, and clinical trial phase analysis enabling stakeholders to monitor drug compliance patterns at a glance.

---


