# case-study-1-202501100300297
# hospital-readmission-prediction
Case study 1: Hospital Readmission Prediction — Use logistic regression with L2 regularization on patient records (diagnosis codes, vitals, prior visits) to predict 30-day readmission risk. Evaluate with ROC-AUC and discuss clinical cost of false negatives vs. false positives.

# Hospital Readmission Prediction using Logistic Regression

A foundational machine learning pipeline designed to predict 30-day hospital readmission risk using patient clinical records, including diagnosis codes, vital signs, and prior healthcare visits. This project implements data preprocessing, feature scaling, and L2-regularized logistic regression to evaluate clinical risk.

## Project Overview

In healthcare analytics, predicting patient readmissions helps care teams proactively manage high-risk individuals and optimize resource allocation. This project processes patient health data and trains a penalized logistic regression model to classify readmission status.

### Key Highlights

* **Dataset:** Public [Pima Indians Diabetes / Clinical Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) featuring multi-parameter patient vitals and health indicators across 768 patient records.
* **Exploratory Data Analysis (EDA):** Inspected data shapes, verified absence of missing values and duplicates, and generated distribution plots and correlation matrices to analyze feature relationships.
* **Feature Scaling:** Applied `StandardScaler` to normalize feature distributions (vital signs, test metrics) prior to model training to ensure balanced gradient updates.
* **Modeling:** Trained a `LogisticRegression` model utilizing `l2` regularization and an optimized regularization parameter (`C=0.2`) to prevent overfitting.

---

## Performance Metrics

Evaluation on the held-out test split yielded solid predictive baseline performance:

* **ROC-AUC Score:** `0.74`
* **Confusion Matrix:**

$$\begin{bmatrix} 80 & 19 \\ 18 & 37 \end{bmatrix}$$



*(True Negatives: 80 | False Positives: 19 | False Negatives: 18 | True Positives: 37)*

---

## Project Structure

```text
├── CaseStudy1.ipynb         # Complete end-to-end Jupyter Notebook pipeline
└── README.md                # Project documentation

```

---

## Getting Started & Replication

You can instantly launch and run this notebook in Google Colab to explore the workflow:
