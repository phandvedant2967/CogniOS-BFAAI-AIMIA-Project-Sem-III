# Baseline Models Evaluation Report

Comparing Model A (Logistic Regression), Model B (Random Forest), and Model C (XGBoost).

## 1. Overall Performance Comparison (Validation Split)

| Model Name | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| **logistic_regression** | 0.8414 | 0.8324 | 0.8432 | 0.8283 | 0.8401 | 0.5918 |
| **random_forest** | 0.9034 | 0.8955 | 0.9067 | 0.8915 | 0.9138 | 0.6244 |
| **xgboost** | 0.9366 | 0.9300 | 0.9366 | 0.9293 | 0.9310 | 0.9933 |

## 2. Per-Class F1 Scores

| Model Name | medication_associated | disease_progression | other_cause | uncertain |
|---|---|---|---|---|
| **logistic_regression** | 0.8394 | 0.8942 | 0.8352 | 0.7607 |
| **random_forest** | 0.9183 | 0.9349 | 0.9116 | 0.8171 |
| **xgboost** | 0.9425 | 0.9569 | 0.9356 | 0.8848 |

## 3. Confusion Matrix (Best Model)

**Best Model:** `xgboost` (Macro F1 = 0.9300)

| True \ Pred | medication_associated | disease_progression | other_cause | uncertain |
|---|---|---|---|---|
| **medication_associated** | 205 | 5 | 7 | 3 |
| **disease_progression** | 2 | 211 | 0 | 4 |
| **other_cause** | 5 | 2 | 167 | 7 |
| **uncertain** | 3 | 6 | 2 | 96 |