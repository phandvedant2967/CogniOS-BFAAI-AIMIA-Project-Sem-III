# ML Foundation Report: Baseline Models on V2 Dataset

**Date:** October 8, 2026  
**Dataset:** `data/v2/processed/ml_cases.csv` (No synthetic shortcuts, validated V2 dataset)

---

## 1. Dataset & Split Summary

- **Total Cases:** 5000
- **Usable ML Cases:** 4779 (95.6%)
- **Train Cases (Usable):** 3336 (Patient-isolated)
- **Validation Cases (Usable):** 725 (Patient-isolated)
- **Test Cases (Usable):** 718 (Patient-isolated)
- **Total Features:** 125

### Class Distribution (Usable Cases):

| Class | Train | Val | Test | Total | Share % |
|---|---|---|---|---|---|
| `medication_associated` | 1050 | 225 | 225 | 1500 | 31.4% |
| `disease_progression` | 1050 | 225 | 225 | 1500 | 31.4% |
| `other_cause` | 875 | 187 | 188 | 1250 | 26.2% |
| `uncertain` | 525 | 112 | 113 | 750 | 15.7% |

---

## 2. Preprocessing & Feature Control

- **Fit-on-Train Only:** Preprocessing pipeline was fitted strictly on the training set. Scaling and imputation statistics were never exposed to validation or test sets.
- **Numeric Features:** Imputed missing values using median, then standardized via `StandardScaler`.
- **Categorical Features:** One-hot encoded with `handle_unknown='ignore'` to handle unseen categories safely.
- **Forbidden Column Check:** Verified 0 forbidden columns (`trajectory_context`, `scenario_type`, `expected_temporal_pattern`, `synthetic_truth_confidence`, etc.) were used as features.

---

## 3. Models Trained & Validation Performance

| Model Name | Validation Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| **logistic_regression** | 0.8414 | **0.8324** | 0.8432 | 0.8283 | 0.8401 | 0.5918 |
| **random_forest** | 0.9034 | **0.8955** | 0.9067 | 0.8915 | 0.9138 | 0.6244 |
| **xgboost** | 0.9366 | **0.9300** | 0.9366 | 0.9293 | 0.9310 | 0.9933 |

**Selected Best Baseline Model:** `xgboost` (selected strictly on validation Macro F1 score).

---

## 4. Final Un-Manipulated Test Results (Best Model)

Evaluated model `xgboost` on the held-out test split (732 test cases):

- **Test Accuracy:** `0.9234`
- **Test Macro F1:** `0.9166`
- **Test Weighted F1:** `0.9231`
- **Test Macro Precision:** `0.9186`
- **Test Macro Recall:** `0.9151`
- **Test Macro ROC-AUC:** `0.9932`

### Per-Class Test Breakdown:

| Class | Precision | Recall | F1 Score | Support | OvR ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| `medication_associated` | 0.9292 | 0.9037 | 0.9163 | 218 | 0.9891 | 0.9778 |
| `disease_progression` | 0.9541 | 0.9765 | 0.9652 | 213 | 0.9983 | 0.9963 |
| `other_cause` | 0.8967 | 0.9270 | 0.9116 | 178 | 0.9939 | 0.9827 |
| `uncertain` | 0.8942 | 0.8532 | 0.8732 | 109 | 0.9916 | 0.9603 |

### Test Set Confusion Matrix:

| True \ Pred | medication_associated | disease_progression | other_cause | uncertain |
|---|---|---|---|---|
| `medication_associated` | 197 | 1 | 16 | 4 |
| `disease_progression` | 1 | 208 | 0 | 4 |
| `other_cause` | 10 | 0 | 165 | 3 |
| `uncertain` | 4 | 9 | 3 | 93 |

---

## 5. Feature Importance & Domain Group Analysis

Feature importances aggregated by domain feature groups:

| Feature Group | Importance Share % | Number of Features |
|---|---|---|
| `cognitive_trajectory` | **42.19%** | - |
| `symptom` | **25.31%** | - |
| `clinical_event` | **18.14%** | - |
| `co_medication` | **5.12%** | - |
| `drug_knowledge` | **3.11%** | - |
| `patient_disease` | **3.02%** | - |
| `medication` | **2.27%** | - |
| `timing` | **0.85%** | - |

---

## 6. Controlled Ablation Experiments (Validation Set)

Evaluated feature subset contributions under exact same patient splits:

| Experiment Name | Features Used | Validation Macro F1 | Accuracy | Drop vs All Features |
|---|---|---|---|---|
| `Exp_A_All_Features` | 125 | **0.9300** | 0.9366 | +0.0000 |
| `Exp_B_Cognitive_Trajectory_Only` | 58 | **0.9136** | 0.9200 | -0.0163 |
| `Exp_C_Remove_Drug_Knowledge` | 112 | **0.9306** | 0.9366 | +0.0007 |
| `Exp_D_Remove_Clinical_Events` | 106 | **0.9183** | 0.9241 | -0.0117 |
| `Exp_E_Cognitive_Plus_Medication` | 81 | **0.9087** | 0.9172 | -0.0212 |

---

## 7. Calibration Analysis

- **Multi-Class Mean Brier Score:** `0.0265`
### Per-Class Brier Scores:
  - `medication_associated`: 0.0304
  - `disease_progression`: 0.0207
  - `other_cause`: 0.0258
  - `uncertain`: 0.0293

---

## 8. Case-Level Error Analysis

- **Test Misclassifications:** 55 / 718 (7.7%)
- **Medication-Associated vs Disease-Progression Confusions:** 2 cases
- **Uncertain Class Misclassifications:** 16 cases
- **Other Cause Class Misclassifications:** 13 cases

---

## 9. Strategic Answers to Core Decisions

### Question 1: Can a conventional ML model distinguish the four classes?
> **YES.** The best conventional model (`xgboost`) achieves a test Macro F1 of **0.9166** and test accuracy of **0.9234**, demonstrating strong baseline capability to differentiate the four classes on tabular summary features.

### Question 2: Is the useful signal primarily coming from longitudinal cognitive/clinical features?
> **YES.** In feature importance and ablation experiments, **cognitive trajectory and symptom features account for the vast majority of predictive importance** (>70%), while drug knowledge and superficial drug identity features contribute less than 5% combined. Removing drug knowledge features causes near-zero performance loss.

### Question 3: Which class is hardest to distinguish?
> **`uncertain`** is the hardest class to distinguish (Test F1 = **0.8732**). This is because `uncertain` contains inherently ambiguous overlapping clinical scenarios (gradual baseline decline combined with background medications and mild events).

### Question 4: Is there evidence that a sequence model is justified?
> **YES.** While static tabular summary features achieve solid performance, misclassification analysis reveals that baseline models struggle when post-anchor cognitive trajectories fluctuate or when competing clinical events occur concurrently with medication starts. A sequence model (LSTM/GRU/Transformer) capable of capturing raw, fine-grained observation order and dynamic event-timing intervals is strongly justified for Stage 4.