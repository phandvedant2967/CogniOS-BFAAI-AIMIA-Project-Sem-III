# V4.2 Model Training and Comparative Evaluation Report

## 1. Executive Summary

- **Cohort**: V4.2 Corrected EHR Cohort (5,000 cases, exactly 1,250 per class balanced)
- **Patient Splits**: Train = 3500 (875/class), Validation = 748 (187/class), Held-Out Test = 752 (188/class)
- **Follow-up Assessment Horizon**: Strictly capped at 180 days post-medication anchor. No future data or labels used.
- **Preprocessing**: Fitted on train split only; SimpleImputer with missingness indicators and StandardScaler.

## 2. Key Performance Comparison (Final Held-Out Test Set)

| Metric | XGBoost Baseline | Longitudinal 1D CNN-GRU | Delta (CNN-GRU vs XGBoost) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.3418 | 0.3644 | +0.0226 |
| **Macro-F1** | 0.3417 | 0.3641 | +0.0225 |
| **Weighted-F1** | 0.3417 | 0.3641 | +0.0225 |
| **Macro-Precision** | 0.3428 | 0.3671 | +0.0243 |
| **Macro-Recall** | 0.3418 | 0.3644 | +0.0226 |
| **Macro ROC-AUC** | 0.6236 | 0.6126 | -0.0110 |

## 3. Per-Class Test Performance

### XGBoost Baseline

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| `medication_associated` | 0.5829 | 0.5798 | 0.5813 |
| `disease_progression` | 0.2679 | 0.2394 | 0.2528 |
| `other_cause` | 0.2523 | 0.2926 | 0.2709 |
| `uncertain` | 0.2682 | 0.2553 | 0.2616 |

### Longitudinal 1D CNN-GRU

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| `medication_associated` | 0.5506 | 0.5213 | 0.5355 |
| `disease_progression` | 0.3370 | 0.3298 | 0.3333 |
| `other_cause` | 0.2803 | 0.2340 | 0.2551 |
| `uncertain` | 0.3004 | 0.3723 | 0.3325 |

## 4. Confusion Matrices (Final Held-Out Test Set)

### XGBoost Confusion Matrix

| True \ Pred | Med-Assoc | Progression | Other Cause | Uncertain |
| :--- | :---: | :---: | :---: | :---: |
| `medication_associated` | 109 | 25 | 32 | 22 |
| `disease_progression` | 23 | 45 | 65 | 55 |
| `other_cause` | 27 | 52 | 55 | 54 |
| `uncertain` | 28 | 46 | 66 | 48 |

### CNN-GRU Confusion Matrix

| True \ Pred | Med-Assoc | Progression | Other Cause | Uncertain |
| :--- | :---: | :---: | :---: |
| `medication_associated` | 98 | 28 | 27 | 35 |
| `disease_progression` | 26 | 62 | 45 | 55 |
| `other_cause` | 24 | 47 | 44 | 73 |
| `uncertain` | 30 | 47 | 41 | 70 |

## 5. Overfitting and Calibration Analysis

- **Validation vs Test Generalization**:
  - XGBoost: Validation Macro-F1 = 0.3793 vs Final Test Macro-F1 = 0.3417
  - CNN-GRU: Validation Macro-F1 = 0.3711 vs Final Test Macro-F1 = 0.3641
  - Stable generalization indicates well-regularized training without severe overfitting.
- **Learning Curves**: Monitored loss and Macro-F1 trajectories saved in `results/v4_2_learning_curves.png`.
- **Calibration**: Multi-class reliability curves saved in `results/v4_2_calibration.png`.

## 6. Pipeline Artifacts and Verification

- **Models Saved**:
  - `models/v4_2_xgboost.joblib`
  - `models/v4_2_cnn_gru.pt`
  - `models/v4_2_preprocessor.joblib`
- **Metrics File**: `results/v4_2_metrics.json`
- **Data Integrity Verified**: Ground-truth class balance (1,250 cases/class), 180-day follow-up assessment cutoff strictly enforced, missingness indicators fitted on training data.
