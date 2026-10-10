# Stage 4D: Generalization & Diverse Trajectory Evaluation Report

## Executive Summary

In Stage 4B and 4C, audit findings revealed that the initial **99.72% test accuracy** achieved on the synthetic V2 cohort was driven by rigid mathematical trajectory artifacts ($\sigma = 0.4$ flat baselines, constant slopes, and textbook U-dips). When subjected to a noisy, realistic stress cohort, model performance plummeted to **~40% macro-F1**. 

In **Stage 4D**, we addressed the root cause of this generalization collapse by fundamentally retiring rigid mathematical generation templates. We developed a reproducible **Version 3 (V3)** longitudinal generator comprising **5,000 synthetic patients**, featuring:
1. **Realistic Trajectory Diversity & Biological Noise:** Non-linear stochastic cognitive decline with variance $\sigma \in [1.2, 2.4]$.
2. **Pre-Existing Baseline Decline:** 48% of medication-associated patients exhibit prior gradual cognitive decline before medication exposure, mirroring real-world polypharmacy in dementia cohorts.
3. **Incomplete Recovery & Competing Causes:** Acute clinical events (e.g., delirium, stroke, trauma) leave residual cognitive deficits in 65% of cases rather than reverting to a textbook baseline.
4. **Stochastic Multi-Medication Dynamics:** Unrelated background medication starts, dose changes, and dechallenges occur across all four classes.
5. **Irregular Observation Intervals & Missingness:** Irregular visit gaps (90–210 days) and realistic ~15% missing cognitive subscores (Executive Function and Mood).

Both models were trained on the V3 training split (3,500 patients) and tuned strictly on the validation split (748 patients). The fresh held-out test cohort (752 patients) was evaluated **strictly once** after all decisions were finalized.

---

## 1. Dataset Breakdown & Cohort Structure

| Cohort Split | Usable Cases | Medication-Associated | Disease Progression | Other Clinical Cause | Uncertain | Share % |
|---|---|---|---|---|---|---|
| **Train** | 3,500 | 875 | 875 | 875 | 875 | 70.0% |
| **Validation** | 748 | 187 | 187 | 187 | 187 | 15.0% |
| **Final-Test (Fresh Held-Out)** | 752 | 188 | 188 | 188 | 188 | 15.0% |
| **Total Cohort** | **5,000** | **1,250** | **1,250** | **1,250** | **1,250** | **100.0%** |

- **Strict Patient Isolation:** Zero patient overlap between splits.
- **Fair Target Distribution:** Perfectly balanced 25.0% representation per class across all splits.
- **Protected Split:** The fresh held-out `final_test` cohort was never accessed for preprocessing parameters, training, early stopping, or hyperparameter selection.

---

## 2. Validation Model Selection

Model selection was conducted strictly using the V3 validation split:

| Model Architecture | Input Representation | Val Loss / LogLoss | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Val Macro ROC-AUC |
|---|---|---|---|---|---|---|
| **XGBoost Baseline** | 148 Engineered Tabular Features | 0.8306 | 65.24% | 0.6472 | 0.6472 | 0.8654 |
| **Longitudinal CNN-GRU** | 41 Raw Temporal Steps $\times$ Features | **0.6659** | **81.95%** | **0.8193** | **0.8193** | **0.9576** |

### Early Stopping & Learning Dynamics (CNN-GRU)
- **Epochs Trained:** 35 epochs (batch size 64, AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, label smoothing 0.05).
- **Best Validation Checkpoint:** Saved at **Epoch 29** (`models/v3_cnn_gru.pt`) with validation macro-F1 of **0.8193** and validation loss of **0.6659**.
- **Overfitting / Underfitting Assessment:**
  - Training loss steadily converged from 1.37 to 0.30; training macro-F1 reached 0.9637.
  - Validation loss plateaued between epochs 15 and 30 (0.6508 to 0.6659) without diverging into runaway overfitting.
  - The combination of 1D CNN feature extraction, single-layer GRU temporal aggregation, and 0.30 dropout provided strong regularization against noise.

---

## 3. Final Held-Out Test Evaluation (Strictly Once)

Both frozen candidate models were evaluated on the 752 fresh held-out patients:

### Summary Comparison Table
| Metric | XGBoost Baseline | Longitudinal CNN-GRU | Delta ($\Delta$) |
|---|---|---|---|
| **Test Accuracy** | 69.55% | **81.78%** | **+12.23%** |
| **Macro F1 Score** | 0.6917 | **0.8183** | **+0.1266** |
| **Weighted F1 Score** | 0.6917 | **0.8183** | **+0.1266** |
| **Macro Precision** | 0.6902 | **0.8200** | **+0.1298** |
| **Macro Recall** | 0.6955 | **0.8178** | **+0.1223** |
| **Macro OvR ROC-AUC** | 0.8916 | **0.9546** | **+0.0630** |

---

## 4. Per-Class Performance Breakdown (Final-Test)

### Longitudinal CNN-GRU (Best Selected Model)
| Target Class | Support | Precision | Recall | F1 Score | OvR ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| `medication_associated` | 188 | 0.7688 | 0.8138 | **0.7907** | 0.9474 | 0.8770 |
| `disease_progression` | 188 | 0.8081 | 0.8511 | **0.8290** | 0.9602 | 0.8999 |
| `other_cause` | 188 | 0.9602 | 0.8989 | **0.9286** | 0.9934 | 0.9846 |
| `uncertain` | 188 | 0.7430 | 0.7074 | **0.7248** | 0.9174 | 0.8129 |
| **Macro Average** | **752** | **0.8200** | **0.8178** | **0.8183** | **0.9546** | **0.8936** |

### XGBoost Baseline
| Target Class | Support | Precision | Recall | F1 Score | OvR ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| `medication_associated` | 188 | 0.6931 | 0.6968 | **0.6950** | 0.8902 | 0.7815 |
| `disease_progression` | 188 | 0.7214 | 0.7713 | **0.7455** | 0.9272 | 0.7904 |
| `other_cause` | 188 | 0.7600 | 0.8085 | **0.7835** | 0.9531 | 0.8839 |
| `uncertain` | 188 | 0.5864 | 0.5053 | **0.5429** | 0.7959 | 0.5981 |
| **Macro Average** | **752** | **0.6902** | **0.6955** | **0.6917** | **0.8916** | **0.7635** |

---

## 5. Confusion Matrix Analysis (Final-Test)

### Longitudinal CNN-GRU Confusion Matrix
| True \\ Predicted | Medication-Associated | Disease Progression | Other Clinical Cause | Uncertain | Recall % |
|---|---|---|---|---|---|
| **Medication-Associated** | **153** | 11 | 2 | 22 | **81.4%** |
| **Disease Progression** | 8 | **160** | 1 | 19 | **85.1%** |
| **Other Clinical Cause** | 7 | 7 | **169** | 5 | **89.9%** |
| **Uncertain** | 31 | 20 | 4 | **133** | **70.7%** |

### XGBoost Confusion Matrix
| True \\ Predicted | Medication-Associated | Disease Progression | Other Clinical Cause | Uncertain | Recall % |
|---|---|---|---|---|---|
| **Medication-Associated** | **131** | 13 | 14 | 30 | **69.7%** |
| **Disease Progression** | 10 | **145** | 5 | 28 | **77.1%** |
| **Other Clinical Cause** | 17 | 10 | **152** | 9 | **80.9%** |
| **Uncertain** | 31 | 33 | 29 | **95** | **50.5%** |

### Clinical Observations from Error Patterns
1. **Separation of Medication-Associated vs. Disease Progression:**
   - CNN-GRU achieved high discrimination between `medication_associated` and `disease_progression`: only 11 out of 188 true medication cases were misclassified as disease progression, and only 8 true progression cases were misclassified as medication.
   - In contrast, XGBoost misclassified 13 medication cases as disease progression and 10 vice versa.
2. **Identification of Acute Events (`other_cause`):**
   - The sequence model excelled at detecting acute clinical events, achieving **0.9286 F1** (169/188 recalled, 96.0% precision). The 1D CNN local filters reliably detect abrupt temporal inflection points coupled with acute event indicators.
3. **The Clinical Ambiguity of `uncertain`:**
   - `uncertain` remains the most difficult class for both models (CNN-GRU F1 = **0.7248**, XGBoost F1 = **0.5429**). 
   - True `uncertain` cases represent borderline trajectories where cognitive drops overlap with mild medical events or unconfirmed medication associations. 31 `uncertain` cases were classified as medication-associated, and 20 as disease progression. This behavior accurately mirrors clinical practice, where subtle or conflicting presentations pose diagnostic challenges.

---

## 6. Key Scientific Findings & Generalization Insights

1. **Why CNN-GRU Substantially Outperformed XGBoost (+12.2% Accuracy, +0.126 F1):**
   - In V2, flat baselines and constant slopes allowed static tabular aggregate features (e.g., pre/post MMSE difference, overall slope) to separate classes easily.
   - In V3, pre-existing decline, non-linear trajectories, and irregular intervals destroyed the reliability of aggregate summary statistics. XGBoost's static averages could not determine *when* an acceleration occurred relative to drug titration.
   - The **1D CNN + GRU** processes the chronological step-by-step observation sequence directly, distinguishing between steady ongoing decline and acute post-prescription dropoffs even against noisy backgrounds.

2. **Honest Synthetic Generalization:**
   - The models did not achieve an artificial 99.7% metric on V3. Instead, the realistic, overlapping distributions produced an honest **81.8% macro-F1** for CNN-GRU and **69.2%** for XGBoost.
   - The calibration curve (`results/v3_calibration.png`) demonstrates that predicted probabilities closely align with empirical frequencies across all four classes.

3. **Remaining Limitations:**
   - This evaluation is based on synthetic longitudinal EHR data. While the generator incorporates biological noise, missingness, and competing clinical causes, real-world EHR data features even higher unmeasured confounding (e.g., social determinants of health, compliance variability, variable testing intervals).
   - Real-world clinical validation requires calibration against real patient registries.

---

## 7. Artifact Manifest

- **Generator:** `scripts/generate_v3_data.py` (5,000-patient diverse cohort generator & pipeline runner)
- **Database:** `data/v3/ehr.db` (clean SQLite schema with 5,000 patients, 49,961 observations)
- **Processed Datasets:** `data/v3/processed/` (`ml_cases.csv`, `patient_observations.csv`, `medication_episodes.csv`)
- **Training Script:** `scripts/train_v3_models.py`
- **Saved Checkpoints:**
  - `models/v3_cnn_gru.pt` (Best validation checkpoint, epoch 29)
  - `models/v3_xgboost.joblib` (Trained XGBoost model)
  - `models/v3_preprocessor.joblib` (Fitted tabular preprocessor)
- **Evaluation Visualizations:**
  - `results/v3_learning_curves.png` (Training & validation loss and macro-F1 curves)
  - `results/v3_confusion_matrices.png` (Side-by-side test confusion heatmaps)
  - `results/v3_calibration.png` (Per-class probability calibration curves)
- **Metrics JSON:** `results/v3_metrics.json`
