# Stage 4B: Robustness Audit Report

**Date:** October 9, 2026  
**Audited Models:**  
- Frozen 1D CNN + GRU (`models/longitudinal_cnn_gru.pt`)  
- Frozen XGBoost Baseline (`models/baseline_xgboost.pkl`, `models/preprocessing.pkl`)  
**Datasets:**  
- V2 Test Split (`data/v2/processed/`, $N=718$)  
- Independent Stress-Test Development Cohort (`data/stress_test/processed/`, $N=600$ dev cases)

---

## 1. Code & Leakage Inspection Findings

We performed a line-by-line inspection of [`ml/sequence_interface.py`](file:///c:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/ml/sequence_interface.py), [`ml/sequence_model.py`](file:///c:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/ml/sequence_model.py), and [`scripts/train_sequence_model.py`](file:///c:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/scripts/train_sequence_model.py).

### Inspection Verification Matrix:
| Audit Dimension | Verification Method | Finding | Status |
|---|---|---|---|
| **Target Label Leakage** | Audited feature vector dictionary in `SequenceFeatureBuilder` | No `target`, `gt_*`, `scenario_type`, or class label is present in the sequence input tensor. | **CLEAN** |
| **Future Information Leakage** | Checked time indexing at observation step $t_{\text{obs}}$ | All medication states, cumulative dose, and clinical events are evaluated strictly at $t \le t_{\text{obs}}$. Post-horizon data is not accessible at earlier steps. | **CLEAN** |
| **Ground-Truth Descriptor Fields** | Scanned for forbidden keywords (`doctor`, `associated_medication`, `alternative_clinical_cause`) | None present. Anchor selection rule is purely algorithmic (`balanced` anchor rule, identical across all 4 classes). | **CLEAN** |
| **Patient Split Isolation** | Verified patient IDs across `train`, `val`, `test` splits | Patient sets are completely disjoint across splits ($0\%$ overlap). Normalizers (means, standard deviations, and imputation medians) were computed **strictly on the training split**. | **CLEAN** |

**Conclusion on Programmatic Leakage:**  
There is no programmatic software bug, metadata exposure, or future leakage in the data loader, model definition, or evaluation loop.

---

## 2. Why Were the Original V2 Results So High (99.72%)?

In Stage 4, the sequence model attained **99.72% test accuracy** and **0.9974 macro-F1**, while the trajectory-only ablation (masking all medication and event features) attained **0.9963 macro-F1**.

Our audit traced the cause directly to the mathematical equations used in `scripts/generate_v2_data.py` (lines 320–380):

1. **`medication_associated`**:  
   - $t \le t_{\text{anchor}}$: MMSE slope is perfectly flat ($0.0 \pm 0.4$).
   - $t > t_{\text{anchor}}$: MMSE initiates a sharp, permanent linear decline proportional to $(t - t_{\text{anchor}})$.
2. **`disease_progression`**:  
   - MMSE decreases at a continuous linear rate across the entire duration ($\approx -2.5$ points/year), completely invariant to $t_{\text{anchor}}$.
3. **`other_cause`**:  
   - Pre-anchor is flat; an acute dip ($3.5$ points) occurs strictly in the window $[t_{\text{anchor}}+10, t_{\text{anchor}}+150]$, after which the score recovers back to within $0.5$ points of baseline.
4. **`uncertain`**:  
   - Low-slope noise ($1.0$ point/year) with minimal fluctuation.

### The Trajectory-Shape Shortcut:
Because the synthetic data generation used distinct, deterministic parametric curves with small Gaussian noise ($\sigma = 0.4$), each class had a **mathematical curvature fingerprint**:
- *Flat then steep decline* $\implies$ `medication_associated`
- *Constant non-zero slope across all visits* $\implies$ `disease_progression`
- *U-shaped drop and recovery* $\implies$ `other_cause`
- *Mild drift* $\implies$ `uncertain`

The 1D CNN + GRU model discovered this curve-fitting shortcut with near-perfection. It did not need to cross-reference drug identities or clinical events because the synthetic cognitive trajectory alone was nearly deterministic of the class label.

---

## 3. The Independent Stress-Test Cohort (`data/stress_test/`)

To test whether the model learned robust clinical relationships or merely memorized synthetic curve shapes, we generated an independent stress-test cohort ($N = 1,200$ cases; 300 per class; $100\%$ usable cases) under `data/stress_test/`:
- **Development Portion (`split='dev'`):** 600 cases (150 per class) — reserved for diagnosis, ablation, and tuning.
- **Final Test Portion (`split='final_test'`):** 600 cases (150 per class) — strictly held out.

### Realism & Stress Factors Introduced:
1. **Realistic Trajectory Overlaps:**
   - 40% of `medication_associated` patients have pre-existing cognitive decline prior to drug start.
   - Variable onset latency (14 to 90 days post-drug).
   - In 65% of `other_cause` patients, acute events leave permanent cognitive deficits (incomplete recovery), mimicking medication-induced decline unless temporal alignment with the acute event is recognized.
   - `disease_progression` trajectories exhibit non-linear fluctuations, temporary plateaus, and steeper deterioration spells.
2. **Irregular Observation Gaps:**
   - Visits occur at random non-uniform intervals (25 to 110 days apart) instead of fixed grids.
3. **Missing Clinical Assessments:**
   - 15.4% missing executive function scores and 15.0% missing mood scores.
4. **Competing Clinical Events:**
   - All classes experience realistic mixtures of clinical events (falls, UTIs, metabolic episodes) and concurrent prescription titrations.

---

## 4. Frozen Models Evaluation on Stress-Test Development Cohort ($N=600$)

We evaluated the frozen Stage 4 CNN-GRU checkpoint (`models/longitudinal_cnn_gru.pt`) and frozen Stage 3A XGBoost baseline (`models/baseline_xgboost.pkl`) on the stress-test development cohort. Neither model was retrained.

### Empirical Performance Comparison:
| Metric | Original V2 Test Split ($N=718$) | Stress-Test Dev Split ($N=600$) | Absolute Delta | Status |
|---|---|---|---|---|
| **CNN-GRU Accuracy** | 0.9972 | **0.3717** | **-0.6255** | Severe Collapse |
| **CNN-GRU Macro F1** | 0.9974 | **0.3782** | **-0.6192** | Severe Collapse |
| **XGBoost Accuracy** | 0.9234 | **0.4267** | **-0.4967** | Severe Collapse |
| **XGBoost Macro F1** | 0.9166 | **0.4131** | **-0.5035** | Severe Collapse |

### Per-Class F1 Breakdown on Stress-Test Dev Cohort:
| Target Class | Support (Dev) | Frozen XGBoost F1 | Frozen CNN-GRU F1 |
|---|---|---|---|
| `medication_associated` | 150 | 0.3459 | 0.3313 |
| `disease_progression` | 150 | 0.6521 | 0.3742 |
| `other_cause` | 150 | 0.3904 | 0.5469 |
| `uncertain` | 150 | 0.2642 | 0.2603 |
| **Macro Average** | **600** | **0.4131** | **0.3782** |

### Confusion Matrix — Frozen CNN-GRU on Stress Dev:
| True Class \ Predicted Class | `med_assoc` | `disease_prog` | `other_cause` | `uncertain` | Total | Recall % |
|---|---|---|---|---|---|---|
| **`medication_associated`** | **54** | 43 | 16 | 37 | 150 | 36.0% |
| **`disease_progression`** | 34 | **61** | 15 | 40 | 150 | 40.7% |
| **`other_cause`** | 34 | 19 | **70** | 27 | 150 | 46.7% |
| **`uncertain`** | 54 | 53 | 5 | **38** | 150 | 25.3% |

### Key Error Patterns:
1. **Medication-Associated Misclassified as Disease Progression (43 cases):**  
   Because 40% of stress-test medication patients had mild pre-existing decline before drug exposure, the frozen CNN-GRU (which expected a perfectly flat pre-anchor baseline) mistook them for pure neurodegenerative disease progression.
2. **Other-Cause Misclassified as Medication-Associated (34 cases):**  
   When acute medical events resulted in permanent incomplete cognitive recovery, the model mistook the permanent deficit for a medication-induced drop.
3. **XGBoost vs. CNN-GRU Trade-off:**  
   XGBoost held up slightly better overall (0.4131 vs 0.3782 macro-F1), especially on `disease_progression` (0.6521 vs 0.3742), because its pre/post slope ratio features were somewhat more noise-tolerant than the CNN-GRU's rigid convolutional filters. However, CNN-GRU recognized `other_cause` better (0.5469 vs 0.3904) by detecting localized time-alignment with acute event steps.

---

## 5. Summary & Audit Verdict

1. **Does the original 99.72% result generalize?**  
   **No.** The 99.72% accuracy achieved in Stage 4 was heavily inflated by idealized synthetic curve geometries. When evaluated on realistic biological noise and overlapping clinical trajectories, macro-F1 plummets to ~38%.
2. **Clinical Validity:**  
   Neither model possesses real-world clinical validity in its current frozen state. High metrics on synthetic benchmarks must not be conflated with real-world clinical efficacy.
3. **Action Plan for Stage 4C:**  
   - We must diagnose the fit of the CNN-GRU architecture.
   - We must train with stronger regularization (higher dropout, weight decay) and temporal feature augmentation so the network learns true multi-modal relationships rather than over-relying on smooth trajectory geometries.
   - All tuning must be guided by the `dev` split; the `final_test` split must remain strictly untouched until final evaluation.
