# Version 2 Synthetic Dataset Audit Report

**Date:** October 8, 2026  
**Pipeline Output:** `data/v2/processed/ml_cases.csv` (5,000 total cases, 4,826 usable ML cases)  
**Database Path:** `data/v2/ehr.db`  
**Raw Dataset Path:** `data/v2/raw/`  

---

## Executive Summary

The Version 2 (V2) synthetic dataset successfully eliminates the major synthetic shortcuts and class-leakage flaws identified in Stage 2C while preserving the project's domain concept, schema, scale (5,000 cases), and four target classes.

### Key Corrections Achieved:
1. **Drug Distribution Shortcut Eliminated:** Anchor drug Cramér's V dropped from **0.22 to 0.015**. High-risk drugs (cognitive_risk_index $\ge$ 6) now account for **33.2%** of ground-truth associated drugs, perfectly matching their prevalence in all medication records (33.5%).
2. **Clinical Event Presence Shortcut Removed:** Every target class now features an identical baseline of **20.0% zero-event cases** and an average of 1.57–1.59 clinical events per patient. `other_cause` is no longer separable by event presence alone.
3. **Longitudinal Logic & Pre/Post Coverage Fixed:** For `medication_associated` cases, **99.9%** (1,449 / 1,450 usable cases) now anchor on the true ground-truth associated drug (up from 28.8% in V1), with complete baseline assessment coverage.
4. **Trajectory Signal Dominance:** Top univariate predictors are now legitimate cognitive trajectory features (`feat_mmse_post_mean` AUC 0.803, `feat_knowledge_concordant_decline` AUC 0.781, `feat_mmse_delta` AUC 0.778) rather than superficial drug or event flags.

**Conclusion:** The corrected V2 dataset is **READY FOR FIRST MODEL TRAINING**.

---

## 1. Class Counts & Usability Breakdown

Across all 5,000 generated cases, usability criteria require $\ge 1$ baseline assessment in the 270 days prior to anchor and $\ge 2$ assessments in the 180 days post-anchor.

| Class | Total Cases | Usable Cases | Usable % | Train / Val / Test (Usable) |
|---|---|---|---|---|
| `medication_associated` | 1,500 | 1,450 | 96.7% | 1,012 / 216 / 222 |
| `disease_progression` | 1,500 | 1,433 | 95.5% | 1,004 / 213 / 216 |
| `other_cause` | 1,250 | 1,211 | 96.9% | 848 / 179 / 184 |
| `uncertain` | 750 | 732 | 97.6% | 512 / 110 / 110 |
| **Total** | **5,000** | **4,826** | **96.5%** | **3,376 / 718 / 732** |

Usability rate is uniform across all four classes (95.5%–97.6%), eliminating missingness/unusability as a class indicator.

---

## 2. Drug Distribution Across Classes

In V1, high-risk drugs were assigned almost exclusively to `medication_associated` cases (99.8%). In V2, drugs across all risk levels and therapeutic areas were drawn with equal probability across all four target classes.

### Drug Risk & Identity Metrics (Usable Cases):
- **Cramér's V (Anchor Drug Name):** `0.015` (V1 was `0.220`)
- **Cramér's V (Therapeutic Area):** `0.012` (V1 was `0.140`)
- **Lookup Purity:** `0.301` vs **Permutation Null:** `0.301` (Zero lookup signal)
- **High-Risk Drug Share (CRI $\ge$ 6):** `33.2%` in `medication_associated` vs `33.5%` across all medication records.

### Anchor Therapeutic Area Breakdown (% by Class):

| Therapeutic Area | `medication_associated` | `disease_progression` | `other_cause` | `uncertain` |
|---|---|---|---|---|
| Cardiovascular | 20.2% | 20.1% | 20.9% | 19.9% |
| Cognitive | 16.8% | 15.5% | 16.8% | 16.0% |
| Diabetes | 8.3% | 8.2% | 7.8% | 7.8% |
| Neurological | 19.4% | 20.1% | 19.9% | 19.4% |
| Other | 4.0% | 3.6% | 3.6% | 4.0% |
| Psychiatric | 27.2% | 28.8% | 27.2% | 28.7% |
| Urological | 4.1% | 3.7% | 3.8% | 4.2% |

---

## 3. Clinical Event Presence by Class

In V1, `other_cause` cases was 100% identifiable by event presence (3.5 events/patient vs 0% zero events). In V2, event frequency and severity were balanced across all classes, while `other_cause` retains competing clinical events aligned temporally with the cognitive drop.

### Event Statistics:

| Metric | `medication_associated` | `disease_progression` | `other_cause` | `uncertain` |
|---|---|---|---|---|
| **Zero Events (% Patients)** | 20.0% (300/1500) | 20.0% (300/1500) | 19.6% (245/1250) | 19.9% (149/750) |
| **Mean Events / Patient** | 1.57 | 1.58 | 1.58 | 1.59 |
| **Window Event Presence** | 41.0% | 42.8% | 55.0% | 41.3% |
| **Delirium / Stroke / UTI Alignment** | Minor | Minor | Competing Cause | Minor |

---

## 4. Longitudinal Logic & Anchor Hit Rate Verification

### Anchor Verification:
- **Pre-Anchor Assessments (Mean):** 3.28 to 3.92 assessments per patient across classes.
- **Post-Anchor Assessments (Mean):** 5.56 to 6.20 assessments per patient across classes.
- **Ground-Truth Associated Drug Anchor Match Rate:** **99.9%** (1,449 / 1,450 usable `medication_associated` cases).
- **Usable Pre/Post Observation Rate for Associated Drug:** **100.0%** of medication-associated cases have baseline pre-exposure observations and post-exposure assessments.

### Class Trajectory Dynamics (Mean MMSE Change across Anchor):
- `medication_associated`: **-3.68 points** (baseline flat $\rightarrow$ post-exposure decline)
- `disease_progression`: **-2.30 points** (ongoing steady decline starting before anchor)
- `other_cause`: **-1.99 points** (drop tied to competing event window)
- `uncertain`: **-1.58 points** (mild drop with noisy/ambiguous baseline)

---

## 5. Shortcut & Leakage Audit

A univariate feature screen (one-vs-rest AUC on the training split, no model fitted) confirms that trajectory features now drive class differentiation:

### Top Univariate Features by AUC (Train Split):

| Feature Name | Feature Group | Train Split AUC | Target Class | Missing Gap |
|---|---|---|---|---|
| `feat_mmse_post_mean` | Trajectory | **0.803** | `medication_associated` | 0.0% |
| `feat_knowledge_concordant_decline` | Knowledge/Trajectory | **0.781** | `medication_associated` | 0.0% |
| `feat_mmse_delta` | Trajectory | **0.778** | `medication_associated` | 0.0% |
| `feat_mmse_delta_early` | Trajectory | **0.762** | `medication_associated` | 0.0% |
| `feat_mmse_delta_late` | Trajectory | **0.758** | `medication_associated` | 0.0% |
| `feat_mmse_post_slope30` | Trajectory | **0.751** | `medication_associated` | 0.0% |
| `feat_memory_post_mean` | Trajectory | **0.726** | `medication_associated` | 0.0% |
| `feat_attention_post_mean` | Trajectory | **0.720** | `medication_associated` | 0.0% |
| `feat_executive_post_mean` | Trajectory | **0.718** | `medication_associated` | 0.0% |
| `feat_memory_delta` | Trajectory | **0.714** | `medication_associated` | 0.0% |

### Superficial Feature Leakage Check:
- **Drug Identity / Risk Features:** Cramér's V $\le 0.017$ across all drug features.
- **Clinical Event Count / Presence:** Zero event rate is equalized at 20%. Window max severity AUC $\le 0.61$.
- **Demographic / Comorbidity Features:** AUCs between 0.49 and 0.51.

---

## 6. Readiness Declaration

### Summary Assessment:
- [x] **Drug distribution shortcut eliminated:** High-risk drug bias removed; Cramér's V $\le 0.015$.
- [x] **Clinical event shortcut eliminated:** Zero-event baseline equalized at 20% across all classes.
- [x] **Longitudinal pre/post logic verified:** 99.9% anchor hit rate on medication cases with full pre/post coverage.
- [x] **Trajectory signals dominate:** Model must rely on temporal cognitive trajectory and competing event relationships.
- [x] **Reproducible generation:** Script `scripts/generate_v2_data.py` reproduces dataset from seed 42.

**FINAL DECLARATION:** The Version 2 synthetic dataset is **READY FOR FIRST MODEL TRAINING**.
