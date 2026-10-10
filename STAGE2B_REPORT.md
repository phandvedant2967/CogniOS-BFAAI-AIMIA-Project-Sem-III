# Stage 2B Report: Ground-Truth Labels and Leakage Audit

Script: `scripts/stage2b_ground_truth.py` (read-only on the DB and raw files; no model, no fixes).
Output: `data/processed/ground_truth_labels.csv` (5,000 rows).
Dataset 6 was copied unchanged into `data/raw/dataset6_Ground_Truth_Cases.xlsx`.

## 1. Matching
- 5,000 cases, 5,000 unique patients. All match `patients`, and all have cognitive and medication data.
- All 1,500 medication-associated cases point to a medication the patient actually has (ID and drug name agree).
- All 1,250 other-cause cases name an event type that exists in that patient's clinical events.

`ground_truth_labels.csv` has: case id, patient_id, `label`, `split`, `gt_associated_medication_id` (kept only for later medication-level work, not a feature), and the match flags. The label-describing text columns (scenario, expected pattern, alternative cause, confidence) are deliberately left out.

## 2. Class counts
| Class | Cases | train / val / test |
|---|---|---|
| medication_associated | 1,500 | 1050 / 225 / 225 |
| disease_progression | 1,500 | 1050 / 225 / 225 |
| other_cause | 1,250 | 875 / 188 / 187 |
| uncertain | 750 | 525 / 112 / 113 |

Splits are stratified and patient-level, so no patient appears in two splits.

## 3. Leakage / shortcut findings
| Finding | Severity |
|---|---|
| **`trajectory_context`**: disease_progression has 0 rows with a value. Any value means "not progression". Share of patients with at least one value: medication 47%, other 23%, uncertain 13%, progression 0%. | High (already excluded from Stage 2 files; must stay out) |
| **Ground-truth columns**: `scenario_type` and `expected_temporal_pattern` identify the class with 100% accuracy. `associated_medication_id/drug` is filled only for medication cases, and `alternative_clinical_cause` only for other-cause cases. `synthetic_truth_confidence` ranges 0.45-0.62 for uncertain vs 0.76-0.96 for every other class. | High (dataset README already forbids use as features) |
| **Number of clinical events per patient**: other_cause patients always have at least 1 event (mean 3.5, 0% with none). Medication and progression patients average ~1.5 (about 16% with none). Uncertain averages 2.7. `cognitive_impact_possible=Yes` is 70% for other/uncertain vs 43-49% otherwise. Event type mix and clinical notes are similar across classes. | High: other_cause is separable by event count alone |
| **Medication mix**: medication/uncertain patients have more psychiatric (25%/23% vs 14%) and urological (5% vs 0.5%) drugs; progression patients have more cardiovascular drugs (35% vs 23%). Adherence is the same across classes. | Medium (partly legitimate, but reflects how the cases were built) |
| **Pre/post observations around the medication** (anchor = associated drug for medication cases, a random drug for others): the number of pre/post assessments is similar (2.7 vs 3.0-3.1 pre, 7.1-7.8 post), and about 54-60% have a pre-start assessment in every class. MMSE drop across the anchor is larger for medication cases (-1.46 vs -0.90 progression, -1.28 other, -0.82 uncertain). | Low-Medium: no structural observation-count shortcut, but the larger drop is a built-in signal, so a model would partly learn how the data was generated |

Also note: the Stage 2 observation file has patient-level "days since last medication start/stop" for any drug, with no drug linked to the observation. Medication cases appear to have their start placed inside the assessment window, which is a possible timing shortcut worth checking after fixes.

## 4. Safe for model training?
**Not yet.** The labels join cleanly and class sizes are reasonable, but the features would let a model succeed through shortcuts.

Must be fixed first:
1. Keep `trajectory_context` and every ground-truth column (including `synthetic_truth_confidence`, scenario, expected pattern, associated drug, alternative cause) out of features. Use them only for evaluation and audit.
2. Remove or neutralise event-count features (total events, events per patient, `cognitive_impact_possible` rates), since they identify other_cause/uncertain without any temporal reasoning.
3. Decide how to handle the drug-mix imbalance (for example, do not use drug identity or class as a global feature; use only drug properties relative to the timeline).
4. Build features around a per-case anchor (medication start, dose change, stop) and apply the same procedure to all classes, then re-check pre/post balance.
5. Treat results as a test of recovering a synthetic generator, not of real-world causality. Report metrics per class and for uncertain separately.
