# Stage 2C Report: First case-level ML dataset

Scripts: `scripts/stage2c_build_ml_cases.py` (build), `scripts/stage2c_audit.py` (shortcut audit).
Output: `data/processed/ml_cases.csv` (5,000 rows, 1 per ground-truth case, 128 `feat_*` columns).
No model was trained. Raw datasets and `ehr.db` are unchanged (checksums verified). Dataset 6 labels are attached only after all features are built.

## 1. Usable cases
| Class | Cases | Usable | % | train / val / test (usable) |
|---|---|---|---|---|
| medication_associated | 1,500 | 1,252 | 83.5 | 881 / 191 / 180 |
| disease_progression | 1,500 | 1,209 | 80.6 | 847 / 190 / 172 |
| other_cause | 1,250 | 1,009 | 80.7 | 702 / 153 / 154 |
| uncertain | 750 | 628 | 83.7 | 433 / 96 / 99 |
| **Total** | **5,000** | **4,098 (82%)** | | **2,863 / 630 / 605** |

All 5,000 rows are kept; filter on `meta_usable_case == 1`. The 902 unusable cases have no feature values. Reasons: no baseline assessment before any medication event (421) and fewer than 2 post-anchor assessments (481). Usability is similar across classes (80-84%), so it does not create a class shortcut. Splits are the original patient-level splits.

## 2. Anchor logic (same for all four classes, no label used)
- **Candidates:** every medication start and every dose change, across all of the patient's records.
- **Eligible:** at least 1 cognitive assessment in the 270 days up to the anchor (same-day = baseline) and at least 2 in the 180 days after it.
- **Choice:** the eligible candidate with the best-supported windows (max of min(n_pre, n_post)); ties go to the earliest date, then record ID, start before dose change. No drug knowledge, outcome or label is used.
- **Anchor mix:** 12-14% dose-change anchors in three classes, 23% in uncertain.
- **Alternative:** `--anchor-rule max_cog_risk` picks the highest cognitive-risk drug first (same usable count, 4,098).

**Important limitation.** The case label refers to a specific "associated drug" for medication cases, but a label-free rule cannot know which drug that is. Audit only: the anchor equals the ground-truth drug in **28.8%** of usable medication cases (361/1,252), roughly chance with ~2.5 eligible anchors per patient. The ceiling is ~49% because the ground-truth drug is often started before the first assessment (no baseline). The risk-first rule raises the hit rate to 52%, but amplifies the drug shortcut (Cramér's V 0.32 vs 0.22), so I did not use it as the default. For most medication cases the features therefore describe a drug that is not the one that generated the label.

## 3. Features created (128, all `feat_*`; categoricals are plain strings, one-hot them at modelling time)
| Group | # | Content |
|---|---|---|
| Pre/post cognitive trajectory | 43 | For MMSE, memory, attention, executive, mood: last baseline value, pre-slope, post-mean, post-slope, post-minus-baseline delta, slope change, early (0-90d) and late (90-180d) delta. Plus MMSE decline beyond the pre-slope trend and days to first MMSE drop of 1.5 points. Slopes need at least 2 points at least 21 days apart. |
| Symptom flags | 12 | Pre rate, post rate and change for confusion, memory loss, attention problem, behavioral change |
| Timing | 2 | Days last baseline to anchor, anchor to first post assessment |
| Medication / dose | 12 | Dose-change vs start, dose, dose relative to that drug's median dose, change ratio, frequency, adherence, days already on drug, stop in post window (dechallenge) and MMSE change after stop, later dose change |
| Other active medications | 12 | Summed anticholinergic / sedation / cognitive-risk / interaction indices of co-medications, max risk, high-anticholinergic / high-sedation / cognitive-therapy flags, days to nearest other medication change or stop, riskiest competing change |
| Clinical events (windowed, no counts) | 19 | Max severity, cognitive-impact flag, hospitalization, days to nearest event in -90/+180d windows; event inside the pre/post assessment gap; max duration; 8 event-group flags; nearest event category |
| Patient / disease | 15 | Age, sex, diagnosis, stage, disease duration at anchor, 10 comorbidity flags |
| Drug knowledge (anchor drug) | 14 | Four indices, memory/confusion/attention/mood/sedation potentials, anticholinergic burden, interaction risk, evidence category, cognitive relevance, and "decline in domains the drug is known to affect" |

Excluded: `trajectory_context`, scenario, expected pattern, truth confidence, associated drug/cause columns (none appear in the CSV; the build asserts this). No doctor-label column exists in any of the six files. No clinical-event counts, observation counts, or enrollment-date offsets are features. Drug ID/name/class/area, anchor date, assessment counts and `n_eligible_anchors` are kept only as `meta_*` and must not be used as features.

## 4. Drug distribution across classes (anchor drug, usable cases)
**There is a strong synthetic drug shortcut.**
- **Ground-truth drug choice:** 99.8% of the ground-truth associated drugs have cognitive_risk_index >= 6, versus 32% of all medication records. Only 9 of 25 drugs are ever used (Gabapentin, Quetiapine, Pregabalin, Clonazepam, Oxybutynin, Amitriptyline, Diphenhydramine, Risperidone, Lorazepam). Valproate, Carbidopa/Levodopa and Levetiracetam (risk 4-6) are never used.
- **Anchor drug by class:** a high-risk anchor drug occurs in 43% of medication, 41% uncertain, 27% other_cause and only 13% of progression cases. For those 9 drugs, progression is just 2-8% of anchors against a 30% base rate. Low-risk drugs are 30-48% progression.
- **Patient level (rule independent):** psychiatric drugs are present in 65-67% of medication/uncertain patients vs 36% of progression; urological 19-21% vs 1%.
- **Strength:** anchor drug vs class Cramér's V = 0.22 (area 0.14). Lookup purity 0.40 vs permutation null 0.33. A single rule "anchor risk >= 6 => medication, else progression" gets 65% accuracy on medication vs progression. Knowledge indices (14 features) are effectively drug identity, so they carry the same shortcut.
- **Signal vs shortcut:** the strongest univariate features (train split, one-vs-rest AUC) are drug/co-medication risk (about 0.34-0.39 for progression) and event severity/cognitive-impact (about 0.65 for other_cause). Pure cognitive-trajectory features are weak: apart from MMSE change after stopping the drug (AUC 0.60 for uncertain, but only ~7% of cases have it), none deviates more than ~0.08 from 0.5 (MMSE delta, medication vs progression: 0.54).

**Recommendation: regenerate or rebalance before any model whose results will be interpreted.** Specifically:
1. Draw the associated drug for medication cases from the full drug list, and give progression, other_cause and uncertain cases high-risk drugs at the same rate, so drug risk is not informative about the class.
2. Give medication and progression cases clinical events at the same rate as other_cause/uncertain (see below).
3. If regeneration is not possible, a fallback is matched sampling or stratified evaluation by drug-risk tier. Drug identity and knowledge features should be dropped or ablated, and results reported with and without them. I did not regenerate or balance anything.

## 5. Remaining leakage / shortcut risks
| Risk | Severity |
|---|---|
| Drug mix and drug-knowledge features (section 4) | **High** |
| Clinical event presence: 100% of other_cause and uncertain patients have at least one event vs 83-84% of the others. After windowing, 83% vs 54-56% of cases have an event in the window. Severity, cognitive-impact, duration and event-group flags carry this. No counts are used, but presence still identifies those two classes (AUC ~0.65). | **High** |
| Co-medication risk features (patient-level drug mix, e.g. other_active_cogrisk_max AUC 0.34 for progression) | Medium-High |
| Anchor often not the associated drug (28.8% hit rate): dilutes the medication signal and makes the medication class noisy | Medium |
| Uncertain cases differ structurally: more dose-change anchors (23% vs 12-14%), more eligible anchors, shorter assessment gaps (53 vs 58-62 days at Stage 2B) | Medium |
| Missingness of optional features (e.g. days to first decline 62-69% missing, similar across classes) | Low |
| Population constants (dose medians, score SDs) use all patients without labels | Low |
| Synthetic data: results show recovery of a generator, not real causality | Inherent |

## 6. Ready for the first model?
**Structurally yes, scientifically not yet.** The file loads cleanly: 4,098 usable cases, patient-level splits, a separate `target`, no forbidden fields, all features anchor-local. It is fine for a **diagnostic baseline** (for example, to confirm the shortcut by ablation: trajectory-only vs +events vs +drug features). It is **not ready for any headline result**, because drug and event-presence shortcuts dominate and the trajectory signal is weak. Resolve section 4 (regenerate/rebalance) first, then rebuild with `python scripts/stage2c_build_ml_cases.py`. Evaluate per class, and treat `uncertain` separately.
