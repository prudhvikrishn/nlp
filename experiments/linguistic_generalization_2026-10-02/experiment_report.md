# Linguistic Generalization Experiment

Date: 2026-10-02

## Summary

A train-only enrichment run improved performance on every held-out evaluation split. The same calibrated LinearSVC and hybrid feature representation were retrained with 145 hand-authored training examples. The original project baseline, raw data, validation/test CSVs, behavioral suite, paraphrase set, and adversarial set were not modified. SHA-256 fingerprints are recorded in `manifest.json`; the runner checks the protected files again after each run.

The frozen 818-query test Macro-F1 moved from 95.75% to 96.68% (+0.93 percentage points). Frozen paraphrase Macro-F1 moved from 53.90% to 92.27% (+38.37 points); frozen adversarial Macro-F1 from 72.71% to 80.39% (+7.68 points). The new 80-query in-domain external-style set scored 50.00% baseline and 81.68% enriched (+31.68 points).

These are diagnostic results, not a claim of population-level generalization: the new set has only 10 queries per intent, was authored for this experiment, and is not an independently sampled customer-query benchmark. The additions and new evaluation were authored in the same task after the known class-level failures and prior audit query wording were visible. No existing evaluation rows were copied into training, but semantic influence cannot be ruled out. Treat all wording-focused gains as exploratory rather than blind confirmation; confirm them with queries authored independently without access to these examples.

## Training-data audit and enrichment

The training split had 3,856 rows. Class counts and sources:

| Intent | Existing training rows | Existing source note | Added rows |
|---|---:|---|---:|
| balance_inquiry | 226 | original queries | 20 |
| card_issue | 189 | synthetic-template only | 20 |
| credit_card_application | 223 | original queries | 15 |
| forgot_pin | 188 | synthetic-template only | 20 |
| fraud_report | 874 | original queries | 15 |
| loan_inquiry | 1118 | original queries | 15 |
| password_reset | 209 | original queries | 20 |
| transaction_query | 829 | original queries | 20 |

The imbalance and template dependence of `card_issue` / `forgot_pin` were the clearest training-set design concerns. The additions are separate hand-authored natural-language rows, include indirect descriptions and alternate banking vocabulary, and are assigned to training only. The original raw data and processed split CSVs remain intact. Exact normalized overlap checks against all three existing splits and all existing evaluation sets found none; the 90-row expanded set also has no exact overlap with those inputs or the additions.

## Evaluation metrics

All intent metrics below exclude OOS probes. The frozen paraphrase/adversarial sets each contain 40 in-domain queries and 5 OOS probes. The expanded set contains 80 in-domain queries (10 per intent) and 10 separate OOS probes.

| Dataset | N | Base accuracy | Base macro-P | Base macro-R | Base Macro-F1 | Base weighted-F1 | Enriched accuracy | Enriched macro-P | Enriched macro-R | Enriched Macro-F1 | Enriched weighted-F1 | Macro-F1 change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 836 | 97.61% | 97.72% | 96.87% | 97.28% | 97.60% | 97.97% | 97.68% | 97.48% | 97.57% | 97.96% | +0.29% |
| frozen_test | 818 | 96.70% | 97.41% | 94.36% | 95.75% | 96.67% | 97.19% | 98.15% | 95.37% | 96.68% | 97.17% | +0.93% |
| frozen_paraphrase | 40 | 52.50% | 75.22% | 52.50% | 53.90% | 53.90% | 92.50% | 94.35% | 92.50% | 92.27% | 92.27% | +38.37% |
| frozen_adversarial | 40 | 72.50% | 85.64% | 72.50% | 72.71% | 72.71% | 80.00% | 89.02% | 80.00% | 80.39% | 80.39% | +7.68% |
| expanded_external | 80 | 50.00% | 72.20% | 50.00% | 50.00% | 50.00% | 81.25% | 87.68% | 81.25% | 81.68% | 81.68% | +31.68% |

Training results (in-sample): baseline train N=3,856, Accuracy 99.82%, Macro Precision 99.77%, Macro Recall 99.86%, Macro-F1 99.82%, Weighted-F1 99.82%; enriched train N=4,001, Accuracy 99.80%, Macro Precision 99.77%, Macro Recall 99.85%, Macro-F1 99.81%, Weighted-F1 99.80%. Training-set Macro-F1 is an in-sample score and should not be interpreted as an independent generalization estimate. The model was not tuned or selected using any external evaluation scores. The classifier, feature kind, and hyperparameters were held fixed: calibrated LinearSVC, sigmoid calibration with 5 folds, hybrid features, `C=1.0`, balanced class weights, seed 42.

## Per-class Macro-F1 on wording-focused evaluations

| Intent | Frozen paraphrase base | Enriched | Frozen adversarial base | Enriched | Expanded external base | Enriched |
|---|---:|---:|---:|---:|---:|---:|
| balance_inquiry | 75.0% | 100.0% | 75.0% | 88.9% | 57.1% | 100.0% |
| card_issue | 75.0% | 100.0% | 75.0% | 75.0% | 66.7% | 75.0% |
| credit_card_application | 75.0% | 88.9% | 80.0% | 88.9% | 90.0% | 85.7% |
| forgot_pin | 0.0% | 100.0% | 57.1% | 88.9% | 33.3% | 84.2% |
| fraud_report | 35.7% | 83.3% | 55.6% | 62.5% | 34.0% | 62.1% |
| loan_inquiry | 80.0% | 90.9% | 90.9% | 90.9% | 72.7% | 90.9% |
| password_reset | 57.1% | 100.0% | 90.9% | 90.9% | 0.0% | 88.9% |
| transaction_query | 33.3% | 75.0% | 57.1% | 57.1% | 46.2% | 66.7% |

## Out-of-scope handling

OOS examples remained outside intent training. The existing confidence-plus-nearest-training-similarity reject mechanism was evaluated separately. Its 5th-percentile similarity threshold was recalculated from validation for the enriched model; confidence threshold remained at 0.60. No threshold was selected using the OOS or external-style evaluation sets.

| Model | Expanded OOS flagged | In-domain expanded queries incorrectly sent to review | Validation in-domain review rate | Frozen test in-domain review rate |
|---|---:|---:|---:|---:|
| Baseline | 6/10 (60%) | 36/80 (45%) | 7.2% | 7.0% |
| Enriched | 6/10 (60%) | 40/80 (50%) | 7.3% | 7.6% |

The current rejection mechanism still misses 4/10 new OOS examples and flags 40/80 in-domain external queries after enrichment. This is not adequate evidence for reliable OOS detection; threshold increases would create a substantial false-review burden. Treat OOS detection as a separate follow-up problem.

## Reproducibility, limits, and files

- Runner: `run_experiment.py` (re-running uses fixed seed 42 and writes only within this experiment folder).
- Training-only additions: `training_additions.csv` (145 rows).
- Frozen external-style evaluation: `expanded_external_eval.json` (80 in-domain + 10 OOS).
- Per-split and per-class metrics: `metrics.json`.
- Separate reject-rule results: `oos_rejection_diagnostic.json`.
- Candidate model, retained for review only: `enriched_model.pkl`.
- Baseline artifact snapshot used in this experiment: `baseline_model_snapshot.pkl`.
- Fingerprints, thresholds, and experiment recipe: `manifest.json`.

No source code, original raw data, canonical model artifact, existing train/validation/test data, or existing evaluation files were changed. The enriched model has not been promoted to the canonical application artifact.

Caveat: the maximum nearest-training TF-IDF similarity in the expanded set is 0.905. Although there are no exact duplicates and the set was created before scoring, this indicates at least one query is lexically close to training language. The small, hand-authored set should therefore be treated as a useful stress check, not an independent benchmark.


