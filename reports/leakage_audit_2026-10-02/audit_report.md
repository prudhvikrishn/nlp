# Banking query classifier: leakage and generalization audit

Audit date: 2026-10-02  
Project copy: `F:\nlp_project`  
Scope: read-only audit of data, model, and pipeline. The baseline model and existing processed splits were preserved. New evaluation files are stored in this report folder, outside `data/`.

## 1. Baseline and split construction

The recorded and freshly reproduced baseline is **95.75% Macro-F1** on **818** test queries (96.70% accuracy). Champion: calibrated LinearSVC with hybrid features; validation Macro-F1: 0.9728.

| Split | Total | Original | Synthetic |
|---|---:|---:|---:|
| Train | 3,856 | 3,479 | 377 |
| Validation | 836 | 746 | 90 |
| Test | 818 | 746 | 72 |

The loader removes 28 normalized raw duplicates from 5,000 records, applies the `credi_card_application` label correction, removes queries near the fixed 29-query behavioral suite, then splits original examples row-wise with stratification. Synthetic `card_issue` and `forgot_pin` records are generated from 30 templates per intent and assigned to splits by `template_id` group. There are no shared synthetic template IDs across train, validation, and test. The label correction for `credit_card_application` is on original rows; that class is not synthetic.

Feature builders and Word2Vec fit on train only. Model/feature candidates are compared on validation. Grid search uses CV within train and its result is checked on validation. LinearSVC calibration uses `CalibratedClassifierCV(method="sigmoid", cv=5)` fitted on training data. The held-out test set is scored after selection; calibration and model selection do not use test labels.

## 2. Exact duplicate audit

Normalization: Unicode NFKC, lowercase, trim, and collapse whitespace. A second check used the project's more aggressive punctuation-removing key.

| Pair of splits | Exact duplicate pairs |
|---|---:|
| Train–validation | 0 |
| Train–test | 0 |
| Validation–test | 0 |

The punctuation-insensitive project-key check also found zero cross-split duplicate pairs. There are no examples to list.

## 3. Near-duplicate audit

Similarity is measured from raw query text using three methods. The TF-IDF vocabularies were fit on train text only. Counts below show cross-split pairs and, after the slash, distinct test queries affected at each cutoff.

| Method | >= 0.90 | >= 0.95 | >= 0.98 |
|---|---:|---:|---:|
| Character TF-IDF cosine, char_wb 3–5 grams | 23 pairs / 18 test queries | 12 / 9 | 1 / 1 |
| Word TF-IDF cosine, 1–2 grams | 1 / 1 | 1 / 1 | 0 / 0 |
| Token-set Jaccard | 3 / 2 | 1 / 1 | 1 / 1 |

Across methods, the union is **26 high-similarity pairs affecting 20 test queries** (2.4% of the test set). All are original-to-original pairs, have the same label on both sides, and the test examples are correctly classified. Examples include train “Can you tell me the date and time of my last purchase?” vs test “Can you tell me the time and date of my last purchase?” (character cosine 1.000, token Jaccard 1.000), and train “How d I sign up 4 a loan?” vs test “how do i sign up 4 a loan?” (character cosine 0.944). The full 26-pair list with labels, sources, scores, and predictions is in [near_duplicate_pairs_ge_0.90.csv](near_duplicate_pairs_ge_0.90.csv).

As a sensitivity check, dropping all 18 test rows whose nearest train query has character cosine >= 0.90 leaves **800** queries and changes Macro-F1 from 95.75% to **95.73%** (accuracy 96.70% to 96.63%). The small score movement indicates these close original examples do not explain the headline score.

## 4. Synthetic-data leakage and source-wise metrics

| Test source | N | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original | 746 | 97.18% | 97.48% | 95.26% | 96.33% | 97.23% |
| Synthetic (`card_issue`, `forgot_pin`) | 72 | 91.67% | 100.00% | 91.67% | 95.45% | 95.45% |
| Combined baseline | 818 | 96.70% | 97.41% | 94.36% | 95.75% | 96.67% |

Macro metrics are calculated over the true labels represented in each subset. Synthetic test rows are not easier than original rows by accuracy or Macro-F1. No exact overlap or shared template family was found. For synthetic test queries, the highest character cosine to a synthetic train query is 0.710; the highest to original train is 0.463. No synthetic-to-train comparison reaches 0.90. Original examples have no retained source-family/row identifier beyond their query, so lineage-level clustering within the original data cannot be checked.

## 5. New paraphrase and adversarial evaluations

Two hand-authored evaluation sets were scored after training and were never used for fitting, tuning, or model selection. Each has five queries per supported intent plus five out-of-scope probes. There are no exact overlaps with train, validation, test, or the existing behavioral suite. Maximum nearest-train character cosine is 0.764 for paraphrases and 0.675 for adversarial queries; neither set has a query at or above 0.90.

| New set (40 in-domain queries) | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | In-domain flagged for review |
|---|---:|---:|---:|---:|---:|---:|
| Paraphrases | 52.50% | 75.22% | 52.50% | 53.90% | 53.90% | 18 / 40 |
| Adversarial wording | 72.50% | 85.64% | 72.50% | 72.71% | 72.71% | 7 / 40 |

The paraphrase set is especially weak for forgot PIN (0/5 correct), transaction query (1/5), and password reset (2/5). The adversarial set improves, but forgot PIN remains 2/5 and transaction query 2/5. Fraud report is 5/5 on both sets. These are exploratory results from only five authored examples per class, not a replacement test benchmark.

Out-of-scope rejection is poor on these new examples: **0/5 paraphrase probes** and **1/5 adversarial probes** were flagged for review. The existing behavioral suite remains 24/25 in-domain intents correct and 3/4 out-of-scope probes flagged. Thus the behavioral-suite result does not establish robust out-of-scope detection.

The two query lists and per-query outputs are kept separately as [paraphrase_eval.json](paraphrase_eval.json), [paraphrase_results.csv](paraphrase_results.csv), [adversarial_eval.json](adversarial_eval.json), and [adversarial_results.csv](adversarial_results.csv).

## 6. Confidence calibration

The loaded model is a five-fold `CalibratedClassifierCV` using sigmoid calibration. Its calibrators are fitted through CV on training data, not on the held-out test set.

On the test set, top-label accuracy is 96.70%, mean top-label confidence is 94.12%, and 27 predictions are incorrect. Correct predictions average 94.88% confidence; incorrect predictions average 71.86%. The 10-bin equal-count top-label ECE is **0.0258** (2.58 percentage points). Multiclass Brier score is **0.0564** using the sum of squared one-hot probability errors per example (lower is better; range 0–2); log loss is 0.1369. The reliability plot is [test_reliability.png](test_reliability.png), with exact bins in [test_reliability_bins.csv](test_reliability_bins.csv).

These in-distribution test metrics suggest reasonable aggregate calibration, with some under-confidence in lower-confidence bins. They do not justify reading an individual 99% score as 99% accuracy. In particular, the new out-of-scope queries were often assigned confident in-domain labels without review.

## 7. Diagnosis

**Category 1: No meaningful evidence of memorization.** No exact split duplicates were found; synthetic template families are isolated by split; high-similarity pairs are limited to 20 original test examples, all same-label and correct; and excluding the closest character matches leaves the test Macro-F1 essentially unchanged. Synthetic test performance is not higher than original-data performance. The audit therefore found no evidence that leakage materially inflated the 95.75% random-split result.

That conclusion is limited to the leakage question. The new paraphrase and adversarial evaluations show a substantial **generalization gap**: the baseline score represents performance on this dataset's random split, not robust performance on broad new customer wording. This gap is not evidence that the split leaked; it is evidence that the current test distribution does not fully represent those harder phrasings and OOS inputs.

## 8. Verified findings, inferences, limitations, and next step

**Verified findings:** split sizes and source counts above; zero exact duplicates; zero shared synthetic template IDs; 26 >=0.90 near pairs (all original, same-label, correct); disaggregated test metrics; calibration metrics; new-set predictions and review flags. The original baseline snapshot remains 95.75% Macro-F1 on 818 queries.

**Inference:** the random-split score is not materially driven by the identified close pairs. The poor paraphrase performance points to distribution sensitivity, especially around PIN, transaction, and password phrasing.

**Limitations:** no original-row/family IDs are retained; similarity thresholds are heuristic; the external-style sets have only five examples per class and were authored for this audit rather than independently sampled or adjudicated. Results on them are diagnostic, not definitive population estimates.

**Recommended next step:** keep the baseline unchanged. Expand the independently authored/collected paraphrase set substantially (with human-checked labels), and use it as a separate external-style evaluation. Address the PIN, transaction, password, and out-of-scope failures only after that set is frozen. Do not tune against these 90 audit queries if they are to remain an unbiased evaluation.
