# NLP-Based Banking Customer Query Classification and Intent Analysis System

## 1. Project Overview

This project classifies natural-language banking customer queries into supported banking intents. It preprocesses a query, builds lexical and Word2Vec features, predicts an intent, estimates confidence, and provides linguistic analysis and a recommended action.

The current source of truth is:

```text
F:\nlp_project
```

The canonical application model is the pipeline champion, a calibrated LinearSVC on `hybrid_nlp` features. The audit and the linguistic-generalization experiment were run against the previous `hybrid` baseline. Experimental models are stored separately and have not been promoted.

## 2. Project Goal

```text
Customer banking query
        |
        v
Text preprocessing and linguistic analysis
        |
        v
Feature representation
        |
        v
Intent classification
        |
        v
Intent, confidence, NLP analysis, and recommended action
```

The project aims to recognize banking intents across varied customer wording. The random-split test score is not, by itself, evidence of robust performance on all new wording.

## 3. Pipeline Stages and Actual Implementation

| Stage | Implementation | Outputs or related artifacts |
|---|---|---|
| Data collection and splitting | `src/data_loader.py` | Reads `data/raw/bank_customer_service_intent_classification_dataset.csv`; writes enriched data and `data/processed/train.csv`, `val.csv`, and `test.csv` |
| Exploratory data analysis | `src/eda.py` | EDA tables and figures under `artifacts/figures/` |
| Text preprocessing | `src/preprocessing.py` | Cleaning, contraction and shorthand handling, tokenization, stopword handling, lemmatization, and stemming |
| Vectorization | `src/feature_engineering.py` and `src/embeddings.py` | Fits and applies BoW and TF-IDF vectorizers on training-fitted vocabulary |
| Word embedding | `src/embeddings.py`, used by `src/feature_engineering.py` | Trains Word2Vec CBOW and Skip-Gram representations |
| NLP analysis | `src/nlp_analysis.py` | POS tags, spaCy dependency data when the spaCy model is available, regex banking entities, and action-to-target extraction |
| Feature engineering | `src/feature_engineering.py` | BoW, TF-IDF, Word2Vec, hybrid, and `hybrid_nlp` (hybrid + Stage 5 NLP-analysis features) representations |
| ML classification | `src/models.py` | Classifier definitions, compatibility rules, and tuning grids |
| Model comparison | `src/models.py`, `src/evaluation.py`, orchestrated by `src/pipeline.py` | Scores compatible classifier/feature pairs on validation; table and chart are under `artifacts/figures/` |
| Pipeline orchestration | `src/pipeline.py` | Fits candidates, selects the champion on validation, evaluates the held-out test set, and writes canonical artifacts |
| Evaluation | `src/evaluation.py`, called by `src/pipeline.py` | Metrics, per-class reports, confusion matrix, error analysis, and calibration figures under `artifacts/figures/` |
| Prediction and confidence handling | `src/predictor.py` | Loads the canonical artifact and returns an intent, confidence, review flag, and analysis |
| Command-line application | `predict.py` | Single-query prediction |
| Customer web application | `app/app.py` | Flask WSGI app for Vercel; BSD Bank customer form, intent prediction, confidence, and result |
| Local Streamlit customer demo | `app/customer_streamlit.py` | Local customer form and intent prediction |
| Admin application | `app/admin_app.py` | Local Streamlit query inbox; the Vercel Flask app serves a password-protected `/admin` inbox |
| Submission storage | `app/submissions.py` | PostgreSQL via `DATABASE_URL` on Vercel; local SQLite at `data/customer_queries.sqlite3` otherwise |

## 4. Project Structure

```text
F:\nlp_project
|
+-- app\
|   +-- app.py
|   +-- customer_streamlit.py
|   +-- admin_app.py
|   +-- submissions.py
|   +-- templates\
|       +-- base.html
|       +-- customer.html
|       +-- admin_login.html
|       +-- admin.html
+-- artifacts\
|   +-- figures\
|   +-- models\
+-- data\
|   +-- raw\
|   +-- processed\
|       +-- banking_queries_enriched.csv
|       +-- train.csv
|       +-- val.csv
|       +-- test.csv
|   +-- benchmarks\
+-- experiments\
|   +-- linguistic_generalization_2026-10-02\
+-- reports\
|   +-- leakage_audit_2026-10-02\
+-- src\
|   +-- data_loader.py
|   +-- eda.py
|   +-- embeddings.py
|   +-- evaluation.py
|   +-- feature_engineering.py
|   +-- models.py
|   +-- nlp_analysis.py
|   +-- pipeline.py
|   +-- predictor.py
|   +-- preprocessing.py
|   +-- utils.py
+-- tests\
+-- predict.py
+-- requirements.txt
+-- requirements-project.txt
+-- nltk_data\
|   +-- corpora\
|   +-- taggers\
|   +-- tokenizers\
+-- .python-version
+-- .vercelignore
+-- vercel.json
+-- README.md
+-- .venv\
```

The processed validation file is named `val.csv` in the current repository.

## 5. Main Components

### Data

The raw dataset is `data/raw/bank_customer_service_intent_classification_dataset.csv` (5,000 queries, 6 intents, columns `query` and `intent`; the `credi_card_application` label typo is corrected by the loader). Running `python -c "from src.data_loader import build_datasets; build_datasets()"` regenerates the processed splits from it: every row lands in the same split as the committed files. The only difference is that the current loader strips trailing whitespace from 7 queries, which the committed splits keep. The data loader creates the enriched dataset and split CSVs under `data/processed/`. The loader adds synthetic examples for `card_issue` and `forgot_pin`; those examples are marked with their source in the enriched data. Do not edit the original raw data or frozen evaluation sets.

### Preprocessing and Features

`src/preprocessing.py` normalizes text, handles contractions and project shorthand, tokenizes, removes stopwords while preserving intent-relevant words, and generates lemmas and stems.

`src/embeddings.py` implements Bag of Words, TF-IDF, and Word2Vec support. `src/feature_engineering.py` fits representations on training data and combines TF-IDF, Word2Vec, and linguistic features for the hybrid representation.

The `hybrid_nlp` representation connects the NLP-analysis stage to feature engineering. For every query, it turns the `BankingNLPAnalyzer` output into sparse indicator features, fitted on training data only and kept when seen at least twice:

- NER: entity labels such as `ent=AMOUNT`, `ent=CARD_TYPE`, `ent=MASKED_CARD`, and spaCy `DATE`/`MONEY`
- Semantic labeling: `target=debit card`, `action=working`, `action_negated`, and the frame `not working->debit card`
- Dependency parsing: relation types, the root word, and head-dependent pairs such as `dobj=forgot_pin`

The block is L2-normalized like the TF-IDF block and appended to the hybrid features. Without that normalization, the validation Macro-F1 of LinearSVC on `hybrid_nlp` was 95.88%, below plain hybrid.

### Linguistic Analysis

`src/nlp_analysis.py` uses spaCy when `en_core_web_sm` is available. Its fallback provides NLTK POS tagging and regex-based banking entities; dependency output is unavailable in fallback mode. The analyzer also extracts a banking action and target.

## 6. Current Canonical Model

The current champion is a calibrated LinearSVC using `hybrid_nlp` features (hybrid + NLP-analysis features):

```text
LinearSVC C=1.0
class_weight=balanced
random_state=42
CalibratedClassifierCV method=sigmoid, cv=5
```

The pipeline compares compatible feature/classifier pairs by validation Macro-F1 and uses its tie rule to select the champion. The test set is held out from this selection. The canonical artifact is `artifacts/models/best_model.pkl`.

The NLP features are fitted with spaCy `en_core_web_sm`. When spaCy is unavailable, as with the minimal Vercel `requirements.txt`, the analyzer falls back to NLTK, which has no dependency parse, and the feature builder warns about the backend mismatch. Measured on the test set, that fallback scores 96.20% Macro-F1 instead of 96.42%.

## 7. Verified Baseline Performance

Current champion (`hybrid_nlp` + LinearSVC), compared with the previous `hybrid` baseline:

| Evaluation | Previous baseline (`hybrid`) | Current (`hybrid_nlp`) |
|---|---:|---:|
| Validation Macro-F1 (used for selection) | 97.28% | 97.82% |
| Test accuracy (818 queries, held out) | 96.70% | 97.07% |
| Test Macro-F1 | 95.75% | 96.42% |
| Test Weighted-F1 | 96.67% | 97.05% |
| 5-fold CV Macro-F1 on train | 96.58% ± 0.75 | 96.50% ± 0.51 |
| Frozen paraphrase Macro-F1 (40) | 53.90% | 59.92% |
| Frozen adversarial Macro-F1 (40) | 72.71% | 70.33% |
| Expanded external-style Macro-F1 (80) | 50.00% | 51.12% |
| Behavioral suite in-domain / OOS flagged | 24/25, 3/4 | 24/25, 3/4 |

The validation and test gains are small (about 4–6 queries), and train CV shows no gain. The block normalization was chosen while looking at validation scores. Treat `hybrid_nlp` as a modest, not conclusive, improvement. On its own, the NLP-analysis block reaches 79.27% validation Macro-F1 with Logistic Regression (`artifacts/figures/extra_experiments.csv`), so it carries real intent signal but complements lexical features rather than replacing them.

The high in-sample training score (99.82% Macro-F1 for the previous baseline) alone does not establish overfitting. The main observed limitation is weaker performance on linguistically novel queries.

## 8. Leakage and Generalization Audit

The audit found no exact train/validation/test duplicates and no shared synthetic template families across splits. Removing highly similar test examples barely changed the baseline score. The full findings and limitations are in `reports/leakage_audit_2026-10-02/audit_report.md`.

The prior manually authored paraphrase and adversarial sets showed weaker performance than the random-split test set. Those small sets are diagnostic and should not be treated as population estimates.

## 9. Linguistic Generalization Experiment

The isolated experiment is under:

```text
experiments\linguistic_generalization_2026-10-02
```

It was run against the previous `hybrid` baseline. It adds 145 hand-authored training examples and retrains the same champion architecture without changing the original train, validation, test, or existing evaluation files.

| Evaluation | Baseline Macro-F1 | Enriched Macro-F1 |
|---|---:|---:|
| Validation | 97.28% | 97.57% |
| Frozen test | 95.75% | 96.68% |
| Frozen paraphrase | 53.90% | 92.27% |
| Frozen adversarial | 72.71% | 80.39% |
| New external-style set | 50.00% | 81.68% |

These results are exploratory. The additions and new evaluation set were authored in the same task after prior class-level failures and evaluation wording were visible. No existing evaluation rows were copied into training, but semantic influence cannot be ruled out. The new set is small and is not independently sampled. The enriched model has not been promoted to the canonical application artifact. See `experiments/linguistic_generalization_2026-10-02/experiment_report.md` for full metrics and limitations.

## 10. Out-of-Scope Detection

Out-of-scope detection remains a separate limitation. In the generalization experiment, the current confidence-plus-similarity review rule flagged 6 of 10 new OOS queries and incorrectly sent 40 of 80 in-domain queries to review. This does not establish reliable OOS detection. The experiment report contains the full diagnostic.

## 11. Setup and Commands

Activate the project environment in PowerShell:

```powershell
cd file path 
.\.venv\Scripts\Activate.ps1
```
Install declared dependencies if needed:

```powershell
python -m pip install -r requirements-project.txt
```

Run the unit tests:

```powershell
python -m unittest discover tests
```

Run a CLI prediction:

```powershell
python predict.py "My debit card is not working."
```

Run the BSD Bank customer page in one PowerShell window:

```powershell
cd F:\nlp_project
.\.venv\Scripts\Activate.ps1
streamlit run app\customer_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://localhost:8501`. Customers enter their name and query, submit it, and see the predicted category and confidence. With no PostgreSQL URL configured, submissions are stored locally in SQLite at `data/customer_queries.sqlite3`.

Run the admin inbox in a second PowerShell window:

```powershell
cd F:\nlp_project
.\.venv\Scripts\Activate.ps1
streamlit run app\admin_app.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://localhost:8502` to view submitted customer names, original queries, predicted categories, and confidence. Both pages bind to loopback on this computer.



## 12. Experiments, Reports, and Promotion

Experiments and audits are kept separate from the canonical application artifact:

```text
reports\leakage_audit_2026-10-02\
experiments\linguistic_generalization_2026-10-02\
```

The generalization experiment includes a runner, training additions, a separate external-style evaluation set, metrics, a candidate model, a baseline snapshot, and a dataset-fingerprint manifest. It refuses to continue if its frozen additions or evaluation file have changed.

Do not promote an experimental model based only on these exploratory results. First evaluate it on a genuinely independent, blinded set authored without access to the development examples, then review class-level errors and the false-review burden. Keep OOS detection as a separate measured capability.

## 13. Project Status

```text
Core NLP pipeline:             Implemented
Data preprocessing:            Implemented
Vectorization and embeddings:  Implemented
NLP analysis:                  Implemented (feeds feature engineering via hybrid_nlp)
Feature engineering:           Implemented
ML classification:             Implemented
Evaluation and leakage audit:  Completed
Linguistic enrichment:         Experiment completed; not promoted
Independent validation:        Pending
OOS robustness:                Needs improvement
```

The project is feature-complete for the core academic intent-classification objective. The next evidence needed is independent validation of the linguistic-generalization improvement.


