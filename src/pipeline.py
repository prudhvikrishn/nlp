"""STAGES 6-9 ORCHESTRATOR: fit features on train, benchmark every valid (feature, model)
pair on VALIDATION, pick the champion by Macro-F1, then score the held-out TEST set ONCE."""
import argparse
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from . import evaluation as ev
from .data_loader import build_datasets
from .embeddings import similarity_report
from .feature_engineering import FEATURE_KINDS, FeatureBuilder
from .models import MODEL_NAMES, TUNING_GRIDS, is_compatible, make_model
from .preprocessing import BankingTextPreprocessor
from .utils import BENCH_PATH, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, SEED, ensure_dirs, set_seed

warnings.filterwarnings("ignore")
TIE_TOL = 0.005       # models within 0.5 pt of the best Macro-F1 are "tied" -> prefer the faster one


def _timed_predict(model, X):
    t = time.perf_counter(); p = model.predict_proba(X); return p, (time.perf_counter() - t) / X.shape[0] * 1000


def run(skip_extras: bool = False):
    ensure_dirs(); set_seed(SEED)
    # ---- Stage 1-3: data + preprocessing ----------------------------------------------
    if not (PROCESSED_DIR / "train.csv").exists():
        build_datasets()
    tr, va, te = (pd.read_csv(PROCESSED_DIR / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test"))
    pre = BankingTextPreprocessor()
    P = {n: pre.process_many(d["query"]) for n, d in (("train", tr), ("val", va), ("test", te))}
    le = LabelEncoder().fit(tr["intent"]); classes = list(le.classes_)
    y = {n: le.transform(d["intent"]) for n, d in (("train", tr), ("val", va), ("test", te))}

    # ---- Stage 4-6: features (fit on TRAIN only) ------------------------------------------
    fb = FeatureBuilder().fit(P["train"])
    X = {k: {n: fb.transform(P[n], k) for n in ("train", "val")} for k in FEATURE_KINDS}

    # ---- Stage 7-8: benchmark all valid pairs on VALIDATION ---------------------------------
    rows, fitted = [], {}
    for k in FEATURE_KINDS:
        for m in MODEL_NAMES:
            if not is_compatible(m, k):
                rows.append({"features": k, "model": m, "status": "N/A (negative values)"}); continue
            Xt, Xv = X[k]["train"], X[k]["val"]
            mdl = make_model(m, Xt.shape[1], sparse.issparse(Xt))
            t = time.perf_counter(); mdl.fit(Xt, y["train"]); fit_s = time.perf_counter() - t
            proba, lat = _timed_predict(mdl, Xv); pred = proba.argmax(1)
            r = ev.metrics(y["val"], pred); r.update(features=k, model=m, status="ok", train_time_s=round(fit_s, 2),
                                                     latency_ms_per_query=round(lat, 4))
            rows.append(r); fitted[(k, m)] = mdl
            print(f"  {k:13s} {m:19s} val macro-F1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}  fit={fit_s:.1f}s")
    comp = pd.DataFrame(rows)
    ok = comp[comp.status == "ok"].copy().sort_values("macro_f1", ascending=False)
    best = ok.macro_f1.iloc[0]
    tied = ok[ok.macro_f1 >= best - TIE_TOL].sort_values(["latency_ms_per_query", "train_time_s"])
    champ = tied.iloc[0]; ck, cm = champ.features, champ.model
    print(f"\nChampion (val macro-F1 within {TIE_TOL} of best, fastest): {cm} on {ck}")

    # ---- Add-on: 5-fold CV stability for top-3 + small tuning of the champion ---------------
    cv_rows = []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for _, r in ok.head(3).iterrows():
        Xt = X[r.features]["train"]
        base = make_model(r.model, Xt.shape[1], sparse.issparse(Xt))
        s = cross_val_score(base, Xt, y["train"], cv=skf, scoring="f1_macro", n_jobs=1)
        cv_rows.append({"features": r.features, "model": r.model, "cv_macro_f1_mean": s.mean().round(4), "cv_macro_f1_std": s.std().round(4)})
    pd.DataFrame(cv_rows).to_csv(FIGURES_DIR / "cv_stability_top3.csv", index=False)

    champ_model = fitted[(ck, cm)]
    if cm in TUNING_GRIDS:
        Xt, Xv = X[ck]["train"], X[ck]["val"]
        gs = GridSearchCV(make_model(cm, Xt.shape[1], sparse.issparse(Xt)), TUNING_GRIDS[cm], scoring="f1_macro", cv=3).fit(Xt, y["train"])
        tuned_f1 = f1_score(y["val"], gs.predict(Xv), average="macro")
        print(f"Tuning {cm}: best params {gs.best_params_}, val macro-F1 {tuned_f1:.4f} (untuned {champ.macro_f1:.4f})")
        if tuned_f1 > champ.macro_f1:
            champ_model = gs.best_estimator_

    # ---- choose reject threshold on VALIDATION --------------------------------------------
    pv = champ_model.predict_proba(X[ck]["val"]); correct = pv.argmax(1) == y["val"]
    thr = float(np.clip(np.percentile(pv.max(1)[correct], 5), 0.35, 0.60))
    sim_val = fb.nearest_similarity(P["val"])
    sim_thr = float(np.percentile(sim_val, 5))          # flag the 5% least-familiar validation queries

    # ---- Stage 9: TEST SET, scored ONCE for the champion --------------------------------------
    Xte = fb.transform(P["test"], ck); pt = champ_model.predict_proba(Xte); pred_te = pt.argmax(1)
    y_true_lbl, y_pred_lbl = le.inverse_transform(y["test"]), le.inverse_transform(pred_te)
    overall = ev.metrics(y_true_lbl, y_pred_lbl)
    print("\n=== TEST (held-out, scored once) ===", {k: round(v, 4) for k, v in overall.items()})
    ev.per_class_report(y_true_lbl, y_pred_lbl, classes).round(4).to_csv(FIGURES_DIR / "per_class_report_test.csv")
    dis = ev.disaggregated(te, y_pred_lbl); dis.round(4).to_csv(FIGURES_DIR / "disaggregated_test_metrics.csv"); print(dis.round(4))
    ev.plot_confusion(y_true_lbl, y_pred_lbl, classes)
    err = ev.error_analysis(te, y_pred_lbl, pt, classes); err.to_csv(FIGURES_DIR / "error_analysis.csv", index=False)
    ev.confusion_pairs(err).to_csv(FIGURES_DIR / "confusion_pairs.csv", index=False)
    ev.plot_confidence(y_true_lbl, y_pred_lbl, pt, classes)

    # ---- comparison table + chart ------------------------------------------------------------
    comp.round(4).to_csv(FIGURES_DIR / "model_comparison.csv", index=False)
    piv = ok.pivot(index="model", columns="features", values="macro_f1")
    ax = piv.plot(kind="bar", figsize=(11, 5), width=0.8); ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Model comparison: classifier x feature set (validation)"); ax.set_ylim(max(0, piv.min().min() - 0.05), 1.0)
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150); plt.close()

    # ---- Word2Vec analysis ----------------------------------------------------------------------
    pairs = [("stolen", "fraud"), ("pin", "password"), ("loan", "balance"), ("card", "debit"), ("dough", "money"), ("charge", "transaction")]
    sims = []
    for nm, wv in (("CBOW", fb.wv_cbow), ("SkipGram", fb.wv_sg)):
        for r in similarity_report(wv, pairs):
            r["model"] = nm; sims.append(r)
    pd.DataFrame(sims).to_csv(FIGURES_DIR / "word2vec_similarity.csv", index=False)
    fb.w2v_cbow_model.save(str(MODELS_DIR / "w2v_cbow.model")); fb.w2v_sg_model.save(str(MODELS_DIR / "w2v_skipgram.model"))
    if not skip_extras:
        _extras(P, y, fb, X)

    # ---- save artifact ------------------------------------------------------------------------------
    fb.drop_training_state()
    joblib.dump({"preprocessor": pre, "feature_builder": fb, "model": champ_model, "label_encoder": le,
                 "feature_kind": ck, "model_name": cm, "reject_threshold": thr, "similarity_threshold": sim_thr}, MODELS_DIR / "best_model.pkl")
    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")
    summary = {"champion": {"model": cm, "features": ck}, "reject_threshold": round(thr, 3), "similarity_threshold": round(sim_thr, 3),
               "validation_macro_f1": round(float(champ.macro_f1), 4), "test": {k: round(v, 4) for k, v in overall.items()}}
    (MODELS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- behavioral suite (post-hoc) -------------------------------------------------------------------
    from .predictor import IntentPredictor
    behavioral_report()
    return summary


def _extras(P, y, fb, X):
    """Add-on experiments: stemming vs lemmatization, mean vs idf-weighted pooling, hybrid ablation."""
    from sklearn.linear_model import LogisticRegression
    from .embeddings import doc_matrix, make_tfidf
    rows = []
    for name, key in (("lemmatized", "lemmas"), ("stemmed", "stems"), ("tokens only", "tokens")):
        v = make_tfidf().fit([" ".join(p[key]) for p in P["train"]])
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=10).fit(v.transform([" ".join(p[key]) for p in P["train"]]), y["train"])
        rows.append({"experiment": f"TF-IDF+LR text normalization: {name}", "val_macro_f1": round(f1_score(y["val"], lr.predict(v.transform([" ".join(p[key]) for p in P["val"]])), average="macro"), 4)})
    for nm, wv in (("CBOW", fb.wv_cbow), ("SkipGram", fb.wv_sg)):
        for pool, w in (("mean", None), ("idf-weighted", fb.idf)):
            Xt = doc_matrix([p["lemmas"] for p in P["train"]], wv, w); Xv = doc_matrix([p["lemmas"] for p in P["val"]], wv, w)
            lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=10).fit(Xt, y["train"])
            rows.append({"experiment": f"Word2Vec {nm} pooling: {pool} (+LR)", "val_macro_f1": round(f1_score(y["val"], lr.predict(Xv), average="macro"), 4)})
    pd.DataFrame(rows).to_csv(FIGURES_DIR / "extra_experiments.csv", index=False)


def behavioral_report():
    """Held-out qualitative suite. In-domain: intent must be right. Out-of-scope: must be flagged."""
    from .predictor import IntentPredictor
    pr = IntentPredictor(); bench = json.loads(BENCH_PATH.read_text()); out = []
    for b in bench:
        r = pr.predict(b["query"]); oos = b["expected"] == "out_of_scope"
        second = sorted(r["probabilities"].values())[-2]
        ok = r["needs_review"] if oos else r["intent"] == b["expected"]
        out.append({"query": b["query"], "expected": b["expected"], "predicted": r["intent"], "confidence": round(r["confidence"], 3),
                    "margin": round(r["confidence"] - second, 3), "similarity": round(r["similarity"], 3),
                    "flagged_for_review": r["needs_review"], "pass": ok})
    df = pd.DataFrame(out); df.to_csv(FIGURES_DIR / "behavioral_suite_results.csv", index=False)
    ind, oos_df = df[df.expected != "out_of_scope"], df[df.expected == "out_of_scope"]
    print(f"\n=== BEHAVIORAL SUITE (held-out) ===\n in-domain intent correct : {int(ind['pass'].sum())}/{len(ind)}"
          f"\n out-of-scope flagged     : {int(oos_df['pass'].sum())}/{len(oos_df)}"
          f"\n false alarms (in-domain flagged): {int(ind.flagged_for_review.sum())}/{len(ind)}")
    print(df.to_string(index=False)); return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-extras", action="store_true"); a = ap.parse_args()
    run(skip_extras=a.skip_extras)
