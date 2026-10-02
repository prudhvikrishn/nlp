"""STAGE 2 - DATA ANALYSIS (EDA): class balance, lengths, vocabulary, n-grams, embedding map."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import BankingTextPreprocessor
from .utils import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, ensure_dirs


def run():
    ensure_dirs(); sns.set_theme(style="whitegrid")
    df = pd.read_csv(PROCESSED_DIR / "banking_queries_enriched.csv", keep_default_na=False)
    pre = BankingTextPreprocessor()
    df["cleaned"] = df["query"].map(pre.clean)
    df["n_chars"], df["n_words"] = df["query"].str.len(), df["cleaned"].str.split().str.len()

    # class balance
    cnt = df.intent.value_counts(); pct = (cnt / len(df) * 100).round(1)
    pd.DataFrame({"count": cnt, "percent": pct}).to_csv(FIGURES_DIR / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(x=cnt.values, y=cnt.index, hue=cnt.index, palette="viridis", legend=False, ax=ax)
    for i, (c, p) in enumerate(zip(cnt.values, pct.values)):
        ax.text(c + 8, i, f"{c} ({p}%)", va="center")
    ax.set_title("Intent distribution (enriched dataset)"); ax.set_xlabel("queries"); ax.set_xlim(0, cnt.max() * 1.18)
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "intent_distribution.png", dpi=150); plt.close(fig)

    # lengths
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for it in df.intent.unique():
        sns.kdeplot(df[df.intent == it].n_words, ax=ax[0], label=it, bw_adjust=1.5, fill=False)
    ax[0].set_title("Query length (words) by intent"); ax[0].legend(fontsize=7)
    sns.boxplot(data=df, y="intent", x="n_chars", hue="intent", legend=False, ax=ax[1], palette="Set2")
    ax[1].set_title("Query length (characters) by intent")
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "query_length_distribution.png", dpi=150); plt.close(fig)
    df.groupby("intent")[["n_chars", "n_words"]].agg(["mean", "median", "max"]).round(2).to_csv(FIGURES_DIR / "length_stats.csv")

    # vocabulary
    toks = [t for s in df.cleaned for t in s.split()]
    summary = {"queries": len(df), "tokens": len(toks), "vocabulary": len(set(toks)),
               "type_token_ratio": round(len(set(toks)) / len(toks), 4),
               "imbalance_ratio_max_to_min": round(cnt.max() / cnt.min(), 2),
               "mean_words": round(df.n_words.mean(), 2)}
    (FIGURES_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # n-grams
    rows, fig, axes = [], *plt.subplots(2, 4, figsize=(20, 9))
    for ax, it in zip(axes.ravel(), sorted(df.intent.unique())):
        for n, name in ((1, "unigram"), (2, "bigram")):
            cv = CountVectorizer(ngram_range=(n, n), stop_words="english").fit(df[df.intent == it].cleaned)
            freq = np.asarray(cv.transform(df[df.intent == it].cleaned).sum(0)).ravel()
            top = sorted(zip(cv.get_feature_names_out(), freq), key=lambda x: -x[1])[:15]
            rows += [{"intent": it, "type": name, "ngram": g, "count": int(c)} for g, c in top]
        t = [r for r in rows if r["intent"] == it][:10]
        ax.barh([r["ngram"] for r in t][::-1], [r["count"] for r in t][::-1], color="#2a9d8f"); ax.set_title(it, fontsize=10)
    plt.tight_layout(); fig.savefig(FIGURES_DIR / "top_ngrams_per_intent.png", dpi=130); plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGURES_DIR / "top_ngrams_per_intent.csv", index=False)

    # Word2Vec map (needs the pipeline to have run once)
    try:
        from gensim.models import Word2Vec
        from sklearn.decomposition import PCA
        wv = Word2Vec.load(str(MODELS_DIR / "w2v_skipgram.model")).wv
        words = [w for w in ["card", "debit", "credit", "pin", "atm", "password", "reset", "forgot", "loan", "interest", "apply",
                             "balance", "savings", "account", "fraud", "stolen", "unauthorized", "charge", "transaction", "payment",
                             "statement", "block", "freeze", "hacked", "scam"] if w in wv.key_to_index]
        xy = PCA(2, random_state=42).fit_transform(np.vstack([wv[w] for w in words]))
        fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(xy[:, 0], xy[:, 1], c="#e76f51")
        for (x, y), w in zip(xy, words): ax.annotate(w, (x, y), fontsize=9)
        ax.set_title("Word2Vec (Skip-Gram) banking keywords - PCA projection"); plt.tight_layout()
        fig.savefig(FIGURES_DIR / "word2vec_pca.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("Skipped Word2Vec plot (run pipeline first):", e)
    print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__":
    run()
