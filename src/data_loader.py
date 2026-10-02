"""STAGE 1 - DATA COLLECTION: load, inspect, de-duplicate, enrich, split.

The raw CSV is never modified. Two intents requested by the assignment
(card_issue, forgot_pin) are NOT in the raw data, so they are generated from
documented templates (see CARD_ISSUE_CORES / FORGOT_PIN_CORES) and flagged with
source='synthetic' so that results can be reported separately.
"""
import json
import random
from difflib import SequenceMatcher

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import (BENCH_PATH, PROCESSED_DIR, RAW_PATH, SEED, ensure_dirs,
                    normalize_key, set_seed)

LABEL_FIX = {"credi_card_application": "credit_card_application"}

# ---------------------------------------------------------------------------
# Held-out behavioral suite (never used for training / validation / testing)
# ---------------------------------------------------------------------------
BENCHMARK = [
    ("My debit card is not working.", "card_issue"),
    ("My credit card stopped working this morning.", "card_issue"),
    ("The magnetic strip on my card is damaged.", "card_issue"),
    ("my atm card got stuck in the machine", "card_issue"),
    ("I forgot my ATM PIN.", "forgot_pin"),
    ("cant remember my debit card pin", "forgot_pin"),
    ("My PIN is blocked after three wrong attempts", "forgot_pin"),
    ("how do i reset my atm pin", "forgot_pin"),
    ("Someone charged $450 in Paris on my visa card!", "fraud_report"),
    ("I did not make this transaction, please block my account", "fraud_report"),
    ("my account was hacked, help!!", "fraud_report"),
    ("What are the interest rates for a personal loan?", "loan_inquiry"),
    ("how long does home loan approval take", "loan_inquiry"),
    ("can i get a loan to buy a car", "loan_inquiry"),
    ("Show me my last five transactions", "transaction_query"),
    ("what is this 250 dollar charge from amazon on my statement", "transaction_query"),
    ("How much money is left in my savings account?", "balance_inquiry"),
    ("what's my current balance", "balance_inquiry"),
    ("check my checking account balance pls", "balance_inquiry"),
    ("How do I apply for a credit card?", "credit_card_application"),
    ("i want a new credit card, what documents do i need", "credit_card_application"),
    ("status of my credit card application", "credit_card_application"),
    ("I forgot my online banking password.", "password_reset"),
    ("reset my netbanking password", "password_reset"),
    ("pls send me a new password link", "password_reset"),
    ("what's the weather like today", "out_of_scope"),
    ("who won the cricket match yesterday", "out_of_scope"),
    ("tell me a joke", "out_of_scope"),
    ("recommend a good pizza place nearby", "out_of_scope"),
]

CARD_TYPES = ["debit card", "credit card", "atm card", "card", "bank card"]
PIN_TYPES = ["atm pin", "debit card pin", "card pin", "pin", "atm pin number", "credit card pin"]
OPENERS = ["", "", "hi, ", "hey, ", "please help: ", "um, ", "hello, ", "urgent: "]
CLOSERS = ["", "", "", " pls help", " what should i do?", " any fix?", " asap", " :(", " please assist."]

CARD_ISSUE_CORES = [
    "my {x} stopped working", "{x} is not reading at the atm", "the chip on my {x} is damaged",
    "my {x} got stuck in the atm machine", "{x} declined at the shop even though my account is fine",
    "the magnetic strip on my {x} is scratched", "my {x} will not swipe at the store",
    "{x} keeps showing an error at the pos terminal", "my {x} got swallowed by the atm",
    "i cannot use my {x} at any terminal", "my {x} is bent and the machine rejects it",
    "contactless tap is not working on my {x}", "my new {x} is not activating",
    "the {x} you sent me does not work", "{x} not accepted online, keeps failing",
    "my {x} is physically broken, need a replacement", "need a replacement for my faulty {x}",
    "chip error every time i insert my {x}", "my {x} is dead, nothing works",
    "atm says invalid card when i insert my {x}", "my {x} gets rejected at every merchant",
    "{x} stopped responding at the machine", "the {x} cracked in my wallet and wont work",
    "why is my {x} malfunctioning", "my {x} wont work abroad", "card reader cannot read my {x}",
    "my {x} has stopped working since yesterday", "{x} issue: terminal keeps saying try again",
    "my {x} is defective", "tap and chip both fail on my {x}",
]
FORGOT_PIN_CORES = [
    "i forgot my {x}", "i cannot remember my {x}", "need to reset my {x}", "how do i change my {x}",
    "my {x} is blocked after wrong attempts", "i entered the wrong {x} three times and now it is locked",
    "{x} reset please", "lost my {x}, how to get a new one", "i do not remember the {x} for my card",
    "my {x} got locked", "how to generate a new {x}", "forgot the {x}, what now",
    "set a new {x} for my card", "unable to recall my {x}", "i dont recall the {x} i set",
    "can you help me recover my {x}", "pin mailer never came, need my {x} again",
    "my {x} is wrong every time i try", "want to create a fresh {x}", "{x} blocked, how can i unblock it",
    "i keep typing the wrong {x}", "what is the process to reset {x}", "my {x} slipped my mind",
    "my {x} attempts exceeded, card locked", "regenerate my {x} please", "cant remember {x}, need help with reset",
    "{x} change request", "i need to retrieve my {x}", "how can i reactivate my {x} after it got blocked",
    "i forgot the secret number for my card",
]


def _add_noise(s: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.20:
        s = s.replace("please", "pls").replace(" you ", " u ")
    elif r < 0.35:
        words = s.split()
        idx = [i for i, w in enumerate(words) if len(w) > 4]
        if idx:
            i = rng.choice(idx); w = words[i]; j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            s = " ".join(words)
    elif r < 0.45:
        s = s.upper()
    elif r < 0.55:
        s += "!!"
    return s


def generate_synthetic(intent: str, cores, slot_values, per_template: int, rng: random.Random):
    rows, seen = [], set()
    for tid, core in enumerate(cores):
        made, tries = 0, 0
        while made < per_template and tries < 200:
            tries += 1
            q = rng.choice(OPENERS) + core.format(x=rng.choice(slot_values)) + rng.choice(CLOSERS)
            q = _add_noise(q.strip(), rng)
            k = normalize_key(q)
            if k in seen:
                continue
            seen.add(k)
            rows.append({"query": q, "intent": intent, "source": "synthetic",
                         "template_id": f"{intent}_{tid:02d}"})
            made += 1
    return rows


def _near_benchmark(key: str, bench_keys, thr: float = 0.88) -> bool:
    for b in bench_keys:
        sm = SequenceMatcher(None, key, b)
        if sm.real_quick_ratio() >= thr and sm.quick_ratio() >= thr and sm.ratio() >= thr:
            return True
    return False


def build_datasets(verbose: bool = True) -> dict:
    ensure_dirs(); set_seed(SEED)
    rng = random.Random(SEED)

    raw = pd.read_csv(RAW_PATH)
    report = {"raw_rows": len(raw), "raw_nulls": int(raw.isna().sum().sum())}
    df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    df["intent"] = df["intent"].replace(LABEL_FIX)
    df["query"] = df["query"].astype(str).str.strip()

    df["_k"] = df["query"].map(normalize_key)
    report["duplicates_removed"] = int(df["_k"].duplicated().sum())
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    df["source"], df["template_id"] = "original", ""

    syn = pd.DataFrame(
        generate_synthetic("card_issue", CARD_ISSUE_CORES, CARD_TYPES, 9, rng)
        + generate_synthetic("forgot_pin", FORGOT_PIN_CORES, PIN_TYPES, 9, rng))
    full = pd.concat([df, syn], ignore_index=True)

    # Remove anything (exact or near-duplicate) that resembles a benchmark query
    bench_keys = [normalize_key(q) for q, _ in BENCHMARK]
    keys = full["query"].map(normalize_key)
    drop = keys.map(lambda k: _near_benchmark(k, bench_keys))
    report["benchmark_like_removed"] = int(drop.sum())
    full = full[~drop].reset_index(drop=True)
    # also drop cross-source duplicates
    full["_k"] = full["query"].map(normalize_key)
    full = full.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    full.to_csv(PROCESSED_DIR / "banking_queries_enriched.csv", index=False)

    # ---- split: originals stratified; synthetic split by template group -------
    orig = full[full.source == "original"]
    tr, tmp = train_test_split(orig, test_size=0.30, stratify=orig.intent, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.intent, random_state=SEED)
    parts = {"train": [tr], "val": [va], "test": [te]}
    for intent, g in full[full.source == "synthetic"].groupby("intent"):
        groups = sorted(g.template_id.unique()); rng.shuffle(groups)
        n = len(groups); a, b = round(0.70 * n), round(0.85 * n)
        for name, gs in (("train", groups[:a]), ("val", groups[a:b]), ("test", groups[b:])):
            parts[name].append(g[g.template_id.isin(gs)])
    out = {}
    for name, lst in parts.items():
        d = pd.concat(lst).sample(frac=1, random_state=SEED).reset_index(drop=True)
        d.to_csv(PROCESSED_DIR / f"{name}.csv", index=False); out[name] = d

    # ---- leakage assertions ---------------------------------------------------
    ks = {n: set(d["query"].map(normalize_key)) for n, d in out.items()}
    assert not (ks["train"] & ks["val"]) and not (ks["train"] & ks["test"]) and not (ks["val"] & ks["test"]), "split leakage"
    assert not (set(bench_keys) & set().union(*ks.values())), "benchmark leakage"

    BENCH_PATH.write_text(json.dumps([{"query": q, "expected": e} for q, e in BENCHMARK], indent=2))
    report["split_sizes"] = {n: len(d) for n, d in out.items()}
    report["intent_counts"] = full.intent.value_counts().to_dict()
    report["source_counts"] = full.source.value_counts().to_dict()
    if verbose:
        print("=== DATA CARD ==="); [print(f"{k}: {v}") for k, v in report.items()]
    return {"report": report, **out}


if __name__ == "__main__":
    build_datasets()
