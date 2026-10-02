"""Fast CLI:  python predict.py "My debit card is not working." """
import sys, warnings
warnings.filterwarnings("ignore")
try:
    from src.predictor import IntentPredictor
except ModuleNotFoundError as e:
    sys.exit(f"Missing package: {e.name}. Run:  pip install -r requirements.txt  then  python -m spacy download en_core_web_sm")


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python predict.py "your banking query"')
    q = " ".join(sys.argv[1:]); r = IntentPredictor().predict(q); a = r["analysis"]
    bar = "=" * 64
    top3 = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    print(f"{bar}\n{'BANKING INTENT CLASSIFICATION RESULT':^64}\n{bar}")
    print(f"Input Query        : \"{q}\"")
    print(f"Predicted Intent   : {r['display']}")
    print(f"Confidence         : {r['confidence'] * 100:.1f}%" + ("   [LOW - needs human review]" if r["needs_review"] else ""))
    if r["ambiguous"]:
        print(f"Ambiguity Warning  : close to {r['runner_up']} (margin {r['margin'] * 100:.0f}%) - ask a clarifying question")
    print(f"Top 3              : " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in top3))
    print(f"Action Verb        : {a['action']}")
    print(f"Target Entity      : {a['target']}")
    ents = ", ".join(f"{e['text']} ({e['label']})" for e in a["entities"]) or "-"
    print(f"Entities           : {ents}")
    print(f"Recommended Action : {r['action']}\n{bar}")


if __name__ == "__main__":
    main()
