"""STAGE 7 - ML CLASSIFICATION: five benchmark classifiers + feature compatibility matrix."""
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .utils import SEED

MODEL_NAMES = ["NaiveBayes", "LogisticRegression", "LinearSVC", "RandomForest", "GradientBoosting"]
NB_OK = {"bow", "tfidf"}          # MultinomialNB needs non-negative features


def is_compatible(model: str, feature: str) -> bool:
    return not (model == "NaiveBayes" and feature not in NB_OK)


def make_model(name: str, n_features: int, is_sparse: bool):
    if name == "NaiveBayes":
        return MultinomialNB(alpha=0.3)
    if name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, C=10, random_state=SEED)
    if name == "LinearSVC":     # Platt scaling -> real probabilities
        return CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), method="sigmoid", cv=5)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)
    if name == "GradientBoosting":
        hgb = HistGradientBoostingClassifier(random_state=SEED)
        if is_sparse:           # HGB rejects sparse input -> SVD to <=300 dense dims first
            return Pipeline([("svd", TruncatedSVD(min(300, n_features - 1), random_state=SEED)), ("clf", hgb)])
        return hgb
    raise ValueError(name)


TUNING_GRIDS = {
    "NaiveBayes": {"alpha": [0.05, 0.1, 0.3, 1.0]},
    "LogisticRegression": {"C": [1, 5, 10, 30]},
    "LinearSVC": {"estimator__C": [0.1, 0.3, 1, 3]},
}
