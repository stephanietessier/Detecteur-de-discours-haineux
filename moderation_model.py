from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, REVIEW_THRESHOLD
from .utils import normalize_text


@dataclass
class ModerationResult:
    message: str
    label: str
    confidence: float
    needs_human_review: bool
    probabilities: Dict[str, float]
    explanation_terms: List[str]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)
    return model, report, matrix, list(model.classes_)


def explain_prediction(model: Pipeline, message: str, label: str, top_n: int = 6) -> List[str]:
    """Retourne les termes TF-IDF les plus contributifs pour la classe prédite."""
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    X = vectorizer.transform([message])
    feature_names = np.array(vectorizer.get_feature_names_out())
    class_index = list(clf.classes_).index(label)
    coefs = clf.coef_[class_index]
    contributions = X.toarray()[0] * coefs
    if np.all(contributions == 0):
        return []
    top_indices = np.argsort(contributions)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if contributions[i] > 0]


def predict_message(model: Pipeline, message: str) -> ModerationResult:
    probabilities_array = model.predict_proba([message])[0]
    classes = list(model.classes_)
    best_index = int(np.argmax(probabilities_array))
    label = classes[best_index]
    confidence = float(probabilities_array[best_index])
    probabilities = {cls: float(prob) for cls, prob in zip(classes, probabilities_array)}
    explanation_terms = explain_prediction(model, message, label)
    return ModerationResult(
        message=message,
        label=label,
        confidence=confidence,
        needs_human_review=confidence < REVIEW_THRESHOLD,
        probabilities=probabilities,
        explanation_terms=explanation_terms,
    )
