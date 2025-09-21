import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from assistant.nlp.nlp_utils import correct_and_detect
from assistant.arts import get_serialized_arts_by_text


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


def build_vectorizers(max_features_w=120000, max_features_c=80000):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features_w,
        token_pattern=r"(?u)\b\w[\w\-]+\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features_c,
        dtype=np.float32,
    )
    return word, char


def _transform(word_vec: TfidfVectorizer, char_vec: TfidfVectorizer, texts: pd.Series):
    Xw = word_vec.transform(texts.map(normalize_text).fillna(""))
    Xc = char_vec.transform(texts.map(normalize_text).fillna(""))
    from scipy.sparse import hstack

    return hstack([Xw, Xc], format="csr")


def fit_intent_clf(
    df: pd.DataFrame, word_vec: TfidfVectorizer, char_vec: TfidfVectorizer
):
    X_text = df["query"].map(normalize_text).fillna("")
    Xw = word_vec.fit_transform(X_text)
    Xc = char_vec.fit_transform(X_text)
    from scipy.sparse import hstack

    X = hstack([Xw, Xc], format="csr")
    y = df["intent"].values
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    return clf


def fit_action_type_clf(
    df_action: pd.DataFrame, word_vec: TfidfVectorizer, char_vec: TfidfVectorizer
):
    X_text = df_action["query"].map(normalize_text).fillna("")
    Xw = word_vec.fit_transform(X_text)
    Xc = char_vec.fit_transform(X_text)
    from scipy.sparse import hstack

    X = hstack([Xw, Xc], format="csr")
    y = df_action["action_type"].fillna("OTHER").replace("", "OTHER").values
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    return clf


def eval_clf(
    df: pd.DataFrame,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    clf,
    label_column: str,
) -> Dict[str, Any]:
    X = _transform(word_vec, char_vec, df["query"])
    y_true = df[label_column].values
    y_pred = clf.predict(X)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_txt = classification_report(
        y_true, y_pred, zero_division=0, output_dict=False
    )
    cm = confusion_matrix(y_true, y_pred, labels=sorted(pd.unique(df[label_column])))
    return {
        "macro_f1": float(macro_f1),
        "report": report_txt,
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm.tolist(),
    }


def save_bundle(out_dir: str, bundle: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    for name, obj in bundle.items():
        joblib.dump(obj, os.path.join(out_dir, f"{name}.joblib"))


def load_bundle(model_dir: str) -> Dict[str, Any]:
    need = [
        "intent_word_vec",
        "intent_char_vec",
        "intent_clf",
        "atype_word_vec",
        "atype_char_vec",
        "atype_clf",
    ]
    out = {}
    for n in need:
        path = os.path.join(model_dir, f"{n}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        out[n] = joblib.load(path)
    return out


@dataclass
class Entities:
    purchase_id: Optional[str] = None
    inn: Optional[str] = None
    as_of_date: Optional[str] = None
    qty: Optional[str] = None
    budget: Optional[str] = None


PURCHASE_ID_RE = re.compile(r"\b\d{5,}\b")
INN_RE = re.compile(r"\b\d{10}(\d{2})?\b")
DATE_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
QTY_RE = re.compile(r"\b(\d{1,5})\s*шт\b", re.IGNORECASE)
BUDGET_RE = re.compile(r"\b(\d[\d\s]{3,})\s*руб\b", re.IGNORECASE)


def extract_entities(text: str) -> Entities:
    t = normalize_text(text)
    pid = PURCHASE_ID_RE.search(t)
    inn = INN_RE.search(t)
    date = DATE_RE.search(t)
    qty = QTY_RE.search(t)
    budget = BUDGET_RE.search(t)
    return Entities(
        purchase_id=pid.group(0) if pid else None,
        inn=inn.group(0) if inn else None,
        as_of_date=date.group(0) if date else None,
        qty=qty.group(1) if qty else None,
        budget=re.sub(r"\s+", "", budget.group(1)) if budget else None,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def proba_for(clf: LogisticRegression, X) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return p, clf.classes_
    df = clf.decision_function(X)
    if df.ndim == 1:
        df = np.vstack([-df, df]).T
    p = softmax(df)
    return p, clf.classes_


def predict_intent(
    text: str, word_vec, char_vec, clf, thresh: float = 0.55
) -> Tuple[str, float]:
    X = _transform(word_vec, char_vec, pd.Series([text]))
    p, classes = proba_for(clf, X)
    idx = int(p[0].argmax())
    label = str(classes[idx])
    conf = float(p[0, idx])
    if conf < thresh:
        return "UNKNOWN", conf
    return label, conf


def predict_action_type(
    text: str, word_vec, char_vec, clf, thresh: float = 0.60
) -> Tuple[str, float]:
    X = _transform(word_vec, char_vec, pd.Series([text]))
    p, classes = proba_for(clf, X)
    idx = int(p[0].argmax())
    label = str(classes[idx])
    conf = float(p[0, idx])
    if conf < thresh:
        return "OTHER", conf
    return label, conf


def quick_rules_override(text: str) -> Optional[str]:
    t = normalize_text(text)
    if "?" in text or re.search(
        r"\b(что такое|как|где|зачем|почему|инструкция|справк|faq)\b", t
    ):
        return "HELP"
    if (
        re.fullmatch(r"\d{5,}", t)
        or INN_RE.search(t)
        or re.search(r"\b(найти|показать|посмотреть)\b", t)
    ):
        return "VIEW"
    if re.search(
        r"\b(создать|оформить|запустить|подать|открыть|разместить|редактировать|изменить|повторить)\b",
        t,
    ):
        return "ACTION"
    return None


def process_query(
    text: str, model_dir: str, profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    bundle = load_bundle(model_dir)
    user_text = text
    text = correct_and_detect(text)["corrected"]
    entities = extract_entities(text)

    rule = quick_rules_override(text)
    if rule:
        intent = rule
        conf = 0.66
    else:
        intent, conf = predict_intent(
            text,
            bundle["intent_word_vec"],
            bundle["intent_char_vec"],
            bundle["intent_clf"],
        )

    action_type, action_conf = None, None
    if intent == "ACTION":
        atype, aconf = predict_action_type(
            text,
            bundle["atype_word_vec"],
            bundle["atype_char_vec"],
            bundle["atype_clf"],
        )
        action_type, action_conf = atype, aconf

    inn = None
    as_of = None
    if profile:
        inn = profile.get("inn") or None
        as_of = profile.get("as_of_date") or None

    result = {
        "input": user_text,
        "intent": intent,
        "intent_confidence": round(conf or 0.0, 4),
        "entities": {
            "purchase_id": entities.purchase_id,
            "inn": entities.inn,
            "as_of_date": entities.as_of_date,
            "qty": entities.qty,
            "budget": entities.budget,
        },
        "action_type": action_type,
        "action_type_confidence": (
            round(action_conf or 0.0, 4) if action_conf is not None else None
        ),
        "routing": None,
    }

    if intent == "HELP":
        result["routing"] = {
            "type": "HELP",
            "search": normalize_text(text),
            "data": get_serialized_arts_by_text(normalize_text(text)),
        }
    elif intent == "VIEW":
        result["routing"] = {
            "type": "VIEW",
            "purchase_id": entities.purchase_id,
            "inn": entities.inn or inn,
            "name_query": None if entities.purchase_id else normalize_text(text),
            "as_of_date": entities.as_of_date or as_of,
        }
    elif intent == "ACTION":
        result["routing"] = {
            "type": "ACTION",
            "action_type": action_type or "OTHER",
            "target_purchase_id": entities.purchase_id,
            "prefill": {
                "inn": inn,
                "qty": entities.qty,
                "budget": entities.budget,
            },
        }
    else:
        result["routing"] = {"type": "UNKNOWN", "hint": "Уточните запрос"}

    return result
