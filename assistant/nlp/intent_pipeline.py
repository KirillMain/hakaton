import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from assistant.arts import get_serialized_arts_by_text

from assistant.nlp.nlp_utils import preprocess_text
from history.utils import find_any_history_model


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


def get_text_series(df: pd.DataFrame) -> pd.Series:
    if "query" in df.columns:
        return df["query"].astype(str)
    if "text" in df.columns:
        return df["text"].astype(str)
    raise ValueError("CSV must contain either 'query' or 'text' column")


def ensure_intent_column(df: pd.DataFrame) -> pd.DataFrame:
    if "intent" not in df.columns:
        raise ValueError("CSV must contain 'intent' column")
    df["intent"] = df["intent"].fillna("").astype(str).str.strip().str.upper()
    return df


def ensure_action_type_column(df: pd.DataFrame) -> pd.DataFrame:
    if "action_type" not in df.columns:
        df["action_type"] = "OTHER"
    df["action_type"] = (
        df["action_type"]
        .fillna("OTHER")
        .astype(str)
        .str.strip()
        .replace("", "OTHER")
        .str.upper()
    )
    return df


def build_vectorizers(max_features_w=120_000, max_features_c=80_000):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features_w,
        token_pattern=r"(?u)\b\w[\w\-]+\b",
        dtype=np.float32,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features_c,
        dtype=np.float32,
        sublinear_tf=True,
    )
    return word, char


def _transform(word_vec: TfidfVectorizer, char_vec: TfidfVectorizer, texts: pd.Series):
    Xw = word_vec.transform(texts.map(normalize_text).fillna(""))
    Xc = char_vec.transform(texts.map(normalize_text).fillna(""))
    from scipy.sparse import hstack

    return hstack([Xw, Xc], format="csr")


def fit_intent_clf(
    texts: pd.Series,
    labels: pd.Series,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
):
    Xw = word_vec.fit_transform(texts.map(normalize_text).fillna(""))
    Xc = char_vec.fit_transform(texts.map(normalize_text).fillna(""))
    from scipy.sparse import hstack

    X = hstack([Xw, Xc], format="csr")
    y = labels.values
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(X, y)
    return clf


def fit_action_type_clf(
    texts: pd.Series,
    labels: pd.Series,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
):
    Xw = word_vec.fit_transform(texts.map(normalize_text).fillna(""))
    Xc = char_vec.fit_transform(texts.map(normalize_text).fillna(""))
    from scipy.sparse import hstack

    X = hstack([Xw, Xc], format="csr")
    y = labels.values
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(X, y)
    return clf


def eval_clf(
    texts: pd.Series,
    labels: pd.Series,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    clf,
) -> Dict[str, Any]:
    X = _transform(word_vec, char_vec, texts)
    y_true = labels.values
    y_pred = clf.predict(X)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_txt = classification_report(
        y_true, y_pred, zero_division=0, output_dict=False
    )
    cm = confusion_matrix(y_true, y_pred, labels=sorted(pd.unique(labels)))
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
    names_required = ["intent_word_vec", "intent_char_vec", "intent_clf"]
    names_optional = ["atype_word_vec", "atype_char_vec", "atype_clf"]
    out = {}
    for n in names_required:
        path = os.path.join(model_dir, f"{n}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required model file: {path}")
        out[n] = joblib.load(path)
    for n in names_optional:
        path = os.path.join(model_dir, f"{n}.joblib")
        out[n] = joblib.load(path) if os.path.exists(path) else None
    return out


@dataclass
class Entities:
    purchase_id: Optional[str] = None
    inn: Optional[str] = None
    as_of_date: Optional[str] = None
    qty: Optional[str] = None
    budget: Optional[str] = None
    name: Optional[str] = None


PURCHASE_ID_RE = re.compile(r"\b\d{5,}\b")
INN_RE = re.compile(r"\b\d{10}(\d{2})?\b")
DATE_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
QTY_RE = re.compile(r"\b(\d{1,5})\s*шт\.?\b", re.IGNORECASE)
BUDGET_RE = re.compile(r"\b(\d[\d\s]{3,})\s*руб\b", re.IGNORECASE)

UNITS_RX = r"(?:шт\.?|штук|ед\.?|уп\.?|упак\.?|комплект(?:а|ов)?|позици(?:я|и))"
EXPL_NAME_RE = re.compile(
    r"(?:наименовани[ея]|предмет\s+закупки|товар)\s*[:\-–]\s*([^\n,.;]+)",
    re.IGNORECASE,
)

QUOTED_NAME_RE = re.compile(r"[\"«“„]([^\"»”]+)[\"»”]")
AFTER_NA_CTX_RE = re.compile(
    rf"(?:закупк\w+|сесс\w+)\s+(?:на|по)\s+([^\n,.;:]+?)(?=\s*\d+\s*{UNITS_RX}\b|[,.;:!?]|\bна\s+сумм|\bза\b|\bпо\b|$)",
    re.IGNORECASE,
)
AFTER_NA_GENERIC_RE = re.compile(
    rf"\bна\s+([^\n,.;:]+?)(?=\s*\d+\s*{UNITS_RX}\b|[,.;:!?]|\bна\s+сумм|\bза\b|\bпо\b|$)",
    re.IGNORECASE,
)
TAIL_JUNK_RE = re.compile(
    rf"\s*(?:\d+\s*{UNITS_RX}\b|на\s+сумм[уы]|за\s+\d+|по\s+\d+)\s*$",
    re.IGNORECASE,
)


def _clean_name_fragment(s: str) -> Optional[str]:
    if not s:
        return None
    s = TAIL_JUNK_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) < 3:
        return None
    if normalize_text(s) in {
        "закупка",
        "котировочная сессия",
        "сессия",
        "прямая закупка",
    }:
        return None
    return s


def extract_name(text: str) -> Optional[str]:
    raw = text or ""
    m = EXPL_NAME_RE.search(raw)
    if m:
        cand = _clean_name_fragment(m.group(1))
        if cand:
            return cand

    quoted = QUOTED_NAME_RE.findall(raw)
    if quoted:
        quoted = sorted((q.strip() for q in quoted), key=len, reverse=True)
        for q in quoted:
            cand = _clean_name_fragment(q)
            if cand:
                return cand

    m = AFTER_NA_CTX_RE.search(raw)
    if m:
        cand = _clean_name_fragment(m.group(1))
        if cand:
            return cand

    m = AFTER_NA_GENERIC_RE.search(raw)
    if m:
        cand = _clean_name_fragment(m.group(1))
        if cand:
            return cand

    return None


def extract_entities(text: str) -> Entities:
    t = normalize_text(text)
    pid = PURCHASE_ID_RE.search(t)
    inn = INN_RE.search(t)
    date = DATE_RE.search(t)
    qty = QTY_RE.search(t)
    budget = BUDGET_RE.search(t)
    name = extract_name(text)

    return Entities(
        purchase_id=pid.group(0) if pid else None,
        inn=inn.group(0) if inn else None,
        as_of_date=date.group(0) if date else None,
        qty=qty.group(1) if qty else None,
        budget=re.sub(r"\s+", "", budget.group(1)) if budget else None,
        name=name,
    )


def extract_tags(text: str) -> List[str]:
    t = normalize_text(text)
    tags = []
    if "?" in text:
        tags.append("QUESTION")
    if re.search(r"\b(что такое|как|где|зачем|почему|инструкция|справк|faq)\b", t):
        tags.append("HELP_KEYWORD")
    if re.search(r"\d{5,}", t):
        tags.append("NUMBER")
    if INN_RE.search(t):
        tags.append("INN")
    if re.search(r"\b(найти|показать|посмотреть)\b", t):
        tags.append("VIEW_VERB")
    if re.search(
        r"\b(создать|создай|оформи|сделай|открой|запусти|размести|измени|повтори|скопируй)\b",
        t,
    ):
        tags.append("ACTION_VERB")
    if re.search(r"\bпрям(ая|ую)?\s+закуп", t):
        tags.append("DIRECT_PURCHASE")
    if "потребност" in t or "по потребност" in t:
        tags.append("NEEDS_PROCUREMENT")
    if "котировоч" in t or re.search(r"\bкс\b", t):
        tags.append("QUOTE_SESSION")

    print(tags)

    return sorted(set(tags))


def build_examples(text: str, tags: List[str], ent: Entities) -> List[str]:
    """Строит примеры запросов на основе тегов и найденных сущностей."""
    ex: List[str] = []

    if "DIRECT_PURCHASE" in tags:
        ex += ["Сделать прямую закупку", "Оформить прямую закупку"]
    if "NEEDS_PROCUREMENT" in tags:
        ex += ["Создать закупку по потребностям", "Оформить закупку по потребностям"]
    if "QUOTE_SESSION" in tags:
        ex += [
            "Создать котировочную сессию",
            "Скопировать котировочную сессию",
            "Разместить котировочную сессию",
        ]
    if "ACTION_VERB" in tags and not any(
        t in tags for t in ["DIRECT_PURCHASE", "NEEDS_PROCUREMENT", "QUOTE_SESSION"]
    ):
        ex += [
            "Создать котировочную сессию",
            "Создать прямую закупку",
            "Создать закупку по потребностям",
        ]

    if "QUESTION" in tags or "HELP_KEYWORD" in tags:
        ex += ["Как создать прямую закупку?", "Как объединить профили организации?"]

    if "VIEW_VERB" in tags or ent.purchase_id or ent.inn:
        if ent.purchase_id:
            ex.append(f"Показать закупку {ent.purchase_id}")
        if ent.inn:
            ex.append(f"Показать закупки по ИНН {ent.inn}")
        if not (ent.purchase_id or ent.inn):
            ex.append("Показать мои закупки")

    if "NUMBER" in tags and not ent.purchase_id:
        ex.append("Показать закупку 123456")

    if not ex:
        ex = [
            "Создать котировочную сессию",
            "Сделать прямую закупку",
            "Как загрузить МЧД в профиль пользователя?",
        ]

    seen = set()
    out = []
    for q in ex:
        if q not in seen:
            out.append(q)
            seen.add(q)
        if len(out) >= 6:
            break
    return out


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
    action_verbs = (
        r"создать|создай|создайте|"
        r"оформить|оформи|оформите|"
        r"запустить|запусти|запустите|"
        r"подать|подай|подайте|"
        r"открыть|открой|откройте|"
        r"разместить|размести|разместите|"
        r"редактировать|редактируй|редактируйте|"
        r"изменить|измени|измените|"
        r"повторить|повтори|повторите|"
        r"скопировать|скопируй|скопируйте|"
        r"сделать|сделай|сделайте"
    )
    if re.search(rf"\b({action_verbs})\b", t):
        return "ACTION"
    if (
        re.search(r"\bпрям(ая|ую)?\s+закуп", t)
        or "по потребност" in t
        or "котировоч" in t
        or re.search(r"\bкс\b", t)
    ):
        return "ACTION"
    return None


def action_type_hint(text: str) -> Optional[str]:
    t = normalize_text(text)
    if re.search(r"\bпрям(ая|ую)?\s+закуп", t):
        return "DIRECT_PURCHASE"
    if "потребност" in t or "по потребност" in t:
        return "NEEDS_PROCUREMENT"
    if "котировоч" in t or re.search(r"\bкс\b", t):
        return "QUOTE_SESSION"
    if re.search(r"\bповтор(ить|и|ите)?\b", t) or "ещё раз" in t or "сделай тоже" in t:
        return "REPEAT"
    return None


def process_query(
    text: str, model_dir: str, profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    user_text = text
    bundle = load_bundle(model_dir)
    entities = extract_entities(text)

    rule = quick_rules_override(text)
    hint = action_type_hint(text)

    if rule:
        intent = rule
        conf = 0.66
    elif hint:
        intent = "ACTION"
        conf = 0.66
    else:
        intent, conf = predict_intent(
            text,
            bundle.get("intent_word_vec"),
            bundle.get("intent_char_vec"),
            bundle.get("intent_clf"),
        )

    action_type, action_conf = None, None
    if intent == "ACTION":
        at_w, at_c, at_clf = (
            bundle.get("atype_word_vec"),
            bundle.get("atype_char_vec"),
            bundle.get("atype_clf"),
        )
        if at_w is not None and at_c is not None and at_clf is not None:
            atype, aconf = predict_action_type(text, at_w, at_c, at_clf)
            action_type, action_conf = atype, aconf

        if action_type is None or (action_conf is not None and action_conf < 0.60):
            if hint:
                action_type = hint

        if not action_type:
            action_type = "OTHER"

    inn = None
    as_of = None
    if profile:
        inn = profile.get("inn") or None
        as_of = profile.get("as_of_date") or None

    name_query = entities.name or (
        None if entities.purchase_id else normalize_text(text)
    )

    result: Dict[str, Any] = {
        "input": user_text,
        "intent": intent,
        "intent_confidence": round(conf or 0.0, 4),
        "entities": {
            "purchase_id": entities.purchase_id,
            "inn": entities.inn,
            "as_of_date": entities.as_of_date,
            "qty": entities.qty,
            "budget": entities.budget,
            "name": entities.name,
        },
        "action_type": action_type if intent == "ACTION" else None,
        "action_type_confidence": (
            round(action_conf or 0.0, 4)
            if (intent == "ACTION" and action_conf is not None)
            else None
        ),
        "routing": None,
    }

    if intent == "HELP":
        query = normalize_text(text)
        result["routing"] = {
            "type": "HELP",
            "search": query,
            "data": get_serialized_arts_by_text(query),
        }

    elif intent == "VIEW":
        history_data = find_any_history_model(
            entities.purchase_id,
            entities.inn or inn,
            name_query or normalize_text(text),
        )
        result["routing"] = {
            "type": "VIEW",
            "purchase_id": entities.purchase_id,
            "inn": entities.inn or inn,
            "name_query": name_query,
            "as_of_date": entities.as_of_date or as_of,
            "data": history_data,
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
                "name": preprocess_text(entities.name),
            },
        }

        if (action_type or "OTHER") == "OTHER":
            tags = extract_tags(text)
            result["tags"] = build_examples(text, tags, entities)

    else:
        # UNKNOWN
        tags = extract_tags(text)
        result["routing"] = {"type": "UNKNOWN", "hint": "Уточните запрос"}
        result["tags"] = build_examples(text, tags, entities)

    return result
