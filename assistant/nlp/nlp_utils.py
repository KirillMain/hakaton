from __future__ import annotations
import os
import re
from collections import Counter
from threading import RLock
from typing import Dict, Optional, List

from symspellpy import SymSpell, Verbosity

try:
    from rapidfuzz import fuzz

    _HAVE_RF = True
except Exception:
    import difflib

    _HAVE_RF = False


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\sё-]", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s


def _score(a: str, b: str) -> int:
    if _HAVE_RF:
        return fuzz.WRatio(a, b)
    # fallback
    import difflib

    return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)


DOMAIN_WORDS: Dict[str, int] = {
    "котировочная": 120,
    "котировочную": 120,
    "сессия": 120,
    "сессию": 120,
    "закупка": 80,
    "закупить": 80,
    "напрямую": 80,
    "услуги": 80,
    "связи": 80,
    "повтор": 50,
    "потребностям": 50,
    "поставщик": 40,
    "контракт": 40,
    "аукцион": 40,
    "реестр": 25,
    "единый": 25,
    "лот": 8,
    "лоты": 8,
}


def _expand_abbrev_quote_session(text: str) -> str:
    # "кот сессию" -> "котировочную сессию"
    text = re.sub(r"\bкот\b(?=\s+сес+\w*\b)", "котировочную", text, flags=re.I)
    # "кот сессия" -> "котировочная сессия"
    text = re.sub(r"\bкот\b(?=\s+сессия\b)", "котировочная", text, flags=re.I)
    # "котсессию" (слитно)
    text = re.sub(r"\bкот(?=сес+\w*\b)", "котировочн", text, flags=re.I)
    return text


_KS_CORE = r"k\.?\s*[-/\\]?\s*s\.?"
_KS_CORE_CYR = r"к\.?\s*[-/\\]?\s*с\.?"
_KS_ANY = rf"(?:{_KS_CORE}|{_KS_CORE_CYR})"

_PREP_PREP = r"(?:о|об|про|по|к|для|до|после|перед|из|из-за)"
_PREP_ACC = r"(?:в|на)"

_VERBS_ACC = (
    r"(?:открыт\w*|создат\w*|объяв\w*|провест\w*|начат\w*|запуст\w*|опубликов\w*)"
)


def _expand_abbrev_ks(text: str) -> str:
    text = re.sub(
        rf"\b({_PREP_PREP})\s+{_KS_ANY}\b",
        r"\1 котировочной сессии",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"\b({_PREP_ACC})\s+{_KS_ANY}\b",
        r"\1 котировочную сессию",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"\b({_VERBS_ACC})\s+{_KS_ANY}\b",
        r"\1 котировочную сессию",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"\b{_KS_ANY}\b",
        "котировочная сессия",
        text,
        flags=re.I,
    )
    return text


def _build_symspell(
    csv_path: Optional[str] = None, extra_words: Optional[Dict[str, int]] = None
) -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    freq = Counter()

    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for raw in f:
                for w in _normalize(raw).split():
                    if len(w) > 1:
                        freq[w] += 1

    boost = extra_words or DOMAIN_WORDS
    for w, c in boost.items():
        for token in _normalize(w).split():
            if len(token) > 1:
                freq[token] += c

    for qw in _QWORDS:
        freq[qw] += 200
    freq["такое"] += 400

    for w, c in freq.items():
        sym.create_dictionary_entry(w, c)

    return sym


_QWORDS = {
    "что",
    "как",
    "где",
    "когда",
    "зачем",
    "почему",
    "сколько",
    "какой",
    "какая",
    "какие",
    "каков",
    "какова",
    "каковы",
    "куда",
    "откуда",
    "чей",
    "чья",
    "чьё",
    "чьи",
    "чем",
    "это",
    "такое",
}

_QPHRASES = {("что", "такое"), ("что", "это")}

_STOPWORDS = {"по", "на", "в", "о", "об", "при", "из", "для", "к"} | _QWORDS

_ACTION_VERBS = {
    "создать",
    "создай",
    "создайте",
    "оформить",
    "оформи",
    "оформите",
    "запустить",
    "запусти",
    "запустите",
    "подать",
    "подай",
    "подайте",
    "открыть",
    "открой",
    "откройте",
    "разместить",
    "размести",
    "разместите",
    "редактировать",
    "редактируй",
    "редактируйте",
    "изменить",
    "измени",
    "измените",
    "повторить",
    "повтори",
    "повторите",
    "скопировать",
    "скопируй",
    "скопируйте",
    "сделать",
    "сделай",
    "сделайте",
}

_DOMAIN_STABLE = {
    "котировочная",
    "котировочную",
    "котировочн",
    "сессия",
    "сессию",
    "закупка",
    "закупку",
    "закупки",
    "закупок",
    "ноутбук",
    "ноутбуки",
    "шт",
}


def _symspell_fix(sym: SymSpell, text: str) -> str:
    text_norm = _normalize(text)
    compound = sym.lookup_compound(text_norm, max_edit_distance=2)
    text_fixed = compound[0].term if compound else text_norm

    out: List[str] = []
    toks = text_fixed.split()
    i = 0
    while i < len(toks):
        w = toks[i]
        wl = w.lower()

        if i + 1 < len(toks):
            pair = (wl, toks[i + 1].lower())
            if pair in _QPHRASES:
                out.append(toks[i])
                out.append(toks[i + 1])
                i += 2
                continue

        if (
            wl in _STOPWORDS
            or wl in _ACTION_VERBS
            or wl in _DOMAIN_STABLE
            or wl.isdigit()
            or wl == "шт"
            or len(w) <= 2
        ):
            out.append(w)
            i += 1
            continue

        sugg = sym.lookup(wl, Verbosity.TOP, max_edit_distance=2, transfer_casing=False)
        if sugg:
            best = sugg[0]
            if best.distance <= 1 and (best.term not in _QWORDS or wl in _QWORDS):
                out.append(best.term)
            else:
                out.append(w)
        else:
            out.append(w)
        i += 1

    return " ".join(out)


def _choose_adj_for_session(session_word: str) -> str:
    return "котировочную" if session_word.lower().endswith("ю") else "котировочная"


def _fix_kotirovochnaya_fuzzy(text: str) -> str:
    toks = text.split()
    if not toks:
        return text

    sess_pos = [
        i for i, t in enumerate(toks) if re.match(r"^сес+и?\w*$", t, flags=re.I)
    ]
    if not sess_pos:
        return text

    targets = ["котировочную", "котировочная", "котировочн"]
    THRESH = 72

    def best_score(w: str) -> int:
        try:
            return max(_score(w, t) for t in targets)
        except ValueError:
            return 0

    for i in sess_pos:
        for j in (i - 1, i - 2):
            if j < 0:
                continue
            w = toks[j].lower()
            if w.startswith("котировоч"):
                continue
            if len(w) >= 5 and best_score(w) >= THRESH:
                toks[j] = _choose_adj_for_session(toks[i])
                return " ".join(toks)

        m = re.match(r"^([кК][а-яё]{3,})(сес+и?\w*)$", toks[i], flags=re.I)
        if m:
            left, sess = m.group(1), m.group(2)
            if best_score(left.lower()) >= THRESH:
                toks[i] = f"{_choose_adj_for_session(sess)} {sess}"
                return " ".join(toks)

    return " ".join(toks)


_INTENT_PATTERNS = {
    "quote_session": [
        re.compile(r"\bкотир\w*\b.*\bсес+\w*\b", re.I),
        re.compile(r"\bкот\b[^\S\r\n]*\bсес+\w*\b", re.I),
        re.compile(rf"\b{_KS_ANY}\b", re.I),
    ],
    "direct_purchase": [
        re.compile(r"\b(прям(ая|о)|напрямую)\b.*\bзакуп\w*\b", re.I),
    ],
    "repeat_purchase": [
        re.compile(r"\bповтор\w*\b.*\bзакуп\w*\b", re.I),
    ],
    "needs_based_procurement": [
        re.compile(r"\bпо\s+потребност(ям|и)\b", re.I),
    ],
}
_INTENT_CANON = {
    "quote_session": "котировочная сессия",
    "direct_purchase": "прямая закупка",
    "repeat_purchase": "повтор закупки",
    "needs_based_procurement": "закупка по потребностям",
}


def _detect_intent(text: str) -> str:
    for intent, pats in _INTENT_PATTERNS.items():
        if any(p.search(text) for p in pats):
            return intent
    return "unknown"


_lock = RLock()
_engine_symspell: Optional[SymSpell] = None


def init_engine(
    csv_path: Optional[str] = None, extra_words: Optional[Dict[str, int]] = None
) -> None:
    global _engine_symspell
    with _lock:
        _engine_symspell = _build_symspell(csv_path=csv_path, extra_words=extra_words)


def _get_engine() -> SymSpell:
    global _engine_symspell
    if _engine_symspell is None:
        with _lock:
            if _engine_symspell is None:
                _engine_symspell = _build_symspell()
    return _engine_symspell


def preprocess_text(text: str) -> str:
    sym = _get_engine()
    t = _expand_abbrev_quote_session(text)
    t = _expand_abbrev_ks(t)
    t = _symspell_fix(sym, t)
    t = _fix_kotirovochnaya_fuzzy(t)
    t = _expand_abbrev_quote_session(t)
    t = _expand_abbrev_ks(t)
    return _normalize(t)


def correct_and_detect(text: str) -> Dict[str, str]:
    corrected = preprocess_text(text)
    intent = _detect_intent(corrected)
    canonical = _INTENT_CANON.get(intent, corrected)
    return {
        "input": text,
        "corrected": corrected,
        "intent": intent,
        "canonical": canonical,
    }
