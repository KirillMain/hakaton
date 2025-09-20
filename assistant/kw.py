import argparse
import json
import re
from typing import Dict, List, Tuple, Optional

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher


def sim_ratio(a: str, b: str) -> int:
    a = a.strip().lower()
    b = b.strip().lower()
    if not a or not b:
        return 0
    if _HAS_RAPIDFUZZ:
        return int(fuzz.token_set_ratio(a, b))
    return int(SequenceMatcher(None, a, b).ratio() * 100)


DEFAULT_DICT = {
    "help": [
        "что такое",
        "инструкция",
        "как сделать",
        "как создать",
        "как редактировать",
        "справка",
        "faq",
        "правила",
        "как работает",
        "как участвовать",
    ],
    "action": [
        "создать",
        "оформить",
        "запустить",
        "подать",
        "открыть",
        "разместить",
        "редактировать",
        "изменить",
        "добавить",
        "закупить",
        "купить",
        "повторить закупку",
    ],
    "profile_update": [
        "изменить инн",
        "обновить инн",
        "поменять инн",
        "настроить уведомления",
        "включить уведомления",
        "отключить уведомления",
        "изменить профиль",
        "обновить профиль",
        "установить дату среза",
        "изменить дату среза",
    ],
    "action_types": {
        "quote_session": [
            "котировочная сессия",
            "мини аукцион",
            "мини-аукцион",
            "котировочную сессию",
            "создать котировочную сессию",
            "редактировать котировочную сессию",
        ],
        "direct_purchase": ["прямая закупка", "закупить напрямую", "купить напрямую"],
        "needs_procurement": [
            "закупка по потребностям",
            "потребность",
            "по потребностям",
        ],
    },
}


def normalize(text: str) -> str:
    t = (text or "").lower().strip()
    t = t.replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    return t


def best_match_score(text: str, phrases: List[str]) -> Tuple[int, Optional[str]]:
    best = 0
    best_p = None
    for p in phrases:
        s = sim_ratio(text, p)
        if p in text or text in p:
            s = max(s, 100 if p == text else 90)
        if s > best:
            best = s
            best_p = p
    return best, best_p


def classify_intent(
    query: str,
    dict: Dict,
    th_help: int = 60,
    th_action: int = 60,
    th_profile: int = 60,
    th_type: int = 60,
) -> Dict:
    t = normalize(query)
    s_help, p_help = best_match_score(t, dict.get("help", []))
    s_action, p_action = best_match_score(t, dict.get("action", []))
    s_profile, p_profile = best_match_score(t, dict.get("profile_update", []))
    atype_scores = {}
    atype_name = None
    atype_score = 0
    for name, phrases in dict.get("action_types", {}).items():
        sc, ph = best_match_score(t, phrases)
        atype_scores[name] = {"score": sc, "phrase": ph}
        if sc > atype_score:
            atype_score = sc
            atype_name = name
    if "?" in query:
        s_help = min(100, s_help + 8)
    if s_profile >= th_profile and s_profile >= s_action and s_profile >= s_help:
        intent = "profile_update"
        conf = max(0.55, s_profile / 100.0)
        action_type = None
    elif s_action >= th_action and s_action >= s_help:
        intent = "action"
        action_type = atype_name if atype_score >= th_type else None
        base = s_action / 100.0
        conf = base if action_type else max(0.5, base - 0.1)
    elif s_help >= th_help:
        intent = "help"
        conf = max(0.5, s_help / 100.0)
        action_type = None
    else:
        intent = "unknown"
        conf = 0.4
        action_type = None
    return {
        "intent": intent,
        "action_type": action_type,
        "confidence": round(conf, 2),
        "matches": {
            "help": {"score": s_help, "phrase": p_help},
            "action": {"score": s_action, "phrase": p_action},
            "profile_update": {"score": s_profile, "phrase": p_profile},
            "action_type": {
                "best": atype_name,
                "score": atype_score,
                "per_type": atype_scores,
            },
        },
    }
