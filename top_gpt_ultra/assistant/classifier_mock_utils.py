import re
import json
from typing import Dict, Optional

HELP_TRIGGERS = [
    r"\bчто такое\b",
    r"\bкак (?:создать|оформить|провести|работать|настроить|подписаться|найти|загрузить|редактировать)\b",
    r"\bкак\b",
    r"\bгде\b",
    r"\bзачем\b",
    r"\bпочему\b",
    r"\bинструкция\b",
    r"\bпомощ\b",
    r"\bстатья\b",
    r"\bсправк\b",
    r"\bfaq\b",
    r"\bправила\b",
]

ACTION_VERBS = [
    r"\bсоздать\b",
    r"\bоформить\b",
    r"\bпровести\b",
    r"\bзапустить\b",
    r"\bоткрыть\b",
    r"\bподать\b",
    r"\bразместить\b",
    r"\bопубликовать\b",
    r"\bредактировать\b",
    r"\bизменить\b",
    r"\bдобавить\b",
    r"\bприкрепить\b",
    r"\bзакупить\b",
    r"\bкупить\b",
    r"\bповторить\b",
]

PROFILE_FIELDS = [
    r"\bинн\b",
    r"\bпрофил\b",
    r"\bуведомлени",
    r"\bнастройк",
    r"\bдата среза\b",
    r"\bсреза\b",
    r"\bэлектронн(?:ая|ую)? подпись\b",
    r"\bэцп\b",
]
PROFILE_VERBS = [
    r"\bобнови(?:ть)?\b",
    r"\bизмени(?:ть)?\b",
    r"\bустанови(?:ть)?\b",
    r"\bвключи(?:ть)?\b",
    r"\bотключи(?:ть)?\b",
    r"\bнастрои(?:ть)?\b",
    r"\bпоменя(?:ть)?\b",
]

ACTION_TYPES = {
    "quote_session": [r"\bкотировочн", r"\bмини[- ]?аукцион", r"\bсессия\b"],
    "direct_purchase": [
        r"\bпрям(?:ая|ую)? закупк",
        r"\bзакупить напрямую\b",
        r"\bкупить у\b",
    ],
    "needs_procurement": [r"\bзакупк[ау] по потребност", r"\bпотребност"],
}

QUESTION_MARK_BONUS = True


def normalize(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("ё", "е")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def score_any(patterns, text) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def detect_action_type(text: str) -> Optional[str]:
    for atype, pats in ACTION_TYPES.items():
        if score_any(pats, text) > 0:
            return atype
    return None


def classify_intent(text: str) -> Dict:
    """
    Возвращает:
    {
      "intent": "help|action|profile_update|unknown",
      "action_type": "quote_session|direct_purchase|needs_procurement|None",
      "confidence": 0.0..1.0,
      "signals": { ... }
    }
    """
    t = normalize(text)

    s_help_q = 1 if (QUESTION_MARK_BONUS and "?" in text) else 0
    s_help_kw = score_any(HELP_TRIGGERS, t)
    s_action_kw = score_any(ACTION_VERBS, t)
    s_profile_kw = score_any(PROFILE_FIELDS, t)
    s_profile_verbs = score_any(PROFILE_VERBS, t)

    s_help = s_help_kw + s_help_q
    s_profile = min(3, s_profile_kw + s_profile_verbs)
    s_action = s_action_kw

    if s_profile >= 2 and s_action == 0:
        intent = "profile_update"
        conf = min(1.0, 0.55 + 0.15 * s_profile)
        action_type = None
    else:
        if s_action > 0 and s_action >= s_help:
            intent = "action"
            action_type = detect_action_type(t)
            base = 0.55 + 0.1 * s_action
            conf = base if action_type else max(0.5, base - 0.1)
        elif s_help > 0:
            intent = "help"
            action_type = None
            conf = min(0.9, 0.5 + 0.1 * s_help)
        else:
            # эвристика
            if re.match(r"^(что|как|где|зачем|почему)\b", t):
                intent, action_type, conf = "help", None, 0.55
            else:
                intent, action_type, conf = "unknown", None, 0.4

    return {
        "intent": intent,
        "action_type": action_type,
        "confidence": round(conf, 2),
        "signals": {
            "help_kw": s_help_kw,
            "help_qmark": s_help_q,
            "action_kw": s_action_kw,
            "profile_kw": s_profile_kw,
            "profile_verbs": s_profile_verbs,
        },
    }
