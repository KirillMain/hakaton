from django.conf import settings
from typing import Dict, Any, Optional

from .intent_pipeline import process_query, load_bundle

MODEL_DIR = str(settings.INTENT_MODEL_DIR)

_bundle_cache = None


def analyze_query(
    text: str, profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    global _bundle_cache
    if _bundle_cache is None:
        _bundle_cache = load_bundle(MODEL_DIR)
    return process_query(text, MODEL_DIR, profile=profile)
