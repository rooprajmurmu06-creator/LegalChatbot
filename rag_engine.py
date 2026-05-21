"""
rag_engine.py — Provides the LegalKGWrapper singleton.
"""

import logging
from typing import Optional

from legal_kg_engine import LegalKGWrapper, KGResult

logger = logging.getLogger(__name__)

_engine: Optional[LegalKGWrapper] = None

def get_rag_engine() -> LegalKGWrapper:
    global _engine
    if _engine is None:
        _engine = LegalKGWrapper()
    return _engine

# Re-export for compatibility
RAGResult = KGResult