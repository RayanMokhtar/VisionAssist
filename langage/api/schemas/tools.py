"""
Schémas pour le suivi des appels d'outils (tool calling).

ToolCallRecord  – enregistrement d'un appel d'outil
ToolResult      – résultat d'un appel d'outil
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ToolCallRecord(BaseModel):
    """Enregistrement d'un appel d'outil effectué par le LLM."""
    tool_name: str
    args: dict[str, Any] = {}
    result: Optional[str] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error_msg: Optional[str] = None


class ToolResult(BaseModel):
    """Résultat renvoyé par un outil au LLM."""
    content: str
    success: bool = True
