"""Broker request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BrokerRequest(BaseModel):
    """Request payload published to the broker."""

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    image_url: str | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] | None = None


class BrokerResponse(BaseModel):
    """Response payload published by the agent."""

    request_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    response: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
