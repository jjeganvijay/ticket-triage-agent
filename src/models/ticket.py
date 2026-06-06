"""Pydantic data models for the Ticket Triage Agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    BUG = "Bug"
    FEATURE_REQUEST = "Feature Request"
    BILLING = "Billing"
    OTHER = "Other"


class Priority(str, Enum):
    P1_CRITICAL = "P1 Critical"
    P2_HIGH = "P2 High"
    P3_MEDIUM = "P3 Medium"
    P4_LOW = "P4 Low"


class Ticket(BaseModel):
    """Raw support ticket loaded from a JSON file."""
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    description: str = Field(..., description="Full description of the issue")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional extra fields")


class TriageResult(BaseModel):
    """Result of triaging a single support ticket."""
    ticket_id: str
    description: str
    category: Category
    priority: Priority
    reasoning: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class AgentStep(BaseModel):
    """One Thought→Action→Observation cycle in the ReAct loop."""
    step: int
    thought: str
    action: str
    observation: str


class AgentRun(BaseModel):
    """Full agent run — contains the trace and final triage results."""
    run_id: str
    steps: List[AgentStep] = []
    results: List[TriageResult] = []
    total_tickets: int = 0
    processed: int = 0
    errors: int = 0
