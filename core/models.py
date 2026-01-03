from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

class AdvinationResultType(Enum):
    FOUND = "FOUND"
    PARTIAL_FOUND = "PARTIAL"
    NO_MATCH = "NO_MATCH"

class OrchestrationOutcome(Enum):
    SUGGEST_EXACT = "SUGGEST_EXACT"
    SUGGEST_ADAPTED = "SUGGEST_ADAPTED"
    START_DIALOG = "START_DIALOG"
    DEFER = "DEFER"

@dataclass
class CommandSuggestion:
    text: str
    source: str
    match_score: float
    metadata: Dict[str, Any] = None

@dataclass
class AdvinationResult:
    result_type: AdvinationResultType
    suggestions: Optional[List[CommandSuggestion]] = None
    confidence: float = 1.0
    raw_prefix: str = ""
    raw_context: Dict[str, Any] = None

@dataclass
class OrchestrationResult:
    outcome: OrchestrationOutcome
    suggestions: Optional[List[CommandSuggestion]] = None
    dialog_id: Optional[str] = None
    first_question: Optional[str] = None
    task_id: Optional[str] = None
    reason: Optional[str] = None
    retry_after: Optional[datetime] = None