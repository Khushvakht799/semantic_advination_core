from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid


class AdvinationResultType(Enum):
    """Типы результатов адивинации"""
    FOUND = "FOUND"          # Точное совпадение
    PARTIAL_FOUND = "PARTIAL" # Похожие варианты
    NO_MATCH = "NO_MATCH"    # Ничего не найдено


class OrchestrationOutcome(Enum):
    """Исходы оркестрации"""
    SUGGEST_EXACT = "SUGGEST_EXACT"      # Предложить точные совпадения
    SUGGEST_ADAPTED = "SUGGEST_ADAPTED"  # Предложить адаптированные
    START_DIALOG = "START_DIALOG"        # Начать диалог композиции
    DEFER = "DEFER"                      # Отложить задачу


class CommandStatus(Enum):
    """Статус команды в БД"""
    CONFIRMED = "confirmed"
    GENERATED = "generated"
    DRAFT = "draft"


@dataclass
class CommandSuggestion:
    """Вариант команды для предложения"""
    text: str
    source: str          # "exact_match", "partial_match", "adapted"
    match_score: float   # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "match_score": self.match_score,
            "metadata": self.metadata
        }


@dataclass
class AdvinationResult:
    """Результат чистой адивинации"""
    result_type: AdvinationResultType
    suggestions: Optional[List[CommandSuggestion]] = None
    confidence: float = 1.0
    raw_prefix: str = ""
    raw_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.result_type.value,
            "suggestions": [s.to_dict() for s in self.suggestions] if self.suggestions else [],
            "confidence": self.confidence,
            "raw_prefix": self.raw_prefix,
            "raw_context": self.raw_context
        }


@dataclass
class OrchestrationResult:
    """Результат оркестрации"""
    outcome: OrchestrationOutcome
    suggestions: Optional[List[CommandSuggestion]] = None
    dialog_id: Optional[str] = None
    first_question: Optional[str] = None
    question_type: str = "text"
    task_id: Optional[str] = None
    reason: Optional[str] = None
    retry_after: Optional[datetime] = None
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "outcome": self.outcome.value,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

        if self.suggestions:
            base["suggestions"] = [s.to_dict() for s in self.suggestions]

        if self.dialog_id:
            base.update({
                "dialog_id": self.dialog_id,
                "first_question": self.first_question,
                "question_type": self.question_type
            })

        if self.task_id:
            base.update({
                "task_id": self.task_id,
                "reason": self.reason,
                "retry_after": self.retry_after.isoformat() if self.retry_after else None,
                "priority": self.priority
            })

        return base


@dataclass
class Command:
    """Семантически размеченная команда"""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    status: CommandStatus = CommandStatus.CONFIRMED
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

    def increment_usage(self):
        """Увеличивает счетчик использования"""
        self.usage_count += 1
        self.last_used = datetime.now()

    def match_score(self, prefix: str, context_tags: List[str]) -> float:
        """Оценка релевантности команды запросу"""
        score = 0.0

        # Совпадение по префиксу (40%)
        if self.text.startswith(prefix):
            score += 0.4

        # Совпадение по тегам (40%)
        common_tags = set(self.tags) & set(context_tags)
        if common_tags:
            score += 0.4 * (len(common_tags) / max(len(self.tags), 1))

        # Частота использования (20%)
        score += 0.2 * min(1.0, self.usage_count / 100)

        return score


@dataclass
class DialogContext:
    """Контекст диалога композиции"""
    dialog_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state: str = "initial"
    user_intent: str = ""
    collected_answers: Dict[str, Any] = field(default_factory=dict)
    current_question_idx: int = 0
    candidate_commands: List[Command] = field(default_factory=list)
    generated_command: Optional[Command] = None


@dataclass
class ClarificationQuestion:
    """Вопрос для уточнения семантики"""
    question_id: str
    text: str
    question_type: str  # "choice", "text", "boolean", "param"
    options: List[Any] = None
    context_key: str = ""
    required: bool = True


@dataclass
class LearningEvent:
    """Событие для обучения движка"""
    full_text: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def _extract_prefix(self, text: str, min_prefix_len: int = 2) -> str:
        """Извлекает префикс для обучения"""
        return text[:min_prefix_len] if len(text) >= min_prefix_len else text