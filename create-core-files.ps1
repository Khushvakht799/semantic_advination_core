# create-core-files.ps1
Write-Host "Создаю core файлы..." -ForegroundColor Yellow

# 1. models.py
@'
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid

class AdvinationResultType(Enum):
    """Типы результатов адивинации"""
    FOUND = "FOUND"               # Точное совпадение
    PARTIAL_FOUND = "PARTIAL"     # Похожие варианты
    NO_MATCH = "NO_MATCH"         # Ничего не найдено

class OrchestrationOutcome(Enum):
    """Исходы оркестрации"""
    SUGGEST_EXACT = "SUGGEST_EXACT"        # Предложить точные совпадения
    SUGGEST_ADAPTED = "SUGGEST_ADAPTED"    # Предложить адаптированные
    START_DIALOG = "START_DIALOG"          # Начать диалог композиции
    DEFER = "DEFER"                        # Отложить задачу

class CommandStatus(Enum):
    """Статус команды в БД"""
    CONFIRMED = "confirmed"
    GENERATED = "generated"
    DRAFT = "draft"

@dataclass
class CommandSuggestion:
    """Вариант команды для предложения"""
    text: str
    source: str                    # "exact_match", "partial_match", "adapted"
    match_score: float             # 0.0 - 1.0
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
'@ | Set-Content -Path "core\models.py" -Encoding UTF8

# 2. adivinator.py
@'
from typing import List, Dict, Any, Optional
from .models import *
from storage.trie_storage import CommandTrie

class Adivinator:
    """
    Чистая адивинация.
    Только угадывает варианты команд по префиксу.
    Не принимает решений о семантике или допустимости.
    """
    
    def __init__(self, storage: CommandTrie, config: Dict[str, Any] = None):
        self.storage = storage
        self.config = config or {
            "min_prefix_length": 2,
            "partial_threshold": 0.3,
            "max_exact_results": 5,
            "max_partial_results": 3
        }
    
    def advinate(self, prefix: str, context: Dict[str, Any] = None) -> AdvinationResult:
        """
        Основной метод адивинации.
        Возвращает только факты о найденных вариантах.
        """
        context = context or {}
        
        # 1. Проверка минимальной длины префикса
        if len(prefix) < self.config["min_prefix_length"]:
            return AdvinationResult(
                result_type=AdvinationResultType.NO_MATCH,
                raw_prefix=prefix,
                raw_context=context
            )
        
        # 2. Поиск точных совпадений
        exact_matches = self.storage.search_exact(prefix, limit=self.config["max_exact_results"])
        if exact_matches:
            suggestions = [
                CommandSuggestion(
                    text=match["command"],
                    source="exact_match",
                    match_score=1.0,
                    metadata={
                        "usage_count": match.get("usage_count", 0),
                        "storage_score": match.get("score", 1.0)
                    }
                )
                for match in exact_matches
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=suggestions,
                confidence=1.0,
                raw_prefix=prefix,
                raw_context=context
            )
        
        # 3. Поиск похожих команд (если префикс достаточно длинный)
        if len(prefix) >= 3:
            similar_matches = self.storage.search_similar(
                prefix, 
                threshold=self.config["partial_threshold"],
                limit=self.config["max_partial_results"]
            )
            
            if similar_matches:
                suggestions = []
                for match in similar_matches:
                    suggestions.append(CommandSuggestion(
                        text=match["command"],
                        source="partial_match",
                        match_score=match.get("similarity", 0.5),
                        metadata={
                            "similarity": match.get("similarity", 0),
                            "distance": match.get("distance", 0),
                            "usage_count": match.get("usage_count", 0)
                        }
                    ))
                
                # Confidence - максимальное значение похожести
                max_similarity = max(match.get("similarity", 0) for match in similar_matches)
                return AdvinationResult(
                    result_type=AdvinationResultType.PARTIAL_FOUND,
                    suggestions=suggestions,
                    confidence=max_similarity,
                    raw_prefix=prefix,
                    raw_context=context
                )
        
        # 4. Ничего не найдено
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            raw_prefix=prefix,
            raw_context=context
        )
    
    def learn(self, command_text: str, context: Dict[str, Any] = None):
        """
        Обучение на лету: добавление новой команды в хранилище.
        """
        self.storage.insert({
            "command": command_text,
            "context": context or {},
            "usage_count": 1
        })
'@ | Set-Content -Path "core\adivinator.py" -Encoding UTF8

# 3. validator.py
@'
from typing import List, Dict, Any
from .models import CommandSuggestion

class CommandValidator:
    """
    Валидатор и адаптер команд.
    Знает о семантике и допустимости.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "enable_adaptation": True,
            "known_tokens": {"git", "find", "docker", "python", "ls", "cd", "mkdir"},
            "min_confidence_for_adaptation": 0.5
        }
    
    def can_adapt(self, suggestions: List[CommandSuggestion], context: Dict[str, Any]) -> bool:
        """
        Можно ли адаптировать эти предложения под контекст?
        """
        if not self.config["enable_adaptation"]:
            return False
        
        if not suggestions:
            return False
        
        # Проверяем, есть ли известные токены в предложениях
        for suggestion in suggestions:
            if suggestion.match_score < self.config["min_confidence_for_adaptation"]:
                continue
            
            first_token = suggestion.text.split()[0] if suggestion.text.split() else ""
            if first_token.lower() in self.config["known_tokens"]:
                return True
        
        return False
    
    def adapt(self, suggestions: List[CommandSuggestion], context: Dict[str, Any]) -> List[CommandSuggestion]:
        """
        Адаптирует предложенные команды под текущий контекст.
        """
        if not self.can_adapt(suggestions, context):
            return []
        
        adapted = []
        for suggestion in suggestions:
            # Базовая адаптация: подстановка контекстных значений
            adapted_text = suggestion.text
            
            # Подстановка пути из контекста
            if "current_path" in context and "." in adapted_text:
                adapted_text = adapted_text.replace(".", context["current_path"])
            
            # Подстановка параметров
            if "params" in context:
                for key, value in context["params"].items():
                    placeholder = f"{{{key}}}"
                    if placeholder in adapted_text:
                        adapted_text = adapted_text.replace(placeholder, str(value))
            
            # Создаём адаптированное предложение
            adapted_suggestion = CommandSuggestion(
                text=adapted_text,
                source="adapted",
                match_score=suggestion.match_score * 0.9,  # Чуть ниже уверенность
                metadata={
                    **suggestion.metadata,
                    "original": suggestion.text,
                    "adapted": True,
                    "adaptation_context": {k: v for k, v in context.items() if k != "params"}
                }
            )
            adapted.append(adapted_suggestion)
        
        return adapted
    
    def validate_command(self, command_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет команду на корректность в данном контексте.
        """
        issues = []
        
        # Проверка токенов
        tokens = command_text.split()
        if tokens:
            first_token = tokens[0].lower()
            if first_token not in self.config["known_tokens"]:
                issues.append(f"Неизвестная команда: {first_token}")
        
        # Проверка параметров (если есть в контексте)
        if "required_params" in context:
            for param in context["required_params"]:
                if f"{{{param}}}" in command_text:
                    issues.append(f"Не заполнен параметр: {param}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "command": command_text
        }
'@ | Set-Content -Path "core\validator.py" -Encoding UTF8

# 4. composer.py
@'
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
from .models import DialogContext, Command

class CommandComposer:
    """
    Композитор команд через диалог.
    Создаёт новые команды на основе уточняющих вопросов.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "max_questions": 5,
            "question_templates": {
                "location": "В какой папке выполнить?",
                "pattern": "Какой шаблон поиска?",
                "message": "Введите сообщение коммита:",
                "branch": "В какую ветку?",
                "filename": "Имя файла:"
            }
        }
        self.active_dialogs: Dict[str, DialogContext] = {}
    
    def can_compose(self, prefix: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Определяет, можно ли начать диалог композиции для этого запроса.
        """
        # Эвристики для определения возможности композиции
        decision = {
            "can_compose": False,
            "reason": "",
            "template": None
        }
        
        # Слишком короткий префикс
        if len(prefix.strip()) < 2:
            decision["reason"] = "Слишком общий запрос"
            return decision
        
        # Анализируем префикс на наличие интентов
        prefix_lower = prefix.lower()
        
        # Шаблоны для разных типов команд
        if any(word in prefix_lower for word in ["найди", "поиск", "find", "search"]):
            decision.update({
                "can_compose": True,
                "reason": "Поиск файлов",
                "template": "file_search",
                "questions": ["location", "pattern"]
            })
        
        elif any(word in prefix_lower for word in ["создай", "make", "create", "mkdir"]):
            decision.update({
                "can_compose": True,
                "reason": "Создание файла/папки",
                "template": "create",
                "questions": ["location", "filename"]
            })
        
        elif "git" in prefix_lower or "коммит" in prefix_lower:
            decision.update({
                "can_compose": True,
                "reason": "Git операция",
                "template": "git_commit",
                "questions": ["message"]
            })
        
        else:
            # Общий шаблон для неизвестных команд
            decision.update({
                "can_compose": True,
                "reason": "Общая композиция",
                "template": "generic",
                "questions": ["location"]
            })
        
        return decision
    
    def start_dialog(self, prefix: str, context: Dict[str, Any]) -> DialogContext:
        """
        Начинает новый диалог композиции.
        """
        composition_decision = self.can_compose(prefix, context)
        
        if not composition_decision["can_compose"]:
            raise ValueError(f"Невозможно начать диалог: {composition_decision['reason']}")
        
        dialog = DialogContext(
            user_intent=prefix,
            state="initial"
        )
        
        # Сохраняем информацию о шаблоне
        dialog.collected_answers["template"] = composition_decision["template"]
        dialog.collected_answers["context"] = context
        
        self.active_dialogs[dialog.dialog_id] = dialog
        
        return dialog
    
    def get_next_question(self, dialog_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает следующий вопрос для диалога.
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return None
        
        template = dialog.collected_answers.get("template", "generic")
        questions = dialog.collected_answers.get("questions", ["location"])
        
        if dialog.current_question_idx >= len(questions):
            # Все вопросы заданы
            return None
        
        question_key = questions[dialog.current_question_idx]
        question_text = self.config["question_templates"].get(
            question_key, 
            f"Уточните {question_key}:"
        )
        
        return {
            "question_id": question_key,
            "text": question_text,
            "type": "text",
            "dialog_id": dialog_id,
            "step": dialog.current_question_idx + 1,
            "total_steps": len(questions)
        }
    
    def process_answer(self, dialog_id: str, answer: str) -> Dict[str, Any]:
        """
        Обрабатывает ответ пользователя.
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return {"error": "Диалог не найден"}
        
        # Сохраняем ответ
        current_question_idx = dialog.current_question_idx
        questions = dialog.collected_answers.get("questions", ["location"])
        
        if current_question_idx < len(questions):
            question_key = questions[current_question_idx]
            dialog.collected_answers[question_key] = answer
            dialog.current_question_idx += 1
        
        # Проверяем, завершён ли диалог
        if dialog.current_question_idx >= len(questions):
            # Композиция завершена
            composed_command = self._compose_command(dialog)
            dialog.generated_command = composed_command
            dialog.state = "completed"
            
            return {
                "status": "completed",
                "command": composed_command.text,
                "dialog_id": dialog_id,
                "metadata": {
                    "template": dialog.collected_answers.get("template"),
                    "answers": dialog.collected_answers
                }
            }
        else:
            # Есть ещё вопросы
            return {
                "status": "continue",
                "dialog_id": dialog_id,
                "next_question": self.get_next_question(dialog_id)
            }
    
    def _compose_command(self, dialog: DialogContext) -> Command:
        """
        Создаёт команду на основе ответов.
        """
        template = dialog.collected_answers.get("template", "generic")
        answers = dialog.collected_answers
        
        # Шаблоны команд
        templates = {
            "file_search": "find {location} -name '{pattern}'",
            "create": "mkdir -p {location}/{filename}",
            "git_commit": "git commit -m '{message}'",
            "generic": "{location}"
        }
        
        # Получаем шаблон
        command_template = templates.get(template, "{location}")
        
        # Заполняем шаблон
        command_text = command_template
        for key, value in answers.items():
            if key not in ["template", "context", "questions"]:
                placeholder = f"{{{key}}}"
                if placeholder in command_text:
                    command_text = command_text.replace(placeholder, str(value))
        
        # Создаём объект команды
        return Command(
            text=command_text,
            category=template,
            status="generated",
            parameters=answers
        )
    
    def get_dialog_result(self, dialog_id: str) -> Optional[Command]:
        """Возвращает скомпонованную команду по ID диалога"""
        dialog = self.active_dialogs.get(dialog_id)
        if dialog and dialog.state == "completed":
            return dialog.generated_command
        return None
'@ | Set-Content -Path "core\composer.py" -Encoding UTF8

# 5. orchestrator.py
@'
from typing import Dict, Any, Optional
import time
from datetime import datetime, timedelta
import uuid
from .models import *
from .adivinator import Adivinator
from .validator import CommandValidator
from .composer import CommandComposer

class DeferralStrategy:
    """Стратегия откладывания задач"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "base_delay_hours": 24,
            "max_retries": 3,
            "exponential_backoff": True,
            "priority_delays": {1: 1.0, 2: 0.5, 3: 0.25}  # множители задержки
        }
    
    def calculate_retry_time(self, attempt: int, priority: int = 1) -> Optional[datetime]:
        """Рассчитывает время следующей попытки"""
        if attempt >= self.config["max_retries"]:
            return None
        
        if self.config["exponential_backoff"]:
            delay_hours = self.config["base_delay_hours"] * (2 ** attempt)
        else:
            delay_hours = self.config["base_delay_hours"]
        
        # Корректируем по приоритету
        priority_factor = self.config["priority_delays"].get(priority, 1.0)
        delay_hours *= priority_factor
        
        return datetime.now() + timedelta(hours=delay_hours)

class OrchestrationMetrics:
    """Метрики оркестрации"""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "advination_results": {"FOUND": 0, "PARTIAL": 0, "NO_MATCH": 0},
            "outcomes": {},
            "processing_times": [],
            "errors": 0
        }
    
    def record_advination(self, result_type: AdvinationResultType):
        self.metrics["advination_results"][result_type.value] += 1
    
    def record_orchestration(self, outcome: OrchestrationOutcome, processing_time: float):
        self.metrics["requests_total"] += 1
        self.metrics["outcomes"][outcome.value] = self.metrics["outcomes"].get(outcome.value, 0) + 1
        self.metrics["processing_times"].append(processing_time)
    
    def record_error(self):
        self.metrics["errors"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку метрик"""
        times = self.metrics["processing_times"]
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            **self.metrics,
            "avg_processing_time": avg_time,
            "success_rate": (self.metrics["requests_total"] - self.metrics["errors"]) / 
                           max(self.metrics["requests_total"], 1)
        }

class ProductionOrchestrator:
    """
    Основной оркестратор workflow.
    Управляет всем процессом от запроса до результата.
    """
    
    def __init__(self, 
                 adivinator: Adivinator = None,
                 validator: CommandValidator = None,
                 composer: CommandComposer = None,
                 config: Dict[str, Any] = None):
        
        self.config = {
            "enable_adaptation": True,
            "enable_composition": True,
            "enable_deferral": True,
            "deferral_strategy": DeferralStrategy(),
            **config or {}
        }
        
        self.adivinator = adivinator or Adivinator(None)
        self.validator = validator or CommandValidator()
        self.composer = composer or CommandComposer()
        self.metrics = OrchestrationMetrics()
        self.deferred_tasks: Dict[str, Dict] = {}
        
    def process_request(self, 
                       prefix: str, 
                       context: Dict[str, Any] = None,
                       user_id: str = None) -> OrchestrationResult:
        """
        Основной метод обработки запроса.
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # 1. Чистая адивинация
            adv_result = self.adivinator.advinate(prefix, context)
            self.metrics.record_advination(adv_result.result_type)
            
            # 2. Обработка результатов адивинации
            if adv_result.result_type == AdvinationResultType.FOUND:
                result = self._handle_found(adv_result, context)
            
            elif adv_result.result_type == AdvinationResultType.PARTIAL_FOUND:
                result = self._handle_partial(adv_result, context)
            
            else:  # NO_MATCH
                result = self._handle_no_match(prefix, context)
            
            # 3. Записываем метрики
            elapsed = time.time() - start_time
            self.metrics.record_orchestration(result.outcome, elapsed)
            
            # 4. Добавляем метаданные
            result.metadata.update({
                "processing_time_ms": elapsed * 1000,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.metrics.record_error()
            
            # При ошибке → откладываем
            if self.config["enable_deferral"]:
                task_id = self._defer_task(
                    prefix, context, 
                    f"Ошибка обработки: {str(e)}", 
                    priority=3
                )
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.DEFER,
                    task_id=task_id,
                    reason="Внутренняя ошибка системы",
                    retry_after=datetime.now() + timedelta(hours=1),
                    priority=3,
                    metadata={"error": str(e)}
                )
            else:
                # Если откладывание отключено, возвращаем ошибку
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.DEFER,
                    reason=f"Ошибка: {str(e)}",
                    metadata={"error": str(e), "deferral_disabled": True}
                )
    
    def _handle_found(self, adv_result: AdvinationResult, context: Dict) -> OrchestrationResult:
        """Обработка точных совпадений"""
        return OrchestrationResult(
            outcome=OrchestrationOutcome.SUGGEST_EXACT,
            suggestions=adv_result.suggestions,
            metadata={"source": "exact_match", "confidence": adv_result.confidence}
        )
    
    def _handle_partial(self, adv_result: AdvinationResult, context: Dict) -> OrchestrationResult:
        """Обработка частичных совпадений"""
        if self.config["enable_adaptation"]:
            # Пытаемся адаптировать
            adapted = self.validator.adapt(adv_result.suggestions, context)
            if adapted:
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.SUGGEST_ADAPTED,
                    suggestions=adapted,
                    metadata={"source": "adapted", "original_confidence": adv_result.confidence}
                )
        
        # Если адаптация невозможна → пробуем композицию
        if self.config["enable_composition"]:
            return self._try_composition(adv_result.raw_prefix, context)
        else:
            # Ничего не можем предложить
            return self._defer_or_fail(adv_result.raw_prefix, context, "Не удалось адаптировать частичные совпадения")
    
    def _handle_no_match(self, prefix: str, context: Dict) -> OrchestrationResult:
        """Обработка отсутствия совпадений"""
        if self.config["enable_composition"]:
            return self._try_composition(prefix, context)
        else:
            return self._defer_or_fail(prefix, context, "Нет совпадений и композиция отключена")
    
    def _try_composition(self, prefix: str, context: Dict) -> OrchestrationResult:
        """Попытка композиции новой команды"""
        composition_decision = self.composer.can_compose(prefix, context)
        
        if composition_decision["can_compose"]:
            # Начинаем диалог композиции
            dialog = self.composer.start_dialog(prefix, context)
            next_question = self.composer.get_next_question(dialog.dialog_id)
            
            if next_question:
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.START_DIALOG,
                    dialog_id=dialog.dialog_id,
                    first_question=next_question["text"],
                    question_type="text",
                    metadata={
                        "template": composition_decision["template"],
                        "reason": composition_decision["reason"]
                    }
                )
        
        # Не можем скомпоновать → откладываем
        return self._defer_or_fail(
            prefix, context, 
            f"Невозможно скомпоновать: {composition_decision.get('reason', 'неизвестно')}"
        )
    
    def _defer_or_fail(self, prefix: str, context: Dict, reason: str) -> OrchestrationResult:
        """Откладывает задачу или возвращает ошибку"""
        if self.config["enable_deferral"]:
            task_id = self._defer_task(prefix, context, reason, priority=2)
            return OrchestrationResult(
                outcome=OrchestrationOutcome.DEFER,
                task_id=task_id,
                reason=reason,
                retry_after=datetime.now() + timedelta(hours=24),
                priority=2
            )
        else:
            # Если откладывание отключено, возвращаем пустой результат
            return OrchestrationResult(
                outcome=OrchestrationOutcome.DEFER,
                reason=reason,
                metadata={"deferral_disabled": True}
            )
    
    def _defer_task(self, prefix: str, context: Dict, reason: str, priority: int = 1) -> str:
        """Создаёт отложенную задачу"""
        task_id = str(uuid.uuid4())
        strategy = self.config["deferral_strategy"]
        
        retry_after = strategy.calculate_retry_time(0, priority)
        
        task = {
            "task_id": task_id,
            "prefix": prefix,
            "context": context,
            "reason": reason,
            "created_at": datetime.now(),
            "retry_after": retry_after,
            "attempt": 0,
            "priority": priority,
            "status": "deferred"
        }
        
        self.deferred_tasks[task_id] = task
        
        # Здесь можно сохранять задачи в файл или БД
        # self._save_deferred_task(task)
        
        return task_id
    
    def continue_dialog(self, dialog_id: str, answer: str) -> OrchestrationResult:
        """Продолжение диалога композиции"""
        try:
            result = self.composer.process_answer(dialog_id, answer)
            
            if result["status"] == "completed":
                # Команда скомпонована
                command = self.composer.get_dialog_result(dialog_id)
                
                if command:
                    # Сохраняем новую команду
                    self.adivinator.learn(command.text, {})
                    
                    return OrchestrationResult(
                        outcome=OrchestrationOutcome.SUGGEST_EXACT,
                        suggestions=[
                            CommandSuggestion(
                                text=command.text,
                                source="composed",
                                match_score=0.9,
                                metadata={"dialog_id": dialog_id, "composed": True}
                            )
                        ],
                        metadata={"dialog_completed": True, "command_id": command.command_id}
                    )
            
            elif result["status"] == "continue":
                # Продолжаем диалог
                next_q = result["next_question"]
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.START_DIALOG,
                    dialog_id=dialog_id,
                    first_question=next_q["text"],
                    question_type="text",
                    metadata={"step": next_q["step"], "total_steps": next_q["total_steps"]}
                )
            
        except Exception as e:
            return OrchestrationResult(
                outcome=OrchestrationOutcome.DEFER,
                reason=f"Ошибка в диалоге: {str(e)}",
                metadata={"error": str(e), "dialog_id": dialog_id}
            )
        
        return OrchestrationResult(
            outcome=OrchestrationOutcome.DEFER,
            reason="Неизвестное состояние диалога",
            metadata={"dialog_id": dialog_id}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает текущие метрики"""
        return self.metrics.get_summary()
'@ | Set-Content -Path "core\orchestrator.py" -Encoding UTF8

# 6. __init__.py
@'
"""
Core module for semantic advination system.
"""

from .models import *
from .adivinator import Adivinator
from .validator import CommandValidator
from .composer import CommandComposer
from .orchestrator import ProductionOrchestrator, DeferralStrategy, OrchestrationMetrics

__version__ = "1.0.0"
__all__ = [
    'Adivinator',
    'CommandValidator',
    'CommandComposer',
    'ProductionOrchestrator',
    'DeferralStrategy',
    'OrchestrationMetrics',
    'AdvinationResult',
    'OrchestrationResult',
    'CommandSuggestion',
    'Command',
    'DialogContext'
]
'@ | Set-Content -Path "core\__init__.py" -Encoding UTF8

Write-Host "✅ Core файлы созданы" -ForegroundColor Green