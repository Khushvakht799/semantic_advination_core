# core/orchestrator.py
"""
Модуль-оркестратор для координации работы Adivinator и Validator.
Управляет workflow семантического предсказания команд.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import uuid

from core.adivinator import Adivinator, AdvinationResult, Suggestion, AdvinationResultType
from core.validator import Validator, ValidationResult, Context


class OrchestrationOutcome(Enum):
    """Результаты оркестрации"""
    SUGGEST_EXACT = "suggest_exact"            # Точные предложения
    SUGGEST_ADAPTED = "suggest_adapted"        # Адаптированные предложения
    SUGGEST_FALLBACK = "suggest_fallback"      # Резервные предложения
    START_DIALOG = "start_dialog"              # Начать диалог
    DEFER = "defer"                            # Отложить решение
    ERROR = "error"                            # Ошибка
    NO_ACTION = "no_action"                    # Бездействие
    RETRY = "retry"                            # Повторить
    MULTIPLE_OPTIONS = "multiple_options"      # Несколько вариантов
    CONFIRMATION_NEEDED = "confirmation_needed"  # Требуется подтверждение


class DialogState(Enum):
    """Состояния диалога"""
    INITIAL = "initial"
    CLARIFYING = "clarifying"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class DialogStep:
    """Шаг диалога"""
    step_id: str
    message: str
    options: List[str] = field(default_factory=list)
    user_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Результат оркестрации"""
    outcome: OrchestrationOutcome
    suggestions: List[Suggestion] = field(default_factory=list)
    dialog_state: DialogState = DialogState.INITIAL
    dialog_steps: List[DialogStep] = field(default_factory=list)
    confidence: float = 0.0
    context_used: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'outcome': self.outcome.value,
            'suggestions_count': len(self.suggestions),
            'suggestions': [s.to_dict() for s in self.suggestions],
            'dialog_state': self.dialog_state.value,
            'dialog_steps_count': len(self.dialog_steps),
            'confidence': self.confidence,
            'context_used': self.context_used,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'request_id': self.request_id
        }
    
    def add_dialog_step(self, message: str, options: List[str] = None) -> 'DialogStep':
        """Добавление шага диалога"""
        step = DialogStep(
            step_id=f"step_{len(self.dialog_steps) + 1}",
            message=message,
            options=options or []
        )
        self.dialog_steps.append(step)
        return step
    
    def get_best_suggestion(self) -> Optional[Suggestion]:
        """Получение лучшего предложения"""
        if not self.suggestions:
            return None
        return max(self.suggestions, key=lambda x: x.confidence)
    
    def get_suggestions_by_threshold(self, threshold: float = 0.3) -> List[Suggestion]:
        """Получение предложений с доверием выше порога"""
        return [s for s in self.suggestions if s.confidence >= threshold]


@dataclass
class OrchestrationConfig:
    """Конфигурация оркестратора"""
    enable_validation: bool = True
    enable_dialog: bool = True
    min_exact_confidence: float = 0.9
    min_partial_confidence: float = 0.5
    max_suggestions: int = 5
    dialog_timeout_seconds: int = 30
    retry_count: int = 2
    fallback_enabled: bool = True
    log_level: str = "INFO"
    workflow_hooks: List[str] = field(default_factory=list)


@dataclass
class WorkflowStep:
    """Шаг workflow"""
    name: str
    handler: Callable
    conditions: List[Callable] = field(default_factory=list)
    priority: int = 0


class Orchestrator:
    """Оркестратор для управления workflow предсказания команд"""
    
    def __init__(self, adivinator: Adivinator, validator: Validator, 
                 config: Optional[OrchestrationConfig] = None):
        """
        Инициализация оркестратора
        
        Args:
            adivinator: Экземпляр Adivinator
            validator: Экземпляр Validator
            config: Конфигурация оркестратора
        """
        self.adivinator = adivinator
        self.validator = validator
        self.config = config or OrchestrationConfig()
        
        self.logger = self._setup_logger()
        self.workflow_steps: List[WorkflowStep] = []
        self.dialog_sessions: Dict[str, List[DialogStep]] = {}
        
        # Инициализация стандартного workflow
        self._init_workflow()
        
        self.logger.info("Orchestrator initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(getattr(logging, self.config.log_level))
        return logger
    
    def _init_workflow(self) -> None:
        """Инициализация стандартного workflow"""
        # Основные шаги workflow
        self.add_workflow_step(
            name="exact_match",
            handler=self._handle_exact_match,
            conditions=[self._condition_exact_match],
            priority=10
        )
        
        self.add_workflow_step(
            name="confident_partial",
            handler=self._handle_confident_partial,
            conditions=[self._condition_confident_partial],
            priority=20
        )
        
        self.add_workflow_step(
            name="multiple_options",
            handler=self._handle_multiple_options,
            conditions=[self._condition_multiple_options],
            priority=30
        )
        
        self.add_workflow_step(
            name="start_dialog",
            handler=self._handle_start_dialog,
            conditions=[self._condition_needs_clarification],
            priority=40
        )
        
        self.add_workflow_step(
            name="fallback",
            handler=self._handle_fallback,
            conditions=[self._condition_needs_fallback],
            priority=50
        )
        
        self.add_workflow_step(
            name="no_match",
            handler=self._handle_no_match,
            conditions=[self._condition_no_match],
            priority=60
        )
    
    def add_workflow_step(self, name: str, handler: Callable, 
                         conditions: List[Callable] = None, priority: int = 0) -> None:
        """
        Добавление шага workflow
        
        Args:
            name: Название шага
            handler: Обработчик шага
            conditions: Условия выполнения
            priority: Приоритет (меньше = выше приоритет)
        """
        step = WorkflowStep(
            name=name,
            handler=handler,
            conditions=conditions or [],
            priority=priority
        )
        self.workflow_steps.append(step)
        self.workflow_steps.sort(key=lambda x: x.priority)
        self.logger.info(f"Added workflow step: {name} (priority: {priority})")
    
    def process(self, prefix: str, context: Optional[Context] = None) -> OrchestrationResult:
        """
        Основной метод обработки запроса
        
        Args:
            prefix: Входной префикс/запрос
            context: Контекст выполнения
            
        Returns:
            Результат оркестрации
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"[{request_id}] Processing request: '{prefix}'")
        
        try:
            # 1. Получаем предсказания
            adv_result = self.adivinator.advinate(prefix)
            self.logger.debug(f"[{request_id}] Advination result: {adv_result.result_type}")
            
            # 2. Создаем базовый результат
            result = OrchestrationResult(
                outcome=OrchestrationOutcome.NO_ACTION,
                confidence=adv_result.confidence,
                request_id=request_id,
                metadata={
                    'original_query': prefix,
                    'advination_type': adv_result.result_type.value
                }
            )
            
            # 3. Выполняем workflow
            for workflow_step in self.workflow_steps:
                if self._check_conditions(workflow_step, adv_result, context):
                    self.logger.debug(f"[{request_id}] Executing workflow step: {workflow_step.name}")
                    
                    step_result = workflow_step.handler(adv_result, context, result)
                    if step_result:
                        result = step_result
                        break
            
            # 4. Добавляем контекстную информацию
            if context:
                result.context_used = True
                result.metadata['context'] = context.to_dict()
            
            # 5. Логируем результат
            self.logger.info(
                f"[{request_id}] Orchestration complete: {result.outcome.value}, "
                f"confidence: {result.confidence:.2f}, "
                f"suggestions: {len(result.suggestions)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Orchestration error: {e}")
            return OrchestrationResult(
                outcome=OrchestrationOutcome.ERROR,
                error_message=str(e),
                request_id=request_id,
                metadata={'error_type': type(e).__name__}
            )
    
    def _check_conditions(self, workflow_step: WorkflowStep, 
                         adv_result: AdvinationResult, 
                         context: Optional[Context]) -> bool:
        """
        Проверка условий выполнения шага workflow
        
        Args:
            workflow_step: Шаг workflow
            adv_result: Результат предсказания
            context: Контекст
            
        Returns:
            True если все условия выполнены
        """
        if not workflow_step.conditions:
            return True
        
        for condition in workflow_step.conditions:
            try:
                if not condition(adv_result, context, self):
                    return False
            except Exception as e:
                self.logger.error(f"Condition check error: {e}")
                return False
        
        return True
    
    # Условия workflow
    def _condition_exact_match(self, adv_result: AdvinationResult, 
                             context: Optional[Context], orchestrator: 'Orchestrator') -> bool:
        """Условие: точное совпадение"""
        return adv_result.result_type == AdvinationResultType.FOUND
    
    def _condition_confident_partial(self, adv_result: AdvinationResult, 
                                   context: Optional[Context], orchestrator: 'Orchestrator') -> bool:
        """Условие: уверенное частичное совпадение"""
        if adv_result.result_type != AdvinationResultType.PARTIAL_FOUND:
            return False
        
        if not self.config.enable_validation:
            return adv_result.confidence >= self.config.min_partial_confidence
        
        return self.validator.is_confident(adv_result, self.config.min_partial_confidence)
    
    def _condition_multiple_options(self, adv_result: AdvinationResult, 
                                  context: Optional[Context], orchestrator: 'Orchestrator') -> bool:
        """Условие: несколько вариантов с похожей уверенностью"""
        if adv_result.result_type != AdvinationResultType.PARTIAL_FOUND:
            return False
        
        if len(adv_result.suggestions) < 2:
            return False
        
        # Проверяем, есть ли несколько вариантов с близкой уверенностью
        confidences = [s.confidence for s in adv_result.suggestions[:3]]
        if len(confidences) >= 2:
            diff = max(confidences) - min(confidences)
            return diff < 0.2  # Разница менее 20%
        
        return False
    
    def _condition_needs_clarification(self, adv_result: AdvinationResult, 
                                     context: Optional[Context], orchestrator: 'Orchestrator') -> bool:
        """Условие: требуется уточнение"""
        if adv_result.result_type != AdvinationResultType.PARTIAL_FOUND:
            return False
        
        if not self.config.enable_dialog:
            return False
        
        # Проверяем, достаточно ли уверенности для диалога
        if adv_result.confidence < 0.3:
            return False
        
        return True
    
    def _condition_needs_fallback(self, adv_result: AdvinationResult, 
                                context: Optional[Context], orchestrator: 'Orchestrator') -> bool:
        """Условие: требуется резервный вариант"""
        if not self.config.fallback_enabled:
            return False
        
        return adv_result.result_type in [
            AdvinationResultType.PARTIAL_FOUND,
            AdvinationResultType.NO_MATCH
        ]
    
    def _condition_no_match(self, adv_result: AdvinationResult, 
                          context: Optional[Context], orchestrator: 'Orchestrator') -> bool:
        """Условие: совпадений не найдено"""
        return adv_result.result_type == AdvinationResultType.NO_MATCH
    
    # Обработчики workflow
    def _handle_exact_match(self, adv_result: AdvinationResult, 
                          context: Optional[Context], 
                          current_result: OrchestrationResult) -> OrchestrationResult:
        """Обработчик: точное совпадение"""
        suggestions = adv_result.suggestions
        
        if self.config.enable_validation and context:
            validation_result = self.validator.validate(adv_result, context)
            if validation_result.is_valid:
                suggestions = validation_result.suggestions
        
        # Ограничиваем количество предложений
        if len(suggestions) > self.config.max_suggestions:
            suggestions = suggestions[:self.config.max_suggestions]
        
        return OrchestrationResult(
            outcome=OrchestrationOutcome.SUGGEST_EXACT,
            suggestions=suggestions,
            confidence=adv_result.confidence,
            dialog_state=DialogState.COMPLETED,
            metadata=current_result.metadata,
            request_id=current_result.request_id
        )
    
    def _handle_confident_partial(self, adv_result: AdvinationResult, 
                                context: Optional[Context], 
                                current_result: OrchestrationResult) -> OrchestrationResult:
        """Обработчик: уверенное частичное совпадение"""
        suggestions = adv_result.suggestions
        
        # Адаптируем предложения
        if context:
            adapted = self.validator.adapt(suggestions, context)
            if adapted:
                suggestions = adapted
        
        # Ограничиваем количество предложений
        if len(suggestions) > self.config.max_suggestions:
            suggestions = suggestions[:self.config.max_suggestions]
        
        return OrchestrationResult(
            outcome=OrchestrationOutcome.SUGGEST_ADAPTED,
            suggestions=suggestions,
            confidence=adv_result.confidence,
            dialog_state=DialogState.COMPLETED,
            metadata=current_result.metadata,
            request_id=current_result.request_id
        )
    
    def _handle_multiple_options(self, adv_result: AdvinationResult, 
                               context: Optional[Context], 
                               current_result: OrchestrationResult) -> OrchestrationResult:
        """Обработчик: несколько вариантов"""
        suggestions = adv_result.suggestions
        
        # Адаптируем предложения
        if context:
            adapted = self.validator.adapt(suggestions, context)
            if adapted:
                suggestions = adapted
        
        # Ограничиваем количество предложений
        suggestions = suggestions[:min(3, self.config.max_suggestions)]
        
        result = OrchestrationResult(
            outcome=OrchestrationOutcome.MULTIPLE_OPTIONS,
            suggestions=suggestions,
            confidence=adv_result.confidence,
            dialog_state=DialogState.CLARIFYING,
            metadata=current_result.metadata,
            request_id=current_result.request_id
        )
        
        # Добавляем шаг диалога для уточнения
        options = [s.command_name for s in suggestions]
        result.add_dialog_step(
            message="Найдено несколько подходящих команд. Уточните, какую вы имели в виду?",
            options=options
        )
        
        return result
    
    def _handle_start_dialog(self, adv_result: AdvinationResult, 
                           context: Optional[Context], 
                           current_result: OrchestrationResult) -> OrchestrationResult:
        """Обработчик: начало диалога"""
        result = OrchestrationResult(
            outcome=OrchestrationOutcome.START_DIALOG,
            suggestions=adv_result.suggestions,
            confidence=adv_result.confidence,
            dialog_state=DialogState.CLARIFYING,
            metadata=current_result.metadata,
            request_id=current_result.request_id
        )
        
        # Добавляем шаг диалога
        if adv_result.suggestions:
            top_commands = [s.command_name for s in adv_result.suggestions[:2]]
            result.add_dialog_step(
                message="Уточните, что вы хотите сделать? Например: " + 
                       ", ".join(top_commands) + " или что-то другое?",
                options=top_commands
            )
        else:
            result.add_dialog_step(
                message="Не совсем понял, что вы хотите сделать. Можете уточнить?",
                options=[]
            )
        
        return result
    
    def _handle_fallback(self, adv_result: AdvinationResult, 
                       context: Optional[Context], 
                       current_result: OrchestrationResult) -> OrchestrationResult:
        """Обработчик: резервный вариант"""
        # Получаем все команды с низким доверием
        fallback_suggestions = self.adivinator.get_all_suggestions(min_confidence=0.1)
        
        if context and fallback_suggestions:
            # Адаптируем резервные предложения
            fallback_suggestions = self.validator.adapt(fallback_suggestions, context)
        
        # Берем топ-N
        fallback_suggestions = fallback_suggestions[:self.config.max_suggestions]
        
        return OrchestrationResult(
            outcome=OrchestrationOutcome.SUGGEST_FALLBACK,
            suggestions=fallback_suggestions,
            confidence=0.1,  # Низкое доверие для резервных вариантов
            dialog_state=DialogState.CLARIFYING,
            metadata=current_result.metadata,
            request_id=current_result.request_id
        )
    
    def _handle_no_match(self, adv_result: AdvinationResult, 
                       context: Optional[Context], 
                       current_result: OrchestrationResult) -> OrchestrationResult:
        """Обработчик: совпадений не найдено"""
        result = OrchestrationResult(
            outcome=OrchestrationOutcome.START_DIALOG,
            suggestions=[],
            confidence=0.0,
            dialog_state=DialogState.CLARIFYING,
            metadata=current_result.metadata,
            request_id=current_result.request_id
        )
        
        result.add_dialog_step(
            message="Не удалось найти подходящую команду. Можете описать, что вы хотите сделать?",
            options=[]
        )
        
        return result
    
    def continue_dialog(self, session_id: str, user_input: str) -> OrchestrationResult:
        """
        Продолжение диалога
        
        Args:
            session_id: ID сессии диалога
            user_input: Ввод пользователя
            
        Returns:
            Результат продолжения диалога
        """
        self.logger.info(f"[{session_id}] Continuing dialog with input: '{user_input}'")
        
        if session_id not in self.dialog_sessions:
            self.dialog_sessions[session_id] = []
        
        # Получаем историю диалога
        dialog_history = self.dialog_sessions[session_id]
        
        # Создаем контекст из истории
        context = Context(
            history=[step.user_response for step in dialog_history if step.user_response]
        )
        
        # Обрабатываем новый ввод
        result = self.process(user_input, context)
        
        # Сохраняем шаг диалога
        if result.dialog_steps:
            last_step = result.dialog_steps[-1]
            last_step.user_response = user_input
            dialog_history.append(last_step)
        
        # Обновляем состояние диалога
        if result.outcome in [OrchestrationOutcome.SUGGEST_EXACT, 
                            OrchestrationOutcome.SUGGEST_ADAPTED]:
            result.dialog_state = DialogState.COMPLETED
        
        return result
    
    def batch_process(self, prefixes: List[str], 
                     context: Optional[Context] = None) -> List[OrchestrationResult]:
        """
        Пакетная обработка запросов
        
        Args:
            prefixes: Список запросов
            context: Контекст выполнения
            
        Returns:
            Список результатов оркестрации
        """
        self.logger.info(f"Batch processing {len(prefixes)} requests")
        return [self.process(prefix, context) for prefix in prefixes]
    
    def update_config(self, **kwargs) -> None:
        """
        Обновление конфигурации
        
        Args:
            **kwargs: Параметры конфигурации
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Получение информации о workflow
        
        Returns:
            Словарь с информацией
        """
        return {
            'workflow_steps': [
                {
                    'name': step.name,
                    'priority': step.priority,
                    'conditions_count': len(step.conditions)
                }
                for step in self.workflow_steps
            ],
            'dialog_sessions_count': len(self.dialog_sessions),
            'config': {
                'enable_validation': self.config.enable_validation,
                'enable_dialog': self.config.enable_dialog,
                'min_exact_confidence': self.config.min_exact_confidence,
                'min_partial_confidence': self.config.min_partial_confidence,
                'max_suggestions': self.config.max_suggestions,
                'fallback_enabled': self.config.fallback_enabled
            }
        }
    
    def clear_dialog_sessions(self) -> None:
        """Очистка сессий диалога"""
        self.dialog_sessions.clear()
        self.logger.info("Cleared all dialog sessions")


# Фабричные функции
def create_orchestrator(adivinator: Optional[Adivinator] = None,
                       validator: Optional[Validator] = None,
                       config: Optional[OrchestrationConfig] = None) -> Orchestrator:
    """
    Создание экземпляра Orchestrator
    
    Args:
        adivinator: Экземпляр Adivinator (если None, создается новый)
        validator: Экземпляр Validator (если None, создается новый)
        config: Конфигурация оркестратора
        
    Returns:
        Экземпляр Orchestrator
    """
    from core.adivinator import create_adivinator
    from core.validator import create_validator
    
    adivinator = adivinator or create_adivinator()
    validator = validator or create_validator()
    
    return Orchestrator(adivinator, validator, config)


def get_default_orchestrator() -> Orchestrator:
    """
    Получение глобального экземпляра Orchestrator
    
    Returns:
        Глобальный экземпляр Orchestrator
    """
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = create_orchestrator()
    return _default_orchestrator


# Глобальный экземпляр
_default_orchestrator: Optional[Orchestrator] = None