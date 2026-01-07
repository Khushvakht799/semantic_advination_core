# core/orchestrator.py
"""
Модуль-оркестратор для координации автодополнения команд.
Управляет workflow от ввода до отображения предложений.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
from core.adivinator import Adivinator, AdvinationResult, AdvinationResultType, Suggestion
from core.validator import Validator, Context, ValidationResult


class OrchestrationMode(Enum):
    """Режимы оркестрации"""
    INTERACTIVE = "interactive"    # Интерактивный режим (по мере ввода)
    BATCH = "batch"                # Пакетный режим
    TAB_COMPLETION = "tab"         # Режим Tab-автодополнения


@dataclass
class OrchestrationResult:
    """Результат оркестрации"""
    suggestions: List[Suggestion]
    display_text: str = ""
    should_auto_complete: bool = False
    auto_complete_text: str = ""
    mode: OrchestrationMode = OrchestrationMode.INTERACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'suggestions': [s.to_dict() for s in self.suggestions],
            'display_text': self.display_text,
            'should_auto_complete': self.should_auto_complete,
            'auto_complete_text': self.auto_complete_text,
            'mode': self.mode.value,
            'metadata': self.metadata
        }


class Orchestrator:
    """Оркестратор для управления workflow автодополнения"""
    
    def __init__(self, adivinator: Adivinator, validator: Validator):
        self.adivinator = adivinator
        self.validator = validator
        self.context = Context()
        self.last_query = ""
        self.last_result = None
        
    def process(self, query: str, context: Optional[Dict] = None, 
                mode: OrchestrationMode = OrchestrationMode.INTERACTIVE) -> OrchestrationResult:
        """
        Основной метод обработки запроса
        
        Args:
            query: Ввод пользователя
            context: Контекст выполнения (опционально)
            mode: Режим оркестрации
            
        Returns:
            OrchestrationResult
        """
        # Обновляем контекст, если предоставлен
        if context:
            self._update_context(context)
        
        # Сохраняем последний запрос
        self.last_query = query
        
        # Получаем предсказания
        adv_result = self.adivinator.predict(query)
        self.last_result = adv_result
        
        # Валидируем предсказания
        validation_result = self.validator.validate(adv_result.suggestions, self.context)
        
        # Адаптируем под контекст
        adapted_suggestions = self.validator.adapt(validation_result.suggestions, self.context)
        
        # Создаём результат оркестрации
        return self._create_orchestration_result(
            query=query,
            suggestions=adapted_suggestions,
            adv_result=adv_result,
            mode=mode
        )
    
    def _update_context(self, context_data: Dict[str, Any]):
        """Обновление контекста"""
        # Добавляем текущий запрос в историю
        if 'recent_queries' not in self.context.metadata:
            self.context.metadata['recent_queries'] = []
        
        self.context.metadata['recent_queries'].append({
            'query': self.last_query,
            'timestamp': time.time()
        })
        
        # Ограничиваем историю
        if len(self.context.metadata['recent_queries']) > 10:
            self.context.metadata['recent_queries'] = self.context.metadata['recent_queries'][-10:]
        
        # Обновляем другие поля контекста
        for key, value in context_data.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value
    
    def _create_orchestration_result(self, query: str, suggestions: List[Suggestion],
                                   adv_result: AdvinationResult, mode: OrchestrationMode) -> OrchestrationResult:
        """Создание результата оркестрации"""
        result = OrchestrationResult(
            suggestions=suggestions,
            mode=mode
        )
        
        # Определяем, нужно ли автоматическое дополнение
        if suggestions and mode == OrchestrationMode.TAB_COMPLETION:
            best_suggestion = suggestions[0]
            if best_suggestion.completion_text:
                result.should_auto_complete = True
                result.auto_complete_text = best_suggestion.completion_text
        
        # Формируем текст для отображения
        result.display_text = self._format_display_text(query, suggestions, adv_result.result_type)
        
        # Добавляем метаданные
        result.metadata = {
            'query': query,
            'result_type': adv_result.result_type.value,
            'suggestion_count': len(suggestions),
            'timestamp': time.time()
        }
        
        return result
    
    def _format_display_text(self, query: str, suggestions: List[Suggestion], 
                           result_type: AdvinationResultType) -> str:
        """Форматирование текста для отображения"""
        if not suggestions:
            return "Нет вариантов для автодополнения"
        
        if result_type == AdvinationResultType.EXACT_MATCH and len(suggestions) == 1:
            suggestion = suggestions[0]
            return f"Нажми Tab для автодополнения: {query}{suggestion.completion_text}"
        
        if result_type == AdvinationResultType.POPULAR:
            return "Популярные команды:"
        
        if len(suggestions) == 1:
            return f"1 вариант: {suggestions[0].command_name}"
        
        return f"{len(suggestions)} вариантов:"
    
    def get_tab_completion(self, query: str) -> Optional[str]:
        """
        Получение текста для Tab-автодополнения
        
        Args:
            query: Текущий ввод
            
        Returns:
            Текст для автодополнения или None
        """
        result = self.process(query, mode=OrchestrationMode.TAB_COMPLETION)
        
        if result.should_auto_complete:
            return result.auto_complete_text
        
        return None
    
    def record_command_execution(self, command_name: str):
        """Запись выполнения команды"""
        # Обновляем статистику использования в Trie
        if hasattr(self.adivinator.trie, 'record_command_usage'):
            self.adivinator.trie.record_command_usage(command_name)
        
        # Добавляем в историю выполнения
        self.context.recent_commands.append(command_name)
        
        # Ограничиваем историю
        if len(self.context.recent_commands) > 20:
            self.context.recent_commands = self.context.recent_commands[-20:]
    
    def get_suggestions_display(self, query: str, limit: int = 5) -> str:
        """
        Получение форматированного отображения предложений
        
        Args:
            query: Текущий ввод
            limit: Максимальное количество предложений
            
        Returns:
            Форматированная строка с предложениями
        """
        result = self.process(query)
        
        if not result.suggestions:
            return "  (нет вариантов)"
        
        display_lines = []
        
        for i, suggestion in enumerate(result.suggestions[:limit]):
            confidence_bar = "█" * int(suggestion.confidence * 10)
            
            # Форматируем строку
            if suggestion.completion_text:
                line = f"  {i+1}. {query}\033[90m{suggestion.completion_text}\033[0m"
            else:
                line = f"  {i+1}. {suggestion.command_name}"
            
            line += f" [{confidence_bar}]"
            
            if suggestion.command_description:
                line += f" - {suggestion.command_description}"
            
            display_lines.append(line)
        
        return "\n".join(display_lines)


# Фабричные функции
def create_orchestrator(adivinator: Optional[Adivinator] = None,
                       validator: Optional[Validator] = None) -> 'Orchestrator':
    """Создание экземпляра Orchestrator"""
    from core.adivinator import create_adivinator, get_default_adivinator
    from core.validator import create_validator, get_default_validator
    
    adivinator = adivinator or get_default_adivinator()
    validator = validator or get_default_validator()
    
    return Orchestrator(adivinator, validator)


# Глобальный экземпляр
_default_orchestrator: Optional[Orchestrator] = None

def get_default_orchestrator() -> Orchestrator:
    """Получение глобального экземпляра Orchestrator"""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = create_orchestrator()
    return _default_orchestrator