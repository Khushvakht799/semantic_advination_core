# core/validator.py
"""
Модуль для валидации и фильтрации предложений.
Обеспечивает контекстную адаптацию результатов.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from core.adivinator import Suggestion, AdvinationResult


class ValidationMode(Enum):
    """Режимы валидации"""
    STRICT = "strict"      # Только точные совпадения
    NORMAL = "normal"      # Обычный режим
    LENIENT = "lenient"    # Либеральный режим


@dataclass
class Context:
    """Контекст выполнения"""
    current_directory: str = ""
    user_roles: List[str] = field(default_factory=list)
    environment: str = "default"
    recent_commands: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    suggestions: List[Suggestion]
    filtered_count: int
    mode: ValidationMode
    message: str = ""


class Validator:
    """Валидатор для фильтрации предложений"""
    
    def __init__(self, mode: ValidationMode = ValidationMode.NORMAL):
        self.mode = mode
        self.filters: List[Callable] = []
        self._init_default_filters()
    
    def _init_default_filters(self):
        """Инициализация стандартных фильтров"""
        self.add_filter(self._filter_by_confidence)
        self.add_filter(self._filter_by_recent_usage)
    
    def add_filter(self, filter_func: Callable):
        """Добавление пользовательского фильтра"""
        self.filters.append(filter_func)
    
    def validate(self, suggestions: List[Suggestion], context: Optional[Context] = None) -> ValidationResult:
        """
        Валидация списка предложений
        
        Args:
            suggestions: Список предложений для валидации
            context: Контекст выполнения
            
        Returns:
            ValidationResult
        """
        if not suggestions:
            return ValidationResult(
                is_valid=False,
                suggestions=[],
                filtered_count=0,
                mode=self.mode,
                message="Нет предложений для валидации"
            )
        
        original_count = len(suggestions)
        filtered = suggestions.copy()
        
        # Применяем все фильтры
        for filter_func in self.filters:
            filtered = filter_func(filtered, context or Context(), self.mode)
        
        # Проверяем минимальное доверие в зависимости от режима
        confidence_threshold = {
            ValidationMode.STRICT: 0.8,
            ValidationMode.NORMAL: 0.5,
            ValidationMode.LENIENT: 0.3
        }.get(self.mode, 0.5)
        
        filtered = [s for s in filtered if s.confidence >= confidence_threshold]
        
        # Сортируем по убыванию доверия
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        is_valid = len(filtered) > 0
        
        return ValidationResult(
            is_valid=is_valid,
            suggestions=filtered,
            filtered_count=original_count - len(filtered),
            mode=self.mode,
            message=f"Отфильтровано {original_count - len(filtered)} предложений"
        )
    
    def _filter_by_confidence(self, suggestions: List[Suggestion], 
                             context: Context, mode: ValidationMode) -> List[Suggestion]:
        """Фильтрация по доверию"""
        if mode == ValidationMode.LENIENT:
            return suggestions
        
        # Для strict и normal режимов применяем разные пороги
        threshold = 0.7 if mode == ValidationMode.STRICT else 0.4
        return [s for s in suggestions if s.confidence >= threshold]
    
    def _filter_by_recent_usage(self, suggestions: List[Suggestion],
                               context: Context, mode: ValidationMode) -> List[Suggestion]:
        """Фильтрация на основе недавнего использования"""
        if not context.recent_commands:
            return suggestions
        
        # Создаём новый список с усилением для недавно использованных команд
        enhanced = []
        
        for suggestion in suggestions:
            # Если команда недавно использовалась, увеличиваем её доверие
            if suggestion.command_name in context.recent_commands:
                # Создаём копию с увеличенным доверием
                enhanced_suggestion = Suggestion(
                    command_name=suggestion.command_name,
                    confidence=min(1.0, suggestion.confidence * 1.3),
                    matched_part=suggestion.matched_part,
                    completion_text=suggestion.completion_text,
                    command_description=suggestion.command_description,
                    metadata={
                        **suggestion.metadata,
                        'recently_used': True,
                        'original_confidence': suggestion.confidence
                    }
                )
                enhanced.append(enhanced_suggestion)
            else:
                enhanced.append(suggestion)
        
        return enhanced
    
    def adapt(self, suggestions: List[Suggestion], context: Optional[Context] = None) -> List[Suggestion]:
        """
        Адаптация предложений под контекст (простая фильтрация)
        
        Args:
            suggestions: Список предложений
            context: Контекст выполнения
            
        Returns:
            Адаптированный список предложений
        """
        if not context:
            return suggestions
        
        context_obj = context if isinstance(context, Context) else Context(**context)
        
        # Простая фильтрация по домену (если указан в metadata контекста)
        if 'domain' in context_obj.metadata:
            domain = context_obj.metadata['domain']
            filtered = []
            
            for suggestion in suggestions:
                # Проверяем теги команды (если есть в metadata)
                cmd_tags = suggestion.metadata.get('tags', [])
                if domain in cmd_tags or not cmd_tags:
                    filtered.append(suggestion)
            
            return filtered
        
        return suggestions


# Фабричные функции
def create_validator(mode: ValidationMode = ValidationMode.NORMAL) -> Validator:
    """Создание экземпляра Validator"""
    return Validator(mode)


# Глобальный экземпляр
_default_validator: Optional[Validator] = None

def get_default_validator() -> Validator:
    """Получение глобального экземпляра Validator"""
    global _default_validator
    if _default_validator is None:
        _default_validator = create_validator()
    return _default_validator