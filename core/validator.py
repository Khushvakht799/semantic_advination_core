# core/validator.py
"""
Модуль для валидации и адаптации результатов предсказания команд.
Обеспечивает фильтрацию, ранжирование и контекстную адаптацию предложений.
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from core.adivinator import AdvinationResult, Suggestion, AdvinationResultType


class ValidationStrategy(Enum):
    """Стратегии валидации"""
    CONFIDENCE_ONLY = "confidence_only"          # Только по доверию
    CONTEXT_AWARE = "context_aware"              # С учетом контекста
    HYBRID = "hybrid"                            # Гибридная стратегия
    ADAPTIVE = "adaptive"                        # Адаптивная стратегия


class AdaptationMode(Enum):
    """Режимы адаптации"""
    FILTER = "filter"                            # Фильтрация
    REORDER = "reorder"                          # Переупорядочивание
    BOOST = "boost"                              # Усиление/ослабление
    TRANSFORM = "transform"                      # Трансформация


@dataclass
class ValidationConfig:
    """Конфигурация валидации"""
    min_confidence: float = 0.5
    min_suggestions: int = 1
    max_suggestions: int = 10
    strategy: ValidationStrategy = ValidationStrategy.HYBRID
    adaptation_mode: AdaptationMode = AdaptationMode.FILTER
    use_context: bool = True
    require_exact_match: bool = False
    enable_logging: bool = True
    fallback_to_partial: bool = True
    context_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.context_weights is None:
            self.context_weights = {
                'domain': 0.3,
                'user_role': 0.2,
                'environment': 0.15,
                'history': 0.15,
                'time': 0.1,
                'location': 0.1
            }


@dataclass
class Context:
    """Контекст выполнения"""
    domain: Optional[str] = None
    user_role: Optional[str] = None
    environment: Optional[str] = None
    history: List[str] = None
    time: Optional[datetime] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.metadata is None:
            self.metadata = {}
        if self.time is None:
            self.time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'domain': self.domain,
            'user_role': self.user_role,
            'environment': self.environment,
            'history': self.history.copy(),
            'time': self.time.isoformat() if self.time else None,
            'location': self.location,
            'metadata': self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Создание из словаря"""
        time_str = data.get('time')
        time = datetime.fromisoformat(time_str) if time_str else None
        
        return cls(
            domain=data.get('domain'),
            user_role=data.get('user_role'),
            environment=data.get('environment'),
            history=data.get('history', []),
            time=time,
            location=data.get('location'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    confidence: float
    suggestions: List[Suggestion]
    filtered_count: int
    validation_strategy: ValidationStrategy
    adaptation_mode: AdaptationMode
    context_used: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'is_valid': self.is_valid,
            'confidence': self.confidence,
            'suggestions_count': len(self.suggestions),
            'filtered_count': self.filtered_count,
            'validation_strategy': self.validation_strategy.value,
            'adaptation_mode': self.adaptation_mode.value,
            'context_used': self.context_used,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class Validator:
    """Класс для валидации и адаптации результатов предсказания"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Инициализация валидатора
        
        Args:
            config: Конфигурация валидации
        """
        self.config = config or ValidationConfig()
        self.logger = self._setup_logger()
        self.context_cache: Dict[str, Context] = {}
        self.validation_rules: List[Callable] = []
        self.adaptation_rules: List[Callable] = []
        
        # Инициализация стандартных правил
        self._init_default_rules()
    
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
        
        if self.config.enable_logging:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        return logger
    
    def _init_default_rules(self) -> None:
        """Инициализация стандартных правил валидации и адаптации"""
        # Правила валидации
        self.add_validation_rule(self._rule_min_confidence)
        self.add_validation_rule(self._rule_min_suggestions)
        self.add_validation_rule(self._rule_exact_match_required)
        
        # Правила адаптации
        self.add_adaptation_rule(self._rule_filter_by_context)
        self.add_adaptation_rule(self._rule_boost_by_history)
        self.add_adaptation_rule(self._rule_adjust_by_time)
        self.add_adaptation_rule(self._rule_limit_suggestions)
    
    def add_validation_rule(self, rule_func: Callable) -> None:
        """
        Добавление пользовательского правила валидации
        
        Args:
            rule_func: Функция правила (принимает suggestions, context, config)
        """
        self.validation_rules.append(rule_func)
        self.logger.info(f"Added validation rule: {rule_func.__name__}")
    
    def add_adaptation_rule(self, rule_func: Callable) -> None:
        """
        Добавление пользовательского правила адаптации
        
        Args:
            rule_func: Функция правила (принимает suggestions, context, config)
        """
        self.adaptation_rules.append(rule_func)
        self.logger.info(f"Added adaptation rule: {rule_func.__name__}")
    
    def is_confident(self, adv_res: AdvinationResult, min_confidence: Optional[float] = None) -> bool:
        """
        Проверка доверия к результату предсказания
        
        Args:
            adv_res: Результат предсказания
            min_confidence: Минимальное доверие (если None, используется из конфига)
            
        Returns:
            True если доверие выше порога
        """
        threshold = min_confidence or self.config.min_confidence
        result = adv_res.confidence >= threshold
        
        self.logger.debug(
            f"Confidence check: {adv_res.confidence:.2f} >= {threshold:.2f} = {result}"
        )
        
        return result
    
    def validate(self, adv_res: AdvinationResult, context: Optional[Context] = None) -> ValidationResult:
        """
        Полная валидация результата предсказания
        
        Args:
            adv_res: Результат предсказания
            context: Контекст выполнения
            
        Returns:
            ValidationResult с результатами валидации
        """
        try:
            self.logger.info(f"Starting validation for query: {adv_res.query}")
            
            # Базовые проверки
            if adv_res.result_type == AdvinationResultType.ERROR:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    suggestions=[],
                    filtered_count=0,
                    validation_strategy=self.config.strategy,
                    adaptation_mode=self.config.adaptation_mode,
                    context_used=False,
                    error_message=adv_res.error_message
                )
            
            if adv_res.result_type == AdvinationResultType.NO_MATCH:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    suggestions=[],
                    filtered_count=0,
                    validation_strategy=self.config.strategy,
                    adaptation_mode=self.config.adaptation_mode,
                    context_used=False,
                    error_message="No matches found"
                )
            
            # Применяем адаптацию
            adapted_suggestions = self.adapt(
                adv_res.suggestions, 
                context or Context()
            )
            
            # Применяем правила валидации
            is_valid = True
            error_messages = []
            
            for rule in self.validation_rules:
                try:
                    rule_result, message = rule(adapted_suggestions, context, self.config)
                    if not rule_result:
                        is_valid = False
                        if message:
                            error_messages.append(message)
                except Exception as e:
                    self.logger.error(f"Error in validation rule {rule.__name__}: {e}")
                    is_valid = False
                    error_messages.append(f"Validation rule error: {e}")
            
            # Вычисляем итоговое доверие
            final_confidence = self._calculate_final_confidence(
                adv_res.confidence, 
                adapted_suggestions
            )
            
            # Создаем результат валидации
            validation_result = ValidationResult(
                is_valid=is_valid,
                confidence=final_confidence,
                suggestions=adapted_suggestions,
                filtered_count=len(adv_res.suggestions) - len(adapted_suggestions),
                validation_strategy=self.config.strategy,
                adaptation_mode=self.config.adaptation_mode,
                context_used=context is not None,
                error_message="; ".join(error_messages) if error_messages else None
            )
            
            self.logger.info(
                f"Validation complete: valid={is_valid}, "
                f"confidence={final_confidence:.2f}, "
                f"suggestions={len(adapted_suggestions)}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                suggestions=[],
                filtered_count=0,
                validation_strategy=self.config.strategy,
                adaptation_mode=self.config.adaptation_mode,
                context_used=False,
                error_message=f"Validation error: {e}"
            )
    
    def adapt(self, suggestions: List[Suggestion], context: Optional[Context] = None) -> List[Suggestion]:
        """
        Адаптация предложений на основе контекста
        
        Args:
            suggestions: Список предложений
            context: Контекст выполнения
            
        Returns:
            Адаптированный список предложений
        """
        if not suggestions:
            return []
        
        context = context or Context()
        adapted = suggestions.copy()
        
        # Применяем правила адаптации в порядке их добавления
        for rule in self.adaptation_rules:
            try:
                adapted = rule(adapted, context, self.config)
                self.logger.debug(
                    f"Applied adaptation rule {rule.__name__}: "
                    f"{len(adapted)} suggestions remaining"
                )
            except Exception as e:
                self.logger.error(f"Error in adaptation rule {rule.__name__}: {e}")
        
        # Сортируем по убыванию доверия
        adapted.sort(key=lambda x: x.confidence, reverse=True)
        
        # Ограничиваем количество предложений
        if len(adapted) > self.config.max_suggestions:
            adapted = adapted[:self.config.max_suggestions]
        
        return adapted
    
    def _rule_min_confidence(self, suggestions: List[Suggestion], 
                           context: Context, config: ValidationConfig) -> Tuple[bool, str]:
        """Правило: минимальное доверие"""
        if not suggestions:
            return False, "No suggestions available"
        
        max_confidence = max(s.confidence for s in suggestions)
        if max_confidence < config.min_confidence:
            return False, f"Max confidence {max_confidence:.2f} below threshold {config.min_confidence:.2f}"
        
        return True, ""
    
    def _rule_min_suggestions(self, suggestions: List[Suggestion],
                            context: Context, config: ValidationConfig) -> Tuple[bool, str]:
        """Правило: минимальное количество предложений"""
        if len(suggestions) < config.min_suggestions:
            return False, f"Only {len(suggestions)} suggestions, need at least {config.min_suggestions}"
        
        return True, ""
    
    def _rule_exact_match_required(self, suggestions: List[Suggestion],
                                 context: Context, config: ValidationConfig) -> Tuple[bool, str]:
        """Правило: требование точного совпадения"""
        if config.require_exact_match:
            exact_matches = [s for s in suggestions if s.confidence == 1.0]
            if not exact_matches:
                return False, "No exact matches found (require_exact_match=True)"
        
        return True, ""
    
    def _rule_filter_by_context(self, suggestions: List[Suggestion],
                              context: Context, config: ValidationConfig) -> List[Suggestion]:
        """Правило: фильтрация по контексту"""
        if not config.use_context or not context.domain:
            return suggestions
        
        filtered = []
        for suggestion in suggestions:
            # Проверяем совпадение домена
            if context.domain in suggestion.matched_tags:
                filtered.append(suggestion)
            # Проверяем метаданные команды
            elif 'domains' in suggestion.metadata:
                if context.domain in suggestion.metadata['domains']:
                    filtered.append(suggestion)
            # Если нет явного совпадения, но есть другие контекстные признаки
            elif self._check_context_compatibility(suggestion, context):
                filtered.append(suggestion)
        
        return filtered
    
    def _rule_boost_by_history(self, suggestions: List[Suggestion],
                             context: Context, config: ValidationConfig) -> List[Suggestion]:
        """Правило: усиление на основе истории"""
        if not context.history:
            return suggestions
        
        boosted = []
        for suggestion in suggestions:
            # Увеличиваем доверие, если команда использовалась ранее
            usage_count = context.history.count(suggestion.command_name)
            if usage_count > 0:
                # Логарифмическое усиление (чем больше использований, тем меньше прирост)
                boost = min(0.2, 0.05 * (usage_count ** 0.5))
                new_suggestion = Suggestion(
                    command_name=suggestion.command_name,
                    confidence=min(1.0, suggestion.confidence + boost),
                    matched_tokens=suggestion.matched_tokens.copy(),
                    command_description=suggestion.command_description,
                    matched_tags=suggestion.matched_tags.copy(),
                    metadata=suggestion.metadata.copy()
                )
                new_suggestion.metadata['history_boost'] = boost
                boosted.append(new_suggestion)
            else:
                boosted.append(suggestion)
        
        return boosted
    
    def _rule_adjust_by_time(self, suggestions: List[Suggestion],
                           context: Context, config: ValidationConfig) -> List[Suggestion]:
        """Правило: корректировка по времени"""
        if not context.time:
            return suggestions
        
        adjusted = []
        current_hour = context.time.hour
        
        for suggestion in suggestions:
            # Пример: усиление команд администрирования в рабочее время
            if 'admin' in suggestion.matched_tags:
                if 9 <= current_hour <= 18:  # Рабочее время
                    boost = 0.1
                    new_confidence = min(1.0, suggestion.confidence + boost)
                    new_suggestion = Suggestion(
                        command_name=suggestion.command_name,
                        confidence=new_confidence,
                        matched_tokens=suggestion.matched_tokens.copy(),
                        command_description=suggestion.command_description,
                        matched_tags=suggestion.matched_tags.copy(),
                        metadata=suggestion.metadata.copy()
                    )
                    new_suggestion.metadata['time_adjustment'] = boost
                    adjusted.append(new_suggestion)
                    continue
            
            # Пример: ослабление системных команд в нерабочее время
            if 'system' in suggestion.matched_tags:
                if current_hour < 6 or current_hour > 22:  # Ночное время
                    penalty = 0.15
                    new_confidence = max(0.0, suggestion.confidence - penalty)
                    new_suggestion = Suggestion(
                        command_name=suggestion.command_name,
                        confidence=new_confidence,
                        matched_tokens=suggestion.matched_tokens.copy(),
                        command_description=suggestion.command_description,
                        matched_tags=suggestion.matched_tags.copy(),
                        metadata=suggestion.metadata.copy()
                    )
                    new_suggestion.metadata['time_adjustment'] = -penalty
                    adjusted.append(new_suggestion)
                    continue
            
            adjusted.append(suggestion)
        
        return adjusted
    
    def _rule_limit_suggestions(self, suggestions: List[Suggestion],
                              context: Context, config: ValidationConfig) -> List[Suggestion]:
        """Правило: ограничение количества предложений"""
        if len(suggestions) <= config.max_suggestions:
            return suggestions
        
        # Оставляем топ-N по доверию
        return suggestions[:config.max_suggestions]
    
    def _check_context_compatibility(self, suggestion: Suggestion, context: Context) -> bool:
        """
        Проверка совместимости предложения с контекстом
        
        Args:
            suggestion: Предложение
            context: Контекст
            
        Returns:
            True если предложение совместимо с контекстом
        """
        # Проверка роли пользователя
        if context.user_role:
            if 'roles' in suggestion.metadata:
                if context.user_role not in suggestion.metadata['roles']:
                    return False
        
        # Проверка окружения
        if context.environment:
            if 'environments' in suggestion.metadata:
                if context.environment not in suggestion.metadata['environments']:
                    return False
        
        # Проверка местоположения
        if context.location:
            if 'locations' in suggestion.metadata:
                if context.location not in suggestion.metadata['locations']:
                    return False
        
        return True
    
    def _calculate_final_confidence(self, base_confidence: float, 
                                  suggestions: List[Suggestion]) -> float:
        """
        Вычисление итогового доверия
        
        Args:
            base_confidence: Базовое доверие из AdvinationResult
            suggestions: Адаптированные предложения
            
        Returns:
            Итоговое доверие
        """
        if not suggestions:
            return 0.0
        
        # Используем максимальное доверие среди предложений
        max_suggestion_confidence = max(s.confidence for s in suggestions)
        
        # Комбинируем с базовым доверием (взвешенное среднее)
        final_confidence = (base_confidence * 0.3 + max_suggestion_confidence * 0.7)
        
        return min(1.0, final_confidence)
    
    def update_config(self, **kwargs) -> None:
        """
        Обновление конфигурации валидатора
        
        Args:
            **kwargs: Параметры конфигурации
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
    
    def save_context(self, key: str, context: Context) -> None:
        """
        Сохранение контекста в кэш
        
        Args:
            key: Ключ для сохранения
            context: Контекст для сохранения
        """
        self.context_cache[key] = context
        self.logger.debug(f"Saved context with key: {key}")
    
    def load_context(self, key: str) -> Optional[Context]:
        """
        Загрузка контекста из кэша
        
        Args:
            key: Ключ контекста
            
        Returns:
            Контекст или None если не найден
        """
        context = self.context_cache.get(key)
        if context:
            self.logger.debug(f"Loaded context with key: {key}")
        return context
    
    def clear_context_cache(self) -> None:
        """Очистка кэша контекстов"""
        self.context_cache.clear()
        self.logger.info("Cleared context cache")
    
    def batch_validate(self, adv_results: List[AdvinationResult], 
                      context: Optional[Context] = None) -> List[ValidationResult]:
        """
        Пакетная валидация результатов
        
        Args:
            adv_results: Список результатов предсказания
            context: Контекст выполнения
            
        Returns:
            Список результатов валидации
        """
        self.logger.info(f"Starting batch validation for {len(adv_results)} results")
        return [self.validate(result, context) for result in adv_results]
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Получение статистики валидатора
        
        Returns:
            Словарь со статистикой
        """
        return {
            'config': {
                'min_confidence': self.config.min_confidence,
                'min_suggestions': self.config.min_suggestions,
                'max_suggestions': self.config.max_suggestions,
                'strategy': self.config.strategy.value,
                'adaptation_mode': self.config.adaptation_mode.value,
                'use_context': self.config.use_context,
                'require_exact_match': self.config.require_exact_match
            },
            'rules': {
                'validation_rules': len(self.validation_rules),
                'adaptation_rules': len(self.adaptation_rules),
                'context_cache_size': len(self.context_cache)
            }
        }


# Фабричные функции
def create_validator(config: Optional[ValidationConfig] = None) -> Validator:
    """
    Создание экземпляра Validator
    
    Args:
        config: Конфигурация валидации
        
    Returns:
        Экземпляр Validator
    """
    return Validator(config)


def get_default_validator() -> Validator:
    """
    Получение глобального экземпляра Validator
    
    Returns:
        Глобальный экземпляр Validator
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = create_validator()
    return _default_validator


# Глобальный экземпляр
_default_validator: Optional[Validator] = None