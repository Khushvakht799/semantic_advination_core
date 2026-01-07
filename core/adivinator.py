# core/adivinator.py
"""
Основной модуль для семантического предсказания команд.
Использует CommandTrie для поиска и ранжирования команд.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from storage.trie_storage import CommandTrie, Command


class AdvinationResultType(Enum):
    """Типы результатов предсказания"""
    FOUND = "found"            # Точное совпадение
    PARTIAL_FOUND = "partial"  # Частичное совпадение
    NO_MATCH = "no_match"      # Совпадений не найдено
    ERROR = "error"            # Ошибка при обработке


@dataclass
class Suggestion:
    """Предложение команды с оценкой релевантности"""
    command_name: str
    confidence: float
    matched_tokens: List[str]
    command_description: str = ""
    matched_tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.matched_tags is None:
            self.matched_tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'command_name': self.command_name,
            'confidence': self.confidence,
            'matched_tokens': self.matched_tokens,
            'command_description': self.command_description,
            'matched_tags': self.matched_tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_command(cls, command: Command, confidence: float = 1.0, 
                    matched_tokens: Optional[List[str]] = None) -> 'Suggestion':
        """Создание предложения из команды"""
        return cls(
            command_name=command.name,
            confidence=confidence,
            matched_tokens=matched_tokens or command.tokens,
            command_description=command.description,
            matched_tags=command.tags,
            metadata=command.metadata
        )


@dataclass
class AdvinationResult:
    """Результат предсказания команд"""
    result_type: AdvinationResultType
    suggestions: List[Suggestion]
    confidence: float
    query: str = ""
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'result_type': self.result_type.value,
            'suggestions': [s.to_dict() for s in self.suggestions],
            'confidence': self.confidence,
            'query': self.query,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
    
    def is_successful(self) -> bool:
        """Проверка успешности поиска"""
        return self.result_type in [AdvinationResultType.FOUND, AdvinationResultType.PARTIAL_FOUND]
    
    def get_best_suggestion(self) -> Optional[Suggestion]:
        """Получение лучшего предложения"""
        if not self.suggestions:
            return None
        return max(self.suggestions, key=lambda x: x.confidence)
    
    def get_suggestions_by_threshold(self, threshold: float = 0.3) -> List[Suggestion]:
        """Получение предложений с доверием выше порога"""
        return [s for s in self.suggestions if s.confidence >= threshold]


class Adivinator:
    """Основной класс для семантического предсказания команд"""
    
    def __init__(self, trie: CommandTrie):
        """
        Инициализация предсказателя
        
        Args:
            trie: Экземпляр CommandTrie для поиска команд
        """
        self.trie = trie
        self.default_threshold = 0.3
        self.exact_match_boost = 1.0
        self.partial_match_penalty = 0.7
    
    def set_threshold(self, threshold: float) -> None:
        """
        Установка порога доверия для частичных совпадений
        
        Args:
            threshold: Пороговое значение (0-1)
        """
        if 0 <= threshold <= 1:
            self.default_threshold = threshold
        else:
            raise ValueError("Threshold must be between 0 and 1")
    
    def advinate(self, prefix: str, threshold: Optional[float] = None, 
                search_type: str = "auto") -> AdvinationResult:
        """
        Основной метод предсказания команд
        
        Args:
            prefix: Входной префикс/запрос для поиска
            threshold: Порог доверия (если None, используется default_threshold)
            search_type: Тип поиска ("auto", "exact", "partial", "tokens", "tags")
            
        Returns:
            AdvinationResult с результатами поиска
        """
        if threshold is None:
            threshold = self.default_threshold
        
        try:
            # Нормализация ввода
            prefix = self._normalize_input(prefix)
            
            if not prefix:
                # Пустой запрос - возвращаем все команды с низким доверием
                all_commands = self.trie.get_all_commands()
                suggestions = [
                    Suggestion.from_command(cmd, confidence=0.1)
                    for cmd in all_commands[:10]  # Ограничиваем количество
                ]
                return AdvinationResult(
                    result_type=AdvinationResultType.PARTIAL_FOUND,
                    suggestions=suggestions,
                    confidence=0.1,
                    query=prefix
                )
            
            # Определяем тип поиска
            if search_type == "auto":
                return self._advinate_auto(prefix, threshold)
            elif search_type == "exact":
                return self._advinate_exact(prefix)
            elif search_type == "partial":
                return self._advinate_partial(prefix, threshold)
            elif search_type == "tokens":
                return self._advinate_by_tokens(prefix)
            elif search_type == "tags":
                return self._advinate_by_tags(prefix)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            return AdvinationResult(
                result_type=AdvinationResultType.ERROR,
                suggestions=[],
                confidence=0.0,
                query=prefix,
                error_message=str(e)
            )
    
    def _normalize_input(self, input_str: str) -> str:
        """
        Нормализация входной строки
        
        Args:
            input_str: Входная строка
            
        Returns:
            Нормализованная строка
        """
        # Приводим к нижнему регистру и убираем лишние пробелы
        normalized = input_str.lower().strip()
        
        # Заменяем множественные пробелы на одиночные
        while '  ' in normalized:
            normalized = normalized.replace('  ', ' ')
        
        return normalized
    
    def _advinate_auto(self, prefix: str, threshold: float) -> AdvinationResult:
        """
        Автоматический поиск с определением оптимальной стратегии
        
        Args:
            prefix: Поисковый запрос
            threshold: Порог доверия
            
        Returns:
            AdvinationResult
        """
        # 1. Пробуем точный поиск по токенам
        token_results = self.trie.search_by_tokens(prefix.split())
        if token_results:
            suggestions = [
                Suggestion.from_command(cmd, confidence=1.0 * self.exact_match_boost)
                for cmd in token_results
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=suggestions,
                confidence=1.0,
                query=prefix
            )
        
        # 2. Пробуем точный поиск по префиксу
        exact_results = self.trie.search_exact(prefix)
        if exact_results:
            suggestions = [
                Suggestion.from_command(cmd, confidence=1.0)
                for cmd in exact_results
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=suggestions,
                confidence=1.0,
                query=prefix
            )
        
        # 3. Пробуем частичный поиск
        partial_results = self.trie.search_partial(prefix, threshold)
        if partial_results:
            suggestions = []
            for cmd, score in partial_results:
                # Находим совпавшие токены
                matched_tokens = self._find_matched_tokens(cmd, prefix)
                suggestion = Suggestion.from_command(
                    cmd, 
                    confidence=score,
                    matched_tokens=matched_tokens
                )
                suggestions.append(suggestion)
            
            max_confidence = max(suggestion.confidence for suggestion in suggestions)
            return AdvinationResult(
                result_type=AdvinationResultType.PARTIAL_FOUND,
                suggestions=suggestions,
                confidence=max_confidence,
                query=prefix
            )
        
        # 4. Пробуем поиск по тегам
        tag_results = self.trie.search_by_tags(prefix.split(), require_all=False)
        if tag_results:
            suggestions = [
                Suggestion.from_command(cmd, confidence=0.5)
                for cmd in tag_results
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.PARTIAL_FOUND,
                suggestions=suggestions,
                confidence=0.5,
                query=prefix
            )
        
        # 5. Ничего не найдено
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            suggestions=[],
            confidence=0.0,
            query=prefix
        )
    
    def _advinate_exact(self, prefix: str) -> AdvinationResult:
        """
        Точный поиск команд
        
        Args:
            prefix: Поисковый запрос
            
        Returns:
            AdvinationResult
        """
        # Ищем точные совпадения
        exact_results = self.trie.search_exact(prefix)
        
        if exact_results:
            suggestions = [
                Suggestion.from_command(cmd, confidence=1.0)
                for cmd in exact_results
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=suggestions,
                confidence=1.0,
                query=prefix
            )
        
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            suggestions=[],
            confidence=0.0,
            query=prefix
        )
    
    def _advinate_partial(self, prefix: str, threshold: float) -> AdvinationResult:
        """
        Частичный поиск команд
        
        Args:
            prefix: Поисковый запрос
            threshold: Порог доверия
            
        Returns:
            AdvinationResult
        """
        # Ищем частичные совпадения
        partial_results = self.trie.search_partial(prefix, threshold)
        
        if partial_results:
            suggestions = []
            for cmd, score in partial_results:
                # Находим совпавшие токены
                matched_tokens = self._find_matched_tokens(cmd, prefix)
                suggestion = Suggestion.from_command(
                    cmd, 
                    confidence=score,
                    matched_tokens=matched_tokens
                )
                suggestions.append(suggestion)
            
            max_confidence = max(suggestion.confidence for suggestion in suggestions)
            return AdvinationResult(
                result_type=AdvinationResultType.PARTIAL_FOUND,
                suggestions=suggestions,
                confidence=max_confidence,
                query=prefix
            )
        
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            suggestions=[],
            confidence=0.0,
            query=prefix
        )
    
    def _advinate_by_tokens(self, prefix: str) -> AdvinationResult:
        """
        Поиск по токенам
        
        Args:
            prefix: Поисковый запрос
            
        Returns:
            AdvinationResult
        """
        tokens = prefix.split()
        token_results = self.trie.search_by_tokens(tokens)
        
        if token_results:
            suggestions = [
                Suggestion.from_command(cmd, confidence=1.0)
                for cmd in token_results
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=suggestions,
                confidence=1.0,
                query=prefix
            )
        
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            suggestions=[],
            confidence=0.0,
            query=prefix
        )
    
    def _advinate_by_tags(self, prefix: str) -> AdvinationResult:
        """
        Поиск по тегам
        
        Args:
            prefix: Поисковый запрос
            
        Returns:
            AdvinationResult
        """
        tags = [tag.strip() for tag in prefix.split(",")]
        tag_results = self.trie.search_by_tags(tags, require_all=False)
        
        if tag_results:
            suggestions = [
                Suggestion.from_command(cmd, confidence=0.7)
                for cmd in tag_results
            ]
            return AdvinationResult(
                result_type=AdvinationResultType.PARTIAL_FOUND,
                suggestions=suggestions,
                confidence=0.7,
                query=prefix
            )
        
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            suggestions=[],
            confidence=0.0,
            query=prefix
        )
    
    def _find_matched_tokens(self, command: Command, query: str) -> List[str]:
        """
        Поиск совпавших токенов в команде
        
        Args:
            command: Команда для проверки
            query: Поисковый запрос
            
        Returns:
            Список совпавших токенов
        """
        matched_tokens = []
        query_lower = query.lower()
        
        for token in command.tokens:
            token_lower = token.lower()
            # Проверяем разные варианты совпадения
            if query_lower in token_lower:
                matched_tokens.append(token)
            elif token_lower in query_lower:
                matched_tokens.append(token)
            elif any(word in token_lower for word in query_lower.split()):
                matched_tokens.append(token)
        
        return matched_tokens
    
    def batch_advinate(self, queries: List[str], threshold: Optional[float] = None) -> List[AdvinationResult]:
        """
        Пакетное предсказание для нескольких запросов
        
        Args:
            queries: Список запросов
            threshold: Порог доверия
            
        Returns:
            Список AdvinationResult
        """
        return [self.advinate(query, threshold) for query in queries]
    
    def get_all_suggestions(self, min_confidence: float = 0.1) -> List[Suggestion]:
        """
        Получение всех команд как предложений
        
        Args:
            min_confidence: Минимальное доверие
            
        Returns:
            Список предложений
        """
        all_commands = self.trie.get_all_commands()
        suggestions = [
            Suggestion.from_command(cmd, confidence=min_confidence)
            for cmd in all_commands
        ]
        return suggestions
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Анализ запроса без выполнения поиска
        
        Args:
            query: Запрос для анализа
            
        Returns:
            Словарь с метаданными анализа
        """
        normalized = self._normalize_input(query)
        words = normalized.split()
        
        analysis = {
            'original_query': query,
            'normalized_query': normalized,
            'word_count': len(words),
            'words': words,
            'is_empty': not normalized,
            'potential_tokens': [],
            'potential_tags': []
        }
        
        # Проверяем, могут ли слова быть токенами
        for word in words:
            # Простая эвристика: если слово есть в токенах каких-либо команд
            for token in self.trie.token_to_commands.keys():
                if word in token.lower():
                    analysis['potential_tokens'].append(word)
                    break
        
        return analysis


# Фабричная функция для создания Adivinator
def create_adivinator(trie: Optional[CommandTrie] = None) -> Adivinator:
    """
    Создание экземпляра Adivinator
    
    Args:
        trie: Экземпляр CommandTrie (если None, создается пустой)
        
    Returns:
        Экземпляр Adivinator
    """
    if trie is None:
        trie = CommandTrie()
    return Adivinator(trie)


# Глобальный экземпляр для удобства
_default_adivinator: Optional[Adivinator] = None


def get_default_adivinator(trie: Optional[CommandTrie] = None) -> Adivinator:
    """
    Получение глобального экземпляра Adivinator
    
    Args:
        trie: Экземпляр CommandTrie (если None и глобальный не создан, создается новый)
        
    Returns:
        Экземпляр Adivinator
    """
    global _default_adivinator
    if _default_adivinator is None:
        _default_adivinator = create_adivinator(trie)
    return _default_adivinator
