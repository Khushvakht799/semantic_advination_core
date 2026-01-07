# core/adivinator.py
"""
Основной модуль для предсказания команд (адивинации).
Обеспечивает инкрементальное автодополнение по мере ввода.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import time
from storage.trie_storage import CommandTrie, Command


class AdvinationResultType(Enum):
    """Типы результатов предсказания"""
    EXACT_MATCH = "exact_match"        # Точное совпадение
    PARTIAL_MATCH = "partial_match"    # Частичное совпадение
    NO_MATCH = "no_match"              # Совпадений не найдено
    MULTIPLE_OPTIONS = "multiple"      # Несколько вариантов
    POPULAR = "popular"                # Популярные команды


@dataclass
class Suggestion:
    """Предложение команды с информацией для автодополнения"""
    command_name: str
    confidence: float
    matched_part: str = ""  # Какая часть команды совпала
    completion_text: str = ""  # Текст для автодополнения
    command_description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'command_name': self.command_name,
            'confidence': self.confidence,
            'matched_part': self.matched_part,
            'completion_text': self.completion_text,
            'command_description': self.command_description,
            'metadata': self.metadata
        }


@dataclass
class AdvinationResult:
    """Результат предсказания"""
    result_type: AdvinationResultType
    suggestions: List[Suggestion]
    query: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'result_type': self.result_type.value,
            'suggestions': [s.to_dict() for s in self.suggestions],
            'query': self.query,
            'timestamp': self.timestamp
        }
    
    def get_best_suggestion(self) -> Optional[Suggestion]:
        """Получение лучшего предложения"""
        if not self.suggestions:
            return None
        return max(self.suggestions, key=lambda x: x.confidence)


class Adivinator:
    """Класс для предсказания команд по мере ввода"""
    
    def __init__(self, trie: CommandTrie):
        self.trie = trie
        self.cache: Dict[str, AdvinationResult] = {}
        self.cache_ttl = 5.0  # 5 секунд TTL для кэша
    
    def predict(self, query: str, cursor_pos: Optional[int] = None) -> AdvinationResult:
        """
        Основной метод предсказания
        
        Args:
            query: Текущий ввод пользователя
            cursor_pos: Позиция курсора (если известна)
            
        Returns:
            AdvinationResult с предсказаниями
        """
        # Нормализуем запрос
        query = query.strip()
        
        # Проверяем кэш
        cache_key = f"{query}_{cursor_pos}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result.timestamp < self.cache_ttl:
                return cached_result
        
        # Обрабатываем пустой запрос
        if not query:
            result = self._predict_empty()
        else:
            # Определяем стратегию предсказания
            if cursor_pos is None or cursor_pos == len(query):
                # Курсор в конце - автодополняем последнее слово
                result = self._predict_autocomplete(query)
            else:
                # Курсор где-то посередине - полный поиск
                result = self._predict_full_search(query)
        
        # Сохраняем в кэш
        result.query = query
        self.cache[cache_key] = result
        
        # Очищаем старый кэш
        self._clean_cache()
        
        return result
    
    def _predict_empty(self) -> AdvinationResult:
        """Предсказание для пустого ввода"""
        # Получаем популярные команды
        popular_commands = self.trie.autocomplete("", limit=5)
        
        suggestions = []
        for cmd, score in popular_commands:
            suggestion = Suggestion(
                command_name=cmd.name,
                confidence=score,
                command_description=cmd.description,
                metadata={
                    'usage_count': cmd.usage_count,
                    'is_popular': True
                }
            )
            suggestions.append(suggestion)
        
        return AdvinationResult(
            result_type=AdvinationResultType.POPULAR,
            suggestions=suggestions
        )
    
    def _predict_autocomplete(self, query: str) -> AdvinationResult:
        """Предсказание с автодополнением (курсор в конце)"""
        # Разбиваем запрос на слова
        words = query.split()
        
        if not words:
            return self._predict_empty()
        
        # Анализируем последнее слово для автодополнения
        last_word = words[-1]
        context_words = words[:-1] if len(words) > 1 else []
        
        # Получаем варианты автодополнения
        completions = self.trie.autocomplete(last_word, limit=10)
        
        # Фильтруем по контексту, если есть
        if context_words and completions:
            completions = self._filter_by_context(completions, context_words)
        
        # Создаём предложения
        suggestions = []
        for cmd, score in completions:
            # Определяем, какая часть команды совпала
            matched_part, completion_text = self._find_completion(cmd.name, last_word)
            
            suggestion = Suggestion(
                command_name=cmd.name,
                confidence=score,
                matched_part=matched_part,
                completion_text=completion_text,
                command_description=cmd.description,
                metadata={
                    'is_exact_match': cmd.name.lower().startswith(last_word.lower()),
                    'usage_count': cmd.usage_count
                }
            )
            suggestions.append(suggestion)
        
        # Определяем тип результата
        if not suggestions:
            result_type = AdvinationResultType.NO_MATCH
        elif len(suggestions) == 1 and suggestions[0].metadata.get('is_exact_match'):
            result_type = AdvinationResultType.EXACT_MATCH
        elif len(suggestions) <= 3:
            result_type = AdvinationResultType.PARTIAL_MATCH
        else:
            result_type = AdvinationResultType.MULTIPLE_OPTIONS
        
        return AdvinationResult(
            result_type=result_type,
            suggestions=suggestions
        )
    
    def _predict_full_search(self, query: str) -> AdvinationResult:
        """Полнотекстовый поиск (курсор не в конце)"""
        # Используем семантический поиск
        semantic_results = self.trie.search_semantic(query, limit=10)
        
        suggestions = []
        for cmd, score in semantic_results:
            # Находим совпадающие части
            matched_parts = self._find_matches(cmd, query)
            
            suggestion = Suggestion(
                command_name=cmd.name,
                confidence=score,
                matched_part=", ".join(matched_parts),
                completion_text="",  # Нет автодополнения при курсоре не в конце
                command_description=cmd.description,
                metadata={
                    'matched_parts': matched_parts,
                    'is_semantic_match': True
                }
            )
            suggestions.append(suggestion)
        
        result_type = AdvinationResultType.PARTIAL_MATCH if suggestions else AdvinationResultType.NO_MATCH
        
        return AdvinationResult(
            result_type=result_type,
            suggestions=suggestions
        )
    
    def _filter_by_context(self, completions: List[Tuple[Command, float]], 
                          context_words: List[str]) -> List[Tuple[Command, float]]:
        """Фильтрация вариантов по контексту"""
        filtered = []
        
        for cmd, score in completions:
            # Проверяем, содержат ли токены команды контекстные слова
            context_match = True
            
            for word in context_words:
                word_lower = word.lower()
                
                # Проверяем в имени команды
                if word_lower in cmd.name.lower():
                    continue
                
                # Проверяем в токенах
                found_in_tokens = False
                for token in cmd.tokens:
                    if word_lower in token.lower():
                        found_in_tokens = True
                        break
                
                if found_in_tokens:
                    continue
                
                # Проверяем в тегах
                found_in_tags = False
                for tag in cmd.tags:
                    if word_lower in tag.lower():
                        found_in_tags = True
                        break
                
                if not found_in_tags:
                    context_match = False
                    break
            
            if context_match:
                # Увеличиваем оценку за совпадение контекста
                filtered.append((cmd, min(1.0, score * 1.2)))
            else:
                # Уменьшаем оценку за несовпадение контекста
                filtered.append((cmd, score * 0.5))
        
        return filtered
    
    def _find_completion(self, command_name: str, prefix: str) -> Tuple[str, str]:
        """Нахождение совпадающей части и текста для автодополнения"""
        if command_name.lower().startswith(prefix.lower()):
            matched = prefix
            completion = command_name[len(prefix):]
        else:
            # Ищем совпадение в токенах
            matched = ""
            completion = ""
            
            # Разбиваем команду на части (по CamelCase, дефисам, подчёркиваниям)
            parts = []
            current_part = ""
            
            for char in command_name:
                if char.isupper() and current_part:
                    parts.append(current_part)
                    current_part = char
                elif char in ['-', '_']:
                    if current_part:
                        parts.append(current_part)
                        current_part = ""
                else:
                    current_part += char
            
            if current_part:
                parts.append(current_part)
            
            # Ищем совпадение с префиксом
            for part in parts:
                if part.lower().startswith(prefix.lower()):
                    matched = prefix
                    completion = command_name[command_name.index(part) + len(prefix):]
                    break
        
        return matched, completion
    
    def _find_matches(self, command: Command, query: str) -> List[str]:
        """Поиск совпадающих частей команды с запросом"""
        matches = []
        query_lower = query.lower()
        
        # Проверяем имя команды
        if query_lower in command.name.lower():
            matches.append(f"name: {command.name}")
        
        # Проверяем токены
        for token in command.tokens:
            if query_lower in token.lower():
                matches.append(f"token: {token}")
        
        # Проверяем теги
        for tag in command.tags:
            if query_lower in tag.lower():
                matches.append(f"tag: {tag}")
        
        return matches
    
    def _clean_cache(self):
        """Очистка устаревшего кэша"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, result in self.cache.items():
            if current_time - result.timestamp > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
    
    def batch_predict(self, queries: List[str]) -> List[AdvinationResult]:
        """Пакетное предсказание"""
        return [self.predict(query) for query in queries]


# Фабричные функции
def create_adivinator(trie: Optional[CommandTrie] = None) -> Adivinator:
    """Создание экземпляра Adivinator"""
    from storage.trie_storage import CommandTrie as DefaultTrie
    
    if trie is None:
        trie = DefaultTrie()
    return Adivinator(trie)


# Глобальный экземпляр
_default_adivinator: Optional[Adivinator] = None

def get_default_adivinator() -> Adivinator:
    """Получение глобального экземпляра Adivinator"""
    global _default_adivinator
    if _default_adivinator is None:
        _default_adivinator = create_adivinator()
    return _default_adivinator
