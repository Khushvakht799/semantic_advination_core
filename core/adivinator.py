# core\adivinator.py
"""
Ядро семантического предсказания команд — Adivinator.
Минимальный, эффективный, расширяемый.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import heapq


@dataclass
class Suggestion:
    """Предложение команды с оценкой."""
    command_name: str
    confidence: float  # 0.0 - 1.0
    matched_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'Suggestion') -> bool:
        """Для сортировки в куче (max-heap по confidence)."""
        return self.confidence > other.confidence  # обратно для max-heap


class TrieNode:
    """Узел префиксного дерева для хранения команд."""
    
    __slots__ = ('children', 'command_names', 'is_terminal')
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.command_names: List[str] = []  # команды, заканчивающиеся в этом узле
        self.is_terminal: bool = False


class Adivinator:
    """
    Ядро предсказания команд на основе префиксного дерева.
    """
    
    def __init__(self):
        self.trie_root = TrieNode()
        self.commands: Dict[str, List[str]] = {}  # name -> tokens
        self._build_cache: Dict[str, List[Suggestion]] = {}
    
    def add_command(self, name: str, tokens: List[str]) -> None:
        """
        Добавляет команду в Trie.
        """
        self.commands[name] = tokens.copy()
        
        # Вставляем каждый токен в Trie
        node = self.trie_root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        
        node.is_terminal = True
        node.command_names.append(name)
        
        # Сбрасываем кэш при изменениях
        self._build_cache.clear()
    
    def _exact_match(self, tokens: List[str]) -> Optional[str]:
        """
        Ищет точное совпадение команды.
        Возвращает имя команды или None.
        """
        node = self.trie_root
        for token in tokens:
            if token not in node.children:
                return None
            node = node.children[token]
        
        return node.command_names[0] if node.command_names else None
    
    def _partial_match(self, tokens: List[str], max_suggestions: int = 5) -> List[Suggestion]:
        """
        Ищет частичные совпадения команд.
        Возвращает ранжированный список предложений.
        """
        if not tokens:
            return []
        
        # Кэшируем результаты для часто используемых запросов
        cache_key = '_'.join(tokens)
        if cache_key in self._build_cache:
            return self._build_cache[cache_key][:max_suggestions]
        
        suggestions = []
        
        # Ищем команды, начинающиеся с данных токенов
        node = self.trie_root
        for i, token in enumerate(tokens):
            if token not in node.children:
                # Если токен не найден, предлагаем команды из текущего узла
                self._collect_suggestions(node, tokens[:i], suggestions)
                break
            node = node.children[token]
        else:
            # Все токены найдены — собираем команды из этого узла и его детей
            self._collect_suggestions(node, tokens, suggestions, exact=True)
        
        # Ранжируем по количеству совпавших токенов
        ranked = self._rank_suggestions(tokens, suggestions)
        
        # Ограничиваем количество и кэшируем
        result = ranked[:max_suggestions]
        self._build_cache[cache_key] = ranked
        
        return result
    
    def _collect_suggestions(self, 
                            node: TrieNode, 
                            matched_tokens: List[str],
                            suggestions: List[Suggestion],
                            exact: bool = False) -> None:
        """
        Собирает предложения из узла и его поддерева.
        """
        if node.is_terminal:
            for cmd_name in node.command_names:
                confidence = 1.0 if exact else len(matched_tokens) / len(self.commands[cmd_name])
                suggestions.append(
                    Suggestion(
                        command_name=cmd_name,
                        confidence=confidence,
                        matched_tokens=matched_tokens.copy(),
                        metadata={'match_type': 'exact' if exact else 'partial'}
                    )
                )
        
        # Рекурсивно обходим детей
        for token, child_node in node.children.items():
            self._collect_suggestions(child_node, matched_tokens + [token], suggestions)
    
    def _rank_suggestions(self, query_tokens: List[str], 
                         suggestions: List[Suggestion]) -> List[Suggestion]:
        """
        Ранжирует предложения по релевантности.
        """
        if not suggestions:
            return []
        
        scored = []
        for sug in suggestions:
            cmd_tokens = self.commands[sug.command_name]
            
            # Базовый score — совпадение префикса
            score = sug.confidence
            
            # Бонус за полное совпадение длины
            if len(cmd_tokens) == len(query_tokens):
                score *= 1.2
            
            # Бонус за короткие команды (менее 3 токенов)
            if len(cmd_tokens) <= 2:
                score *= 1.1
            
            scored.append((score, sug))
        
        # Сортируем по убыванию score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Обновляем confidence в предложениях
        result = []
        max_score = scored[0][0] if scored else 1.0
        
        for score, sug in scored:
            normalized_score = score / max_score if max_score > 0 else score
            sug.confidence = min(1.0, normalized_score)  # обрезаем до 1.0
            result.append(sug)
        
        return result
    
    def suggest(self, 
               input_text: str, 
               max_suggestions: int = 5) -> List[Suggestion]:
        """
        Основной метод: принимает текст, возвращает предложения команд.
        """
        # Простая токенизация (можно заменить на utils/tokenizer.py)
        tokens = [t.lower().strip() for t in input_text.split() if t.strip()]
        
        if not tokens:
            return []
        
        # Сначала проверяем точное совпадение
        exact_name = self._exact_match(tokens)
        if exact_name:
            return [Suggestion(
                command_name=exact_name,
                confidence=1.0,
                matched_tokens=tokens.copy(),
                metadata={'match_type': 'exact'}
            )]
        
        # Если нет точного — ищем частичные
        return self._partial_match(tokens, max_suggestions)
    
    def batch_suggest(self, 
                     inputs: List[str], 
                     max_suggestions: int = 3) -> Dict[str, List[Suggestion]]:
        """
        Пакетное предсказание для нескольких входов.
        """
        return {text: self.suggest(text, max_suggestions) for text in inputs}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по Adivinator.
        """
        def count_nodes(node: TrieNode) -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(self.trie_root) - 1  # без корня
        
        return {
            'total_commands': len(self.commands),
            'total_trie_nodes': total_nodes,
            'avg_tokens_per_command': sum(len(t) for t in self.commands.values()) / len(self.commands) if self.commands else 0,
            'cache_size': len(self._build_cache)
        }


# Фабричная функция для удобного создания
def create_adivinator(commands: Dict[str, List[str]] = None) -> Adivinator:
    """
    Создает и наполняет Adivinator командами.
    """
    adv = Adivinator()
    if commands:
        for name, tokens in commands.items():
            adv.add_command(name, tokens)
    return adv


# Пример использования
if __name__ == "__main__":
    # Тестовые команды
    test_commands = {
        "create_project": ["create", "project"],
        "create_file": ["create", "file"],
        "delete_project": ["delete", "project"],
        "start_server": ["start", "server"],
        "stop_server": ["stop", "server"],
        "show_logs": ["show", "logs"],
        "show_status": ["show", "status"],
        "help": ["help"],
        "list": ["list"],
    }
    
    # Создаем и наполняем Adivinator
    adv = create_adivinator(test_commands)
    
    # Тестируем
    test_inputs = [
        "create",
        "create proj",
        "show",
        "start",
        "unknown command",
        "",
    ]
    
    print("🔮 Adivinator Demo")
    print("=" * 50)
    
    for inp in test_inputs:
        suggestions = adv.suggest(inp, max_suggestions=3)
        print(f"\nInput: '{inp}'")
        if suggestions:
            for i, sug in enumerate(suggestions, 1):
                print(f"  {i}. {sug.command_name} ({sug.confidence:.2f}) - {sug.metadata['match_type']}")
        else:
            print("  (no suggestions)")
    
    # Статистика
    stats = adv.get_stats()
    print(f"\n📊 Stats: {stats}")