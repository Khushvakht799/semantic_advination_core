# storage/trie_storage.py
"""
Модуль для хранения и поиска команд с использованием префиксного дерева (Trie).
Поддерживает автодополнение по префиксам и семантический поиск.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
from datetime import datetime
import heapq


class TrieNode:
    """Узел префиксного дерева (Trie)"""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.command_ids: Set[str] = set()
        self.is_end_of_word: bool = False


class Command:
    """Класс для представления команды"""
    
    def __init__(self, name: str, description: str = "", 
                 tokens: List[str] = None, tags: List[str] = None, 
                 metadata: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.tokens = tokens or []
        self.tags = tags or []
        self.metadata = metadata or {}
        self.usage_count: int = 0
        self.last_used: Optional[datetime] = None
    
    def __repr__(self):
        return f"Command(name='{self.name}', tokens={self.tokens})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование команды в словарь"""
        return {
            'name': self.name,
            'description': self.description,
            'tokens': self.tokens,
            'tags': self.tags,
            'metadata': self.metadata,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Command':
        """Создание команды из словаря"""
        cmd = cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            tokens=data.get('tokens', []),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
        
        cmd.usage_count = data.get('usage_count', 0)
        last_used_str = data.get('last_used')
        if last_used_str:
            cmd.last_used = datetime.fromisoformat(last_used_str)
        
        return cmd
    
    def record_usage(self):
        """Запись использования команды"""
        self.usage_count += 1
        self.last_used = datetime.now()


class CommandTrie:
    """Префиксное дерево для хранения и автодополнения команд"""
    
    def __init__(self):
        self.root = TrieNode()
        self.commands: Dict[str, Command] = {}
        self.token_to_commands: Dict[str, Set[str]] = defaultdict(set)
        self.tag_to_commands: Dict[str, Set[str]] = defaultdict(set)
        self.prefix_cache: Dict[str, List[str]] = {}
    
    def insert(self, command: Command) -> None:
        """
        Вставка команды в Trie
        
        Args:
            command: Команда для вставки
        """
        # Сохраняем команду
        self.commands[command.name] = command
        
        # Вставляем имя команды в Trie для автодополнения
        node = self.root
        for char in command.name.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.command_ids.add(command.name)
        
        # Индексируем по токенам
        for token in command.tokens:
            self.token_to_commands[token.lower()].add(command.name)
            
            # Также вставляем токены в Trie для автодополнения по токенам
            token_node = self.root
            for char in token.lower():
                if char not in token_node.children:
                    token_node.children[char] = TrieNode()
                token_node = token_node.children[char]
            token_node.command_ids.add(command.name)
        
        # Индексируем по тегам
        for tag in command.tags:
            self.tag_to_commands[tag.lower()].add(command.name)
        
        # Очищаем кэш префиксов
        self.prefix_cache.clear()
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[Tuple[Command, float]]:
        """
        Автодополнение по префиксу (основной метод)
        
        Args:
            prefix: Частично введённая строка
            limit: Максимальное количество результатов
            
        Returns:
            Список кортежей (команда, оценка релевантности)
        """
        if not prefix:
            return self._get_popular_commands(limit)
        
        # Проверяем кэш
        cache_key = f"{prefix}_{limit}"
        if cache_key in self.prefix_cache:
            return [(self.commands[name], score) 
                   for name, score in self.prefix_cache[cache_key]]
        
        results = []
        
        # 1. Поиск по именам команд (точный префикс)
        name_results = self._autocomplete_by_name(prefix, limit * 2)
        results.extend(name_results)
        
        # 2. Поиск по токенам (семантическое автодополнение)
        if len(results) < limit:
            token_results = self._autocomplete_by_tokens(prefix, limit * 2)
            # Добавляем с немного меньшим весом
            token_results = [(cmd, score * 0.8) for cmd, score in token_results]
            results.extend(token_results)
        
        # 3. Дедупликация и сортировка
        seen = set()
        unique_results = []
        for cmd, score in results:
            if cmd.name not in seen:
                seen.add(cmd.name)
                unique_results.append((cmd, score))
        
        # 4. Применяем статистику использования
        for i, (cmd, score) in enumerate(unique_results):
            usage_boost = self._calculate_usage_boost(cmd)
            unique_results[i] = (cmd, min(1.0, score + usage_boost))
        
        # 5. Сортируем и ограничиваем
        unique_results.sort(key=lambda x: x[1], reverse=True)
        final_results = unique_results[:limit]
        
        # Сохраняем в кэш
        self.prefix_cache[cache_key] = [(cmd.name, score) for cmd, score in final_results]
        
        return final_results
    
    def _autocomplete_by_name(self, prefix: str, limit: int) -> List[Tuple[Command, float]]:
        """Автодополнение по именам команд"""
        results = []
        
        # Находим узел, соответствующий префиксу
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return results
            node = node.children[char]
        
        # Собираем все команды из поддерева
        stack = [(node, prefix)]
        while stack and len(results) < limit:
            current_node, current_prefix = stack.pop()
            
            # Если это конец слова, добавляем команды
            if current_node.is_end_of_word:
                for cmd_id in current_node.command_ids:
                    cmd = self.commands[cmd_id]
                    # Оценка: чем длиннее совпадение, тем лучше
                    score = len(prefix) / len(cmd.name)
                    results.append((cmd, score))
            
            # Добавляем дочерние узлы
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))
        
        return results
    
    def _autocomplete_by_tokens(self, prefix: str, limit: int) -> List[Tuple[Command, float]]:
        """Автодополнение по токенам команд"""
        results = []
        prefix_lower = prefix.lower()
        
        # Проверяем каждый токен каждой команды
        for cmd in self.commands.values():
            best_score = 0.0
            
            # Проверяем совпадение с токенами
            for token in cmd.tokens:
                token_lower = token.lower()
                
                # Точное начало токена
                if token_lower.startswith(prefix_lower):
                    score = len(prefix) / len(token)
                    best_score = max(best_score, score)
                
                # Частичное вхождение в токен
                elif prefix_lower in token_lower:
                    score = len(prefix) / len(token) * 0.7
                    best_score = max(best_score, score)
            
            # Проверяем совпадение с тегами
            for tag in cmd.tags:
                tag_lower = tag.lower()
                if tag_lower.startswith(prefix_lower):
                    score = len(prefix) / len(tag) * 0.9
                    best_score = max(best_score, score)
            
            if best_score > 0:
                results.append((cmd, best_score))
        
        # Ограничиваем количество результатов
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _get_popular_commands(self, limit: int) -> List[Tuple[Command, float]]:
        """Получение популярных команд (для пустого ввода)"""
        # Сортируем по частоте использования и времени последнего использования
        sorted_commands = sorted(
            self.commands.values(),
            key=lambda cmd: (
                cmd.usage_count,
                cmd.last_used.timestamp() if cmd.last_used else 0
            ),
            reverse=True
        )
        
        # Создаём результаты с оценкой на основе популярности
        results = []
        max_usage = max((cmd.usage_count for cmd in sorted_commands), default=1)
        
        for i, cmd in enumerate(sorted_commands[:limit]):
            # Базовая оценка убывает с позицией в списке
            base_score = 0.7 - (i * 0.1)
            
            # Учитываем частоту использования
            usage_score = (cmd.usage_count / max_usage) * 0.3
            
            score = min(0.9, base_score + usage_score)
            results.append((cmd, score))
        
        return results
    
    def _calculate_usage_boost(self, command: Command) -> float:
        """Вычисление усиления на основе статистики использования"""
        if command.usage_count == 0:
            return 0.0
        
        # Логарифмическое усиление (первые использования дают больше прироста)
        boost = min(0.3, 0.1 * (command.usage_count ** 0.5))
        
        # Учитываем время последнего использования (недавние команды важнее)
        if command.last_used:
            hours_since_use = (datetime.now() - command.last_used).total_seconds() / 3600
            if hours_since_use < 24:  # Использовано менее суток назад
                boost += 0.1
        
        return boost
    
    def record_command_usage(self, command_name: str) -> None:
        """Запись использования команды"""
        if command_name in self.commands:
            self.commands[command_name].record_usage()
            self.prefix_cache.clear()  # Очищаем кэш при изменении статистики
    
    def get_command_completions(self, prefix: str) -> List[str]:
        """
        Получение вариантов автодополнения для введённого префикса
        
        Args:
            prefix: Префикс для автодополнения
            
        Returns:
            Список полных имён команд, начинающихся с префикса
        """
        completions = []
        node = self.root
        
        # Находим узел для префикса
        for char in prefix.lower():
            if char not in node.children:
                return completions
            node = node.children[char]
        
        # Собираем все команды из поддерева
        stack = [(node, prefix)]
        while stack:
            current_node, current_prefix = stack.pop()
            
            if current_node.is_end_of_word:
                completions.extend(current_node.command_ids)
            
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))
        
        return sorted(list(set(completions)))
    
    def find_best_completion(self, prefix: str) -> Optional[Tuple[str, float]]:
        """
        Нахождение лучшего варианта автодополнения
        
        Args:
            prefix: Префикс для автодополнения
            
        Returns:
            Кортеж (имя команды, оценка) или None
        """
        completions = self.autocomplete(prefix, 1)
        if completions:
            cmd, score = completions[0]
            return cmd.name, score
        return None
    
    def search_semantic(self, query: str, limit: int = 5) -> List[Tuple[Command, float]]:
        """
        Семантический поиск команд по запросу
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список кортежей (команда, оценка схожести)
        """
        query_words = query.lower().split()
        if not query_words:
            return []
        
        scored_commands = []
        
        for cmd in self.commands.values():
            score = self._calculate_semantic_score(cmd, query_words)
            if score > 0:
                scored_commands.append((cmd, score))
        
        # Сортируем по убыванию оценки
        scored_commands.sort(key=lambda x: x[1], reverse=True)
        
        # Учитываем статистику использования
        for i, (cmd, score) in enumerate(scored_commands):
            usage_boost = self._calculate_usage_boost(cmd)
            scored_commands[i] = (cmd, min(1.0, score + usage_boost * 0.5))
        
        return scored_commands[:limit]
    
    def _calculate_semantic_score(self, command: Command, query_words: List[str]) -> float:
        """Вычисление семантической схожести"""
        score = 0.0
        
        # Проверяем имя команды
        cmd_name_lower = command.name.lower()
        for word in query_words:
            if word in cmd_name_lower:
                score += 0.3
        
        # Проверяем токены
        token_scores = []
        for token in command.tokens:
            token_lower = token.lower()
            token_score = 0.0
            
            for word in query_words:
                if word in token_lower:
                    # Чем больше слова, тем выше оценка
                    token_score += len(word) / len(token_lower)
            
            if token_score > 0:
                token_scores.append(token_score)
        
        if token_scores:
            score += sum(token_scores) / len(token_scores) * 0.5
        
        # Проверяем теги
        for tag in command.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower:
                    score += 0.2
                    break
        
        # Проверяем описание
        if command.description:
            desc_lower = command.description.lower()
            for word in query_words:
                if word in desc_lower:
                    score += 0.1
        
        return min(1.0, score)
    
    def clear(self) -> None:
        """Очистка всех данных"""
        self.root = TrieNode()
        self.commands.clear()
        self.token_to_commands.clear()
        self.tag_to_commands.clear()
        self.prefix_cache.clear()


class TrieStorage:
    """Хранилище команд с поддержкой сохранения/загрузки"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.trie = CommandTrie()
        self.db_path = db_path
    
    def load_from_json(self, json_path: Union[str, Path]) -> None:
        """Загрузка команд из JSON файла"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for cmd_data in data:
            command = Command.from_dict(cmd_data)
            self.trie.insert(command)
    
    def save_to_json(self, json_path: Union[str, Path]) -> None:
        """Сохранение команд в JSON файл"""
        json_path = Path(json_path)
        
        commands_data = []
        for command in self.trie.commands.values():
            commands_data.append(command.to_dict())
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(commands_data, f, ensure_ascii=False, indent=2)


# Глобальный экземпляр для удобства
_storage_instance: Optional[TrieStorage] = None

def get_storage(db_path: Optional[str] = None) -> TrieStorage:
    """Получение глобального экземпляра хранилища"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = TrieStorage(db_path)
    return _storage_instance
