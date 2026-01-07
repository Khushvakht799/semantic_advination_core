# storage/trie_storage.py
"""
Модуль для хранения и поиска команд с использованием префиксного дерева (Trie).
Поддерживает точный и частичный поиск по токенам, префиксам и тегам.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import math


class TrieNode:
    """Узел префиксного дерева (Trie)"""
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.command_ids: Set[str] = set()
        self.is_end_of_token: bool = False


class Command:
    """Класс для представления команды"""
    
    def __init__(self, name: str, description: str = "", tokens: List[str] = None, 
                 tags: List[str] = None, metadata: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.tokens = tokens or []
        self.tags = tags or []
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Command(name='{self.name}', tokens={self.tokens}, tags={self.tags})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование команды в словарь"""
        return {
            'name': self.name,
            'description': self.description,
            'tokens': self.tokens,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Command':
        """Создание команды из словаря"""
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            tokens=data.get('tokens', []),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )


class CommandTrie:
    """Префиксное дерево для хранения и поиска команд"""
    
    def __init__(self):
        self.root = TrieNode()
        self.commands: Dict[str, Command] = {}
        self.token_to_commands: Dict[str, Set[str]] = {}
        self.tag_to_commands: Dict[str, Set[str]] = {}
    
    def insert(self, command: Command) -> None:
        """
        Вставка команды в Trie
        
        Args:
            command: Команда для вставки
        """
        # Сохраняем команду
        self.commands[command.name] = command
        
        # Индексируем по токенам
        for token in command.tokens:
            # Вставляем в Trie
            node = self.root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_token = True
            node.command_ids.add(command.name)
            
            # Индекс токен->команды
            if token not in self.token_to_commands:
                self.token_to_commands[token] = set()
            self.token_to_commands[token].add(command.name)
        
        # Индексируем по тегам
        for tag in command.tags:
            if tag not in self.tag_to_commands:
                self.tag_to_commands[tag] = set()
            self.tag_to_commands[tag].add(command.name)
    
    def search_exact(self, prefix: str) -> List[Command]:
        """
        Точный поиск команд по префиксу
        
        Args:
            prefix: Префикс для поиска (может быть токеном или частью токена)
            
        Returns:
            Список команд, точно соответствующих префиксу
        """
        # Поиск по токенам
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Если это конец токена, возвращаем команды
        if node.is_end_of_token:
            return [self.commands[cid] for cid in node.command_ids]
        
        # Иначе проверяем все поддеревья
        results = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_end_of_token:
                results.extend([self.commands[cid] for cid in current.command_ids])
            stack.extend(current.children.values())
        
        return list(set(results))  # Убираем дубликаты
    
    def search_by_tokens(self, tokens: List[str]) -> List[Command]:
        """
        Поиск команд по полному соответствию токенов
        
        Args:
            tokens: Список токенов для поиска
            
        Returns:
            Список команд, содержащих все указанные токены
        """
        if not tokens:
            return []
        
        # Находим команды для каждого токена
        command_sets = []
        for token in tokens:
            if token in self.token_to_commands:
                command_sets.append(self.token_to_commands[token])
            else:
                return []  # Если хотя бы один токен не найден
        
        # Находим пересечение
        common_commands = set.intersection(*command_sets)
        return [self.commands[cid] for cid in common_commands]
    
    def search_partial(self, prefix: str, threshold: float = 0.3) -> List[Tuple[Command, float]]:
        """
        Частичный поиск команд по префиксу с оценкой схожести
        
        Args:
            prefix: Префикс для поиска
            threshold: Порог схожести (0-1)
            
        Returns:
            Список кортежей (команда, оценка_схожести)
        """
        results = []
        
        # Разбиваем префикс на слова
        prefix_words = prefix.lower().split()
        
        for cmd in self.commands.values():
            score = self._calculate_similarity_score(cmd, prefix_words, prefix)
            if score >= threshold:
                results.append((cmd, score))
        
        # Сортировка по убыванию оценки
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_similarity_score(self, command: Command, 
                                   prefix_words: List[str], 
                                   full_prefix: str) -> float:
        """
        Вычисление оценки схожести команды с запросом
        
        Args:
            command: Команда для сравнения
            prefix_words: Разбитый на слова запрос
            full_prefix: Полный запрос
            
        Returns:
            Оценка схожести от 0 до 1
        """
        if not prefix_words:
            return 0.0
        
        # 1. Проверка по токенам
        token_scores = []
        for token in command.tokens:
            token_lower = token.lower()
            # Проверяем начало токена
            for word in prefix_words:
                if token_lower.startswith(word):
                    token_scores.append(len(word) / len(token_lower) if token_lower else 0)
                    break
            else:
                # Проверяем частичное вхождение
                for word in prefix_words:
                    if word in token_lower:
                        token_scores.append(len(word) / len(token_lower))
                        break
        
        # 2. Проверка по тегам
        tag_scores = []
        for tag in command.tags:
            tag_lower = tag.lower()
            for word in prefix_words:
                if word in tag_lower:
                    tag_scores.append(len(word) / len(tag_lower) if tag_lower else 0)
                    break
        
        # 3. Проверка по названию команды
        name_score = 0.0
        name_lower = command.name.lower()
        for word in prefix_words:
            if word in name_lower:
                name_score = max(name_score, len(word) / len(name_lower) if name_lower else 0)
        
        # 4. Комбинируем оценки
        scores = []
        if token_scores:
            scores.append(sum(token_scores) / len(token_scores))
        if tag_scores:
            scores.append(sum(tag_scores) / len(tag_scores))
        if name_score > 0:
            scores.append(name_score)
        
        # Если нет совпадений, проверяем полный префикс
        if not scores:
            full_prefix_lower = full_prefix.lower()
            for token in command.tokens + command.tags + [command.name]:
                if full_prefix_lower in token.lower():
                    return len(full_prefix_lower) / len(token) if token else 0
        
        return max(scores) if scores else 0.0
    
    def search_by_tags(self, tags: List[str], require_all: bool = True) -> List[Command]:
        """
        Поиск команд по тегам
        
        Args:
            tags: Список тегов для поиска
            require_all: Если True, команда должна содержать все теги
            
        Returns:
            Список команд, соответствующих тегам
        """
        if not tags:
            return []
        
        if require_all:
            # Команда должна содержать все теги
            command_sets = []
            for tag in tags:
                if tag in self.tag_to_commands:
                    command_sets.append(self.tag_to_commands[tag])
                else:
                    return []
            
            if not command_sets:
                return []
            
            common_commands = set.intersection(*command_sets)
            return [self.commands[cid] for cid in common_commands]
        else:
            # Команда должна содержать хотя бы один тег
            found_commands = set()
            for tag in tags:
                if tag in self.tag_to_commands:
                    found_commands.update(self.tag_to_commands[tag])
            
            return [self.commands[cid] for cid in found_commands]
    
    def get_all_commands(self) -> List[Command]:
        """Получение всех команд"""
        return list(self.commands.values())
    
    def clear(self) -> None:
        """Очистка всех данных"""
        self.root = TrieNode()
        self.commands.clear()
        self.token_to_commands.clear()
        self.tag_to_commands.clear()


class TrieStorage:
    """Главный класс для работы с хранилищем команд"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Инициализация хранилища
        
        Args:
            db_path: Путь к SQLite базе данных (опционально)
        """
        self.trie = CommandTrie()
        self.db_path = db_path
        self.loaded = False
    
    def load_from_json(self, json_path: Union[str, Path]) -> None:
        """
        Загрузка команд из JSON файла
        
        Args:
            json_path: Путь к JSON файлу
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._load_commands(data)
        self.loaded = True
    
    def load_from_sqlite(self, db_path: Optional[str] = None) -> None:
        """
        Загрузка команд из SQLite базы данных
        
        Args:
            db_path: Путь к SQLite базе данных (если None, используется self.db_path)
        """
        db_path = db_path or self.db_path
        if not db_path:
            raise ValueError("Database path not specified")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Проверяем существование таблицы
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='commands'
            """)
            if not cursor.fetchone():
                # Создаем таблицу, если её нет
                cursor.execute("""
                    CREATE TABLE commands (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        tokens TEXT,
                        tags TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
                return
            
            # Загружаем команды
            cursor.execute("SELECT name, description, tokens, tags, metadata FROM commands")
            rows = cursor.fetchall()
            
            commands_data = []
            for row in rows:
                name, description, tokens_str, tags_str, metadata_str = row
                
                # Парсим токены и теги
                tokens = json.loads(tokens_str) if tokens_str else []
                tags = json.loads(tags_str) if tags_str else []
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                commands_data.append({
                    'name': name,
                    'description': description,
                    'tokens': tokens,
                    'tags': tags,
                    'metadata': metadata
                })
            
            self._load_commands(commands_data)
            self.loaded = True
            
        finally:
            conn.close()
    
    def _load_commands(self, commands_data: List[Dict[str, Any]]) -> None:
        """
        Загрузка команд из данных
        
        Args:
            commands_data: Список словарей с данными команд
        """
        self.trie.clear()
        
        for cmd_data in commands_data:
            command = Command.from_dict(cmd_data)
            self.trie.insert(command)
    
    def save_to_sqlite(self, db_path: Optional[str] = None) -> None:
        """
        Сохранение команд в SQLite базу данных
        
        Args:
            db_path: Путь к SQLite базе данных (если None, используется self.db_path)
        """
        db_path = db_path or self.db_path
        if not db_path:
            raise ValueError("Database path not specified")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Создаем таблицу, если её нет
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    tokens TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            # Очищаем таблицу
            cursor.execute("DELETE FROM commands")
            
            # Сохраняем команды
            for command in self.trie.get_all_commands():
                cursor.execute("""
                    INSERT INTO commands (name, description, tokens, tags, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    command.name,
                    command.description,
                    json.dumps(command.tokens, ensure_ascii=False),
                    json.dumps(command.tags, ensure_ascii=False),
                    json.dumps(command.metadata, ensure_ascii=False)
                ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def save_to_json(self, json_path: Union[str, Path]) -> None:
        """
        Сохранение команд в JSON файл
        
        Args:
            json_path: Путь к JSON файлу
        """
        json_path = Path(json_path)
        
        commands_data = []
        for command in self.trie.get_all_commands():
            commands_data.append(command.to_dict())
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(commands_data, f, ensure_ascii=False, indent=2)
    
    def add_command(self, name: str, description: str = "", 
                   tokens: List[str] = None, tags: List[str] = None, 
                   metadata: Dict[str, Any] = None) -> Command:
        """
        Добавление новой команды
        
        Args:
            name: Название команды
            description: Описание команды
            tokens: Токены команды
            tags: Теги команды
            metadata: Дополнительные метаданные
            
        Returns:
            Созданная команда
        """
        command = Command(
            name=name,
            description=description,
            tokens=tokens or [],
            tags=tags or [],
            metadata=metadata or {}
        )
        self.trie.insert(command)
        return command
    
    def remove_command(self, name: str) -> bool:
        """
        Удаление команды по имени
        
        Args:
            name: Название команды
            
        Returns:
            True если команда была удалена, False если команда не найдена
        """
        if name not in self.trie.commands:
            return False
        
        # Получаем команду для получения её токенов и тегов
        command = self.trie.commands[name]
        
        # Удаляем из индексов
        for token in command.tokens:
            if token in self.trie.token_to_commands:
                self.trie.token_to_commands[token].discard(name)
                if not self.trie.token_to_commands[token]:
                    del self.trie.token_to_commands[token]
        
        for tag in command.tags:
            if tag in self.trie.tag_to_commands:
                self.trie.tag_to_commands[tag].discard(name)
                if not self.trie.tag_to_commands[tag]:
                    del self.trie.tag_to_commands[tag]
        
        # Удаляем из Trie
        # Для простоты перестраиваем Trie без этой команды
        # В реальной реализации нужно более эффективное удаление
        commands = list(self.trie.commands.values())
        self.trie.clear()
        for cmd in commands:
            if cmd.name != name:
                self.trie.insert(cmd)
        
        return True
    
    def search(self, query: str, search_type: str = "partial", 
               threshold: float = 0.3) -> List[Tuple[Command, float]]:
        """
        Универсальный поиск команд
        
        Args:
            query: Поисковый запрос
            search_type: Тип поиска ("exact", "partial", "tokens", "tags")
            threshold: Порог схожести (для partial поиска)
            
        Returns:
            Список кортежей (команда, оценка_схожести)
        """
        if not self.loaded:
            raise RuntimeError("Storage not loaded. Call load_from_json() or load_from_sqlite() first.")
        
        query = query.strip()
        if not query:
            return [(cmd, 1.0) for cmd in self.trie.get_all_commands()]
        
        if search_type == "exact":
            commands = self.trie.search_exact(query)
            return [(cmd, 1.0) for cmd in commands]
        
        elif search_type == "tokens":
            tokens = query.split()
            commands = self.trie.search_by_tokens(tokens)
            return [(cmd, 1.0) for cmd in commands]
        
        elif search_type == "tags":
            tags = [tag.strip() for tag in query.split(",")]
            commands = self.trie.search_by_tags(tags, require_all=False)
            return [(cmd, 1.0) for cmd in commands]
        
        else:  # partial
            return self.trie.search_partial(query, threshold)
    
    def get_command(self, name: str) -> Optional[Command]:
        """Получение команды по имени"""
        return self.trie.commands.get(name)
    
    def command_exists(self, name: str) -> bool:
        """Проверка существования команды"""
        return name in self.trie.commands
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики хранилища"""
        return {
            'total_commands': len(self.trie.commands),
            'total_tokens': len(self.trie.token_to_commands),
            'total_tags': len(self.trie.tag_to_commands),
            'loaded': self.loaded
        }


# Создание глобального экземпляра хранилища для удобства
_storage_instance: Optional[TrieStorage] = None


def get_storage(db_path: Optional[str] = None) -> TrieStorage:
    """
    Получение глобального экземпляра хранилища
    
    Args:
        db_path: Путь к SQLite базе данных (опционально)
        
    Returns:
        Экземпляр TrieStorage
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = TrieStorage(db_path)
    return _storage_instance


def load_default_storage(json_path: Optional[Union[str, Path]] = None, 
                        db_path: Optional[str] = None) -> TrieStorage:
    """
    Загрузка хранилища по умолчанию
    
    Args:
        json_path: Путь к JSON файлу (приоритет)
        db_path: Путь к SQLite базе данных
        
    Returns:
        Загруженный экземпляр TrieStorage
    """
    storage = get_storage(db_path)
    
    if json_path and Path(json_path).exists():
        storage.load_from_json(json_path)
    elif db_path:
        storage.load_from_sqlite(db_path)
    
    return storage
