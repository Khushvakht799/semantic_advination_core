# storage\command_db.py

import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

class CommandDatabase:
    """
    База данных для хранения команд, их метаданных и истории использования.
    Использует SQLite.
    """

    def __init__(self, db_path: str = "data/commands.db"):
        """
        Инициализация базы данных.
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_db()

    def _ensure_data_dir(self):
        """Создаёт директорию data, если её нет."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Инициализирует таблицы базы данных."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица команд
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    tokens TEXT NOT NULL,  -- JSON массив токенов
                    description TEXT,
                    action_type TEXT DEFAULT 'function',
                    action_data TEXT,      -- JSON с данными для выполнения
                    category TEXT,
                    tags TEXT,             -- JSON массив тегов
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(name)
                )
            ''')
            
            # Таблица истории выполнения
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_id INTEGER,
                    input_text TEXT,
                    parameters TEXT,       -- JSON параметров
                    result TEXT,
                    success BOOLEAN,
                    execution_time_ms INTEGER,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    FOREIGN KEY (command_id) REFERENCES commands (id)
                )
            ''')
            
            # Таблица синонимов/алиасов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_id INTEGER,
                    alias TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (command_id) REFERENCES commands (id),
                    UNIQUE(alias)
                )
            ''')
            
            # Индексы для ускорения поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_name ON commands (name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_category ON commands (category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_command_id ON execution_history (command_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_executed_at ON execution_history (executed_at)')
            
            conn.commit()

    def add_command(self, 
                   name: str, 
                   tokens: List[str],
                   description: str = "",
                   action_type: str = "function",
                   action_data: Dict[str, Any] = None,
                   category: str = None,
                   tags: List[str] = None) -> int:
        """
        Добавляет новую команду в базу данных.
        
        Returns:
            ID добавленной команды
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO commands 
                (name, tokens, description, action_type, action_data, category, tags, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name,
                json.dumps(tokens),
                description,
                action_type,
                json.dumps(action_data or {}),
                category,
                json.dumps(tags or []),
                datetime.now().isoformat()
            ))
            
            command_id = cursor.lastrowid
            conn.commit()
            
            return command_id

    def get_command(self, command_id: int = None, name: str = None) -> Optional[Dict[str, Any]]:
        """
        Получает команду по ID или имени.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if command_id:
                cursor.execute('SELECT * FROM commands WHERE id = ?', (command_id,))
            elif name:
                cursor.execute('SELECT * FROM commands WHERE name = ?', (name,))
            else:
                return None
            
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def search_commands(self, 
                       query: str = None, 
                       category: str = None,
                       tags: List[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """
        Поиск команд по различным критериям.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            sql = "SELECT * FROM commands WHERE is_active = 1"
            params = []
            
            if query:
                sql += " AND (name LIKE ? OR description LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            if category:
                sql += " AND category = ?"
                params.append(category)
            
            if tags:
                # Поиск по тегам (JSON массив)
                for tag in tags:
                    sql += f" AND tags LIKE ?"
                    params.append(f'%"{tag}"%')
            
            sql += " ORDER BY usage_count DESC, name ASC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]

    def log_execution(self, 
                     command_id: int, 
                     input_text: str,
                     parameters: Dict[str, Any] = None,
                     result: str = None,
                     success: bool = True,
                     execution_time_ms: int = 0,
                     user_id: str = None):
        """
        Логирует выполнение команды.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Логируем выполнение
            cursor.execute('''
                INSERT INTO execution_history 
                (command_id, input_text, parameters, result, success, execution_time_ms, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                command_id,
                input_text,
                json.dumps(parameters or {}),
                result,
                success,
                execution_time_ms,
                user_id
            ))
            
            # Увеличиваем счётчик использования
            cursor.execute('''
                UPDATE commands 
                SET usage_count = usage_count + 1, updated_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), command_id))
            
            conn.commit()

    def add_alias(self, command_id: int, alias: str):
        """
        Добавляет алиас (синоним) для команды.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO aliases (command_id, alias)
                VALUES (?, ?)
            ''', (command_id, alias))
            conn.commit()

    def get_command_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """
        Находит команду по её алиасу.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.* FROM commands c
                JOIN aliases a ON c.id = a.command_id
                WHERE a.alias = ? AND c.is_active = 1
            ''', (alias,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def get_popular_commands(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Возвращает самые популярные команды (по usage_count).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM commands 
                WHERE is_active = 1 
                ORDER BY usage_count DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def delete_command(self, command_id: int, soft_delete: bool = True):
        """
        Удаляет команду (мягко или физически).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if soft_delete:
                cursor.execute('UPDATE commands SET is_active = 0 WHERE id = ?', (command_id,))
            else:
                cursor.execute('DELETE FROM commands WHERE id = ?', (command_id,))
                cursor.execute('DELETE FROM aliases WHERE command_id = ?', (command_id,))
            
            conn.commit()

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Конвертирует строку SQLite в словарь Python."""
        d = dict(row)
        
        # Парсим JSON поля
        json_fields = ['tokens', 'action_data', 'tags']
        for field in json_fields:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except:
                    d[field] = []
        
        return d

    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по базе данных.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Общее количество команд
            cursor.execute('SELECT COUNT(*) FROM commands WHERE is_active = 1')
            stats['total_commands'] = cursor.fetchone()[0]
            
            # Количество категорий
            cursor.execute('SELECT COUNT(DISTINCT category) FROM commands WHERE is_active = 1')
            stats['total_categories'] = cursor.fetchone()[0]
            
            # Общее количество выполнений
            cursor.execute('SELECT COUNT(*) FROM execution_history')
            stats['total_executions'] = cursor.fetchone()[0]
            
            # Самая популярная команда
            cursor.execute('SELECT name, usage_count FROM commands ORDER BY usage_count DESC LIMIT 1')
            row = cursor.fetchone()
            if row:
                stats['most_popular_command'] = {'name': row[0], 'usage_count': row[1]}
            
            # Количество выполнений за последние 7 дней
            cursor.execute('''
                SELECT COUNT(*) FROM execution_history 
                WHERE executed_at >= datetime('now', '-7 days')
            ''')
            stats['executions_last_7_days'] = cursor.fetchone()[0]
            
            return stats


# Пример использования
if __name__ == "__main__":
    # Инициализация базы данных
    db = CommandDatabase()
    
    # Добавление тестовой команды
    command_id = db.add_command(
        name="create_project",
        tokens=["create", "project"],
        description="Создает новый проект",
        action_type="shell",
        action_data={"command": "mkdir {project_name}"},
        category="project_management",
        tags=["project", "create", "new"]
    )
    
    print(f"Добавлена команда с ID: {command_id}")
    
    # Поиск команд
    commands = db.search_commands(category="project_management")
    print(f"Найдено команд: {len(commands)}")
    
    # Получение статистики
    stats = db.get_statistics()
    print(f"Статистика: {stats}")