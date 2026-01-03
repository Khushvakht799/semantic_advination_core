from typing import List, Dict, Any, Optional, Tuple
import re
import sqlite3
import json
from pathlib import Path
from datetime import datetime

from .models import *

class SemanticCommandSystem:
    """
    Семантико-композиционная система команд.
    Угадывает существующие или создаёт новые через диалог.
    """
    
    def __init__(self, db_path: str = "commands.db"):
        self.db_path = db_path
        self._init_database()
        self.active_dialogs: Dict[str, DialogContext] = {}
        
        # Шаблоны для генерации команд по категориям
        self.command_templates = {
            "shell_find": {
                "template": "find {path} -name '{pattern}' {options}",
                "params": ["path", "pattern", "options"],
                "questions": [
                    ClarificationQuestion(
                        question_id="path",
                        text="В какой папке искать?",
                        question_type="text",
                        context_key="path",
                        required=True
                    ),
                    ClarificationQuestion(
                        question_id="pattern", 
                        text="Какой шаблон имени файлов? (например, *.txt)",
                        question_type="text",
                        context_key="pattern",
                        required=True
                    )
                ]
            },
            "sql_select": {
                "template": "SELECT {columns} FROM {table} WHERE {conditions}",
                "params": ["columns", "table", "conditions"],
                "questions": [
                    # ... вопросы для SQL
                ]
            }
        }
    
    def _init_database(self):
        """Инициализация БД команд"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS commands (
                    command_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    tags TEXT,  -- JSON список тегов
                    parameters TEXT,  -- JSON словарь параметров
                    usage_count INTEGER DEFAULT 0,
                    status TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_id TEXT,
                    used_at TIMESTAMP,
                    context TEXT,  -- JSON контекста использования
                    FOREIGN KEY (command_id) REFERENCES commands(command_id)
                )
            """)
    
    # --- ФАЗА 1: Поиск в БД ---
    
    def find_commands(self, prefix: str, context_tags: List[str] = None, limit: int = 5) -> List[Command]:
        """
        Ищет команды в БД по префиксу и семантическим тегам.
        Возвращает отсортированные по релевантности.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Поиск по префиксу
            cursor.execute(
                "SELECT * FROM commands WHERE text LIKE ? AND status = 'confirmed'",
                (f"{prefix}%",)
            )
            
            commands = []
            for row in cursor.fetchall():
                # Преобразуем JSON поля
                tags = json.loads(row['tags']) if row['tags'] else []
                parameters = json.loads(row['parameters']) if row['parameters'] else {}
                
                cmd = Command(
                    command_id=row['command_id'],
                    text=row['text'],
                    description=row['description'],
                    category=row['category'],
                    tags=tags,
                    parameters=parameters,
                    usage_count=row['usage_count'],
                    status=CommandStatus(row['status'])
                )
                cmd.last_used = datetime.fromisoformat(row['last_used']) if row['last_used'] else None
                
                # Вычисляем score релевантности
                cmd._match_score = cmd.match_score(prefix, context_tags or [])
                commands.append(cmd)
            
            # Сортируем по убыванию релевантности
            commands.sort(key=lambda c: c._match_score, reverse=True)
            return commands[:limit]
    
    # --- ФАЗА 2: Диалог композиции ---
    
    def start_composition_dialog(self, prefix: str, user_intent: str = "") -> DialogContext:
        """
        Начинает диалог композиции новой команды.
        Возвращает первый вопрос для уточнения.
        """
        dialog = DialogContext(
            user_intent=user_intent or prefix
        )
        
        # Определяем категорию по префиксу/интенту
        category = self._detect_category(prefix, user_intent)
        dialog.collected_answers['category'] = category
        
        # Выбираем цепочку вопросов для этой категории
        if category in ["shell", "file_operations"]:
            template_key = "shell_find"
        elif category in ["sql", "database"]:
            template_key = "sql_select"
        else:
            template_key = "generic"
        
        # Загружаем вопросы из шаблона или генерируем общие
        if template_key in self.command_templates:
            dialog.questions_chain = self.command_templates[template_key]["questions"]
        else:
            dialog.questions_chain = self._generate_generic_questions(user_intent)
        
        self.active_dialogs[dialog.dialog_id] = dialog
        dialog.state = DialogState.CLARIFY_INTENT
        
        return dialog
    
    def process_dialog_answer(self, dialog_id: str, answer: Any) -> Tuple[DialogContext, Optional[ClarificationQuestion]]:
        """
        Обрабатывает ответ пользователя в диалоге.
        Возвращает обновлённый контекст и следующий вопрос (или None если диалог завершён).
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            raise ValueError(f"Диалог {dialog_id} не найден")
        
        # Сохраняем ответ
        if dialog.current_question_idx < len(dialog.questions_chain):
            current_q = dialog.questions_chain[dialog.current_question_idx]
            dialog.collected_answers[current_q.context_key] = answer
        
        # Переходим к следующему вопросу
        dialog.current_question_idx += 1
        
        if dialog.current_question_idx >= len(dialog.questions_chain):
            # Все вопросы получены → генерируем команду
            dialog.generated_command = self._generate_command_from_answers(dialog)
            dialog.state = DialogState.CONFIRMATION
            
            # Возвращаем None для вопроса (вместо вопроса - предложение сгенерированной команды)
            return dialog, None
        else:
            # Возвращаем следующий вопрос
            next_q = dialog.questions_chain[dialog.current_question_idx]
            return dialog, next_q
    
    def _detect_category(self, prefix: str, intent: str) -> str:
        """Определяет категорию команды по префиксу и намерению"""
        intent_lower = intent.lower()
        
        if any(word in intent_lower for word in ["найди", "поиск", "find", "search"]):
            return "file_operations"
        elif any(word in intent_lower for word in ["база", "данн", "sql", "select"]):
            return "database"
        elif prefix.startswith("curl") or "api" in intent_lower:
            return "api"
        elif prefix.startswith("kubectl") or "kubernetes"