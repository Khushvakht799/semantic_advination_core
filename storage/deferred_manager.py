# storage\deferred_manager.py
"""
Менеджер отложенного выполнения команд.
Позволяет планировать выполнение команд на будущее время.
"""

import json
import threading
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from pathlib import Path


class TaskStatus(Enum):
    """Статусы отложенных задач."""
    PENDING = "pending"          # Ожидает выполнения
    SCHEDULED = "scheduled"     # Запланирована
    PROCESSING = "processing"   # Выполняется
    COMPLETED = "completed"     # Завершена успешно
    FAILED = "failed"           # Завершена с ошибкой
    CANCELLED = "cancelled"     # Отменена


class TaskPriority(Enum):
    """Приоритеты задач."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class DeferredTask:
    """Модель отложенной задачи."""
    
    id: Optional[int] = None
    task_id: str = ""                      # Уникальный ID задачи
    command_name: str = ""                 # Имя команды для выполнения
    command_data: Dict[str, Any] = None    # Данные команды (JSON)
    parameters: Dict[str, Any] = None      # Параметры выполнения
    scheduled_time: datetime = None        # Время запланированного выполнения
    created_at: datetime = None            # Время создания
    updated_at: datetime = None            # Время последнего обновления
    status: TaskStatus = TaskStatus.PENDING # Статус задачи
    priority: TaskPriority = TaskPriority.NORMAL # Приоритет
    retry_count: int = 0                   # Количество попыток
    max_retries: int = 3                   # Максимальное количество попыток
    result: str = ""                       # Результат выполнения
    error_message: str = ""                # Сообщение об ошибке
    created_by: str = "system"             # Кто создал задачу
    metadata: Dict[str, Any] = None        # Дополнительные метаданные
    
    def __post_init__(self):
        """Инициализация значений по умолчанию."""
        if self.command_data is None:
            self.command_data = {}
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}
        if self.scheduled_time is None:
            self.scheduled_time = datetime.now() + timedelta(minutes=5)
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if not self.task_id:
            self.task_id = self._generate_task_id()
    
    def _generate_task_id(self) -> str:
        """Генерирует уникальный ID задачи."""
        import uuid
        prefix = self.command_name.replace(' ', '_').lower()[:20]
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует задачу в словарь."""
        result = asdict(self)
        
        # Конвертируем специальные типы
        result['status'] = self.status.value
        result['priority'] = self.priority.value
        
        # Конвертируем datetime в строки ISO формата
        for field in ['scheduled_time', 'created_at', 'updated_at']:
            if result[field]:
                result[field] = result[field].isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeferredTask':
        """Создает задачу из словаря."""
        # Конвертируем строки обратно в datetime
        datetime_fields = ['scheduled_time', 'created_at', 'updated_at']
        for field in datetime_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        # Конвертируем строки обратно в Enum
        if 'status' in data and data['status']:
            data['status'] = TaskStatus(data['status'])
        if 'priority' in data and data['priority']:
            data['priority'] = TaskPriority(data['priority'])
        
        return cls(**data)


class DeferredTaskManager:
    """
    Менеджер для работы с отложенными задачами.
    """
    
    def __init__(self, db_path: str = "data/deferred_tasks.db"):
        """
        Инициализация менеджера.
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_db()
        
        self.running = False
        self.processor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Колбэки для обработки задач
        self.task_handlers: Dict[str, Callable] = {}
        self.default_handler = None
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
    
    def _ensure_data_dir(self):
        """Создаёт директорию data, если её нет."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Инициализирует таблицы базы данных."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deferred_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    command_name TEXT NOT NULL,
                    command_data TEXT,           -- JSON данные команды
                    parameters TEXT,             -- JSON параметры
                    scheduled_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 1,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    result TEXT,
                    error_message TEXT,
                    created_by TEXT DEFAULT 'system',
                    metadata TEXT,                -- JSON метаданные
                    CONSTRAINT unique_task_id UNIQUE(task_id)
                )
            ''')
            
            # Индексы для ускорения поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_status ON deferred_tasks (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_scheduled ON deferred_tasks (scheduled_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_priority ON deferred_tasks (priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_command ON deferred_tasks (command_name)')
            
            conn.commit()
    
    def register_handler(self, command_name: str, handler: Callable):
        """
        Регистрирует обработчик для конкретной команды.
        
        Args:
            command_name: Имя команды
            handler: Функция-обработчик (принимает DeferredTask, возвращает результат)
        """
        self.task_handlers[command_name] = handler
    
    def set_default_handler(self, handler: Callable):
        """Устанавливает обработчик по умолчанию."""
        self.default_handler = handler
    
    def schedule_task(self, task: DeferredTask) -> str:
        """
        Планирует новую отложенную задачу.
        
        Returns:
            task_id: Уникальный ID созданной задачи
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Проверяем, не существует ли уже задача с таким ID
                cursor.execute('SELECT COUNT(*) FROM deferred_tasks WHERE task_id = ?', (task.task_id,))
                if cursor.fetchone()[0] > 0:
                    # Генерируем новый ID
                    task.task_id = task._generate_task_id()
                
                # Вставляем задачу
                cursor.execute('''
                    INSERT INTO deferred_tasks 
                    (task_id, command_name, command_data, parameters, scheduled_time, 
                     status, priority, max_retries, created_by, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task.task_id,
                    task.command_name,
                    json.dumps(task.command_data),
                    json.dumps(task.parameters),
                    task.scheduled_time.isoformat(),
                    task.status.value,
                    task.priority.value,
                    task.max_retries,
                    task.created_by,
                    json.dumps(task.metadata)
                ))
                
                task.id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Запланирована задача {task.task_id} на {task.scheduled_time}")
                return task.task_id
    
    def create_task(self, 
                   command_name: str,
                   command_data: Dict[str, Any] = None,
                   parameters: Dict[str, Any] = None,
                   delay_seconds: int = 300,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   created_by: str = "system",
                   metadata: Dict[str, Any] = None) -> str:
        """
        Создает и планирует задачу с удобным интерфейсом.
        
        Args:
            command_name: Имя команды для выполнения
            command_data: Данные команды
            parameters: Параметры выполнения
            delay_seconds: Задержка в секундах до выполнения
            priority: Приоритет задачи
            created_by: Кто создал задачу
            metadata: Дополнительные метаданные
            
        Returns:
            task_id: Уникальный ID созданной задачи
        """
        scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)
        
        task = DeferredTask(
            command_name=command_name,
            command_data=command_data or {},
            parameters=parameters or {},
            scheduled_time=scheduled_time,
            priority=priority,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        return self.schedule_task(task)
    
    def get_task(self, task_id: str) -> Optional[DeferredTask]:
        """Получает задачу по ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM deferred_tasks WHERE task_id = ?', (task_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_task(row)
            return None
    
    def get_pending_tasks(self, limit: int = 100) -> List[DeferredTask]:
        """Получает задачи, готовые к выполнению."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                SELECT * FROM deferred_tasks 
                WHERE status IN ('pending', 'scheduled') 
                  AND scheduled_time <= ?
                ORDER BY priority DESC, scheduled_time ASC
                LIMIT ?
            ''', (now, limit))
            
            rows = cursor.fetchall()
            return [self._row_to_task(row) for row in rows]
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: str = "", error_message: str = ""):
        """Обновляет статус задачи."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE deferred_tasks 
                    SET status = ?, 
                        result = ?, 
                        error_message = ?, 
                        updated_at = ?,
                        retry_count = retry_count + 1
                    WHERE task_id = ?
                ''', (
                    status.value,
                    result,
                    error_message,
                    datetime.now().isoformat(),
                    task_id
                ))
                
                conn.commit()
    
    def cancel_task(self, task_id: str) -> bool:
        """Отменяет задачу."""
        task = self.get_task(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
            self.update_task_status(task_id, TaskStatus.CANCELLED, "Задача отменена")
            self.logger.info(f"Задача {task_id} отменена")
            return True
        return False
    
    def reschedule_task(self, task_id: str, delay_seconds: int) -> bool:
        """Переносит задачу на более позднее время."""
        task = self.get_task(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
            new_time = datetime.now() + timedelta(seconds=delay_seconds)
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        UPDATE deferred_tasks 
                        SET scheduled_time = ?, 
                            updated_at = ?,
                            retry_count = retry_count + 1
                        WHERE task_id = ?
                    ''', (
                        new_time.isoformat(),
                        datetime.now().isoformat(),
                        task_id
                    ))
                    
                    conn.commit()
            
            self.logger.info(f"Задача {task_id} перенесена на {new_time}")
            return True
        return False
    
    def _execute_task(self, task: DeferredTask) -> bool:
        """Выполняет задачу."""
        try:
            self.logger.info(f"Выполняется задача {task.task_id}: {task.command_name}")
            
            # Обновляем статус на "выполняется"
            self.update_task_status(task.task_id, TaskStatus.PROCESSING)
            
            # Ищем обработчик
            handler = self.task_handlers.get(task.command_name, self.default_handler)
            
            if not handler:
                raise ValueError(f"Нет обработчика для команды: {task.command_name}")
            
            # Выполняем задачу
            result = handler(task)
            
            # Обновляем статус на "завершено"
            self.update_task_status(
                task.task_id, 
                TaskStatus.COMPLETED, 
                result=str(result)
            )
            
            self.logger.info(f"Задача {task.task_id} выполнена успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка выполнения задачи {task.task_id}: {e}")
            
            # Проверяем, нужно ли повторять
            if task.retry_count < task.max_retries:
                # Переносим задачу на более позднее время
                delay = 60 * (2 ** task.retry_count)  # Экспоненциальная задержка
                self.reschedule_task(task.task_id, delay)
                self.logger.info(f"Задача {task.task_id} запланирована на повтор через {delay} сек")
            else:
                # Превышено максимальное количество попыток
                self.update_task_status(
                    task.task_id, 
                    TaskStatus.FAILED, 
                    error_message=str(e)
                )
            
            return False
    
    def _process_pending_tasks(self):
        """Фоновый процессор задач."""
        self.logger.info("Запущен процессор отложенных задач")
        
        while self.running:
            try:
                # Получаем задачи, готовые к выполнению
                pending_tasks = self.get_pending_tasks(limit=10)
                
                for task in pending_tasks:
                    if not self.running:
                        break
                    
                    self._execute_task(task)
                
                # Пауза между проверками
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Ошибка в процессоре задач: {e}")
                time.sleep(5)
    
    def start_processor(self):
        """Запускает фоновый процессор задач."""
        if self.running:
            return
        
        self.running = True
        self.processor_thread = threading.Thread(
            target=self._process_pending_tasks,
            daemon=True,
            name="DeferredTaskProcessor"
        )
        self.processor_thread.start()
        self.logger.info("Процессор отложенных задач запущен")
    
    def stop_processor(self):
        """Останавливает фоновый процессор задач."""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
        self.logger.info("Процессор отложенных задач остановлен")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по задачам."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Общее количество задач
            cursor.execute('SELECT COUNT(*) FROM deferred_tasks')
            stats['total_tasks'] = cursor.fetchone()[0]
            
            # Количество по статусам
            cursor.execute('SELECT status, COUNT(*) FROM deferred_tasks GROUP BY status')
            stats['by_status'] = dict(cursor.fetchall())
            
            # Количество по приоритетам
            cursor.execute('SELECT priority, COUNT(*) FROM deferred_tasks GROUP BY priority')
            stats['by_priority'] = dict(cursor.fetchall())
            
            # Количество задач, созданных сегодня
            today = datetime.now().date().isoformat()
            cursor.execute('SELECT COUNT(*) FROM deferred_tasks WHERE DATE(created_at) = ?', (today,))
            stats['created_today'] = cursor.fetchone()[0]
            
            # Средняя задержка выполнения
            cursor.execute('''
                SELECT AVG((julianday(updated_at) - julianday(created_at)) * 86400)
                FROM deferred_tasks 
                WHERE status = 'completed'
            ''')
            avg_delay = cursor.fetchone()[0]
            stats['avg_completion_time_seconds'] = round(avg_delay or 0, 2)
            
            return stats
    
    def _row_to_task(self, row) -> DeferredTask:
        """Конвертирует строку SQLite в DeferredTask."""
        data = dict(row)
        
        # Парсим JSON поля
        json_fields = ['command_data', 'parameters', 'metadata']
        for field in json_fields:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except:
                    data[field] = {}
        
        # Конвертируем строки в Enum
        if data.get('status'):
            data['status'] = TaskStatus(data['status'])
        if data.get('priority'):
            try:
                data['priority'] = TaskPriority(data['priority'])
            except:
                data['priority'] = TaskPriority.NORMAL
        
        return DeferredTask.from_dict(data)


# Пример использования
if __name__ == "__main__":
    import logging
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем менеджер
    manager = DeferredTaskManager()
    
    # Пример обработчика
    def example_handler(task: DeferredTask):
        print(f"Выполняется команда: {task.command_name}")
        print(f"Параметры: {task.parameters}")
        return f"Результат выполнения {task.command_name}"
    
    # Регистрируем обработчик
    manager.register_handler("example_command", example_handler)
    
    # Создаем тестовую задачу
    task_id = manager.create_task(
        command_name="example_command",
        parameters={"param1": "value1", "param2": 123},
        delay_seconds=10,  # Выполнить через 10 секунд
        priority=TaskPriority.NORMAL,
        created_by="test_user",
        metadata={"test": True}
    )
    
    print(f"Создана задача с ID: {task_id}")
    
    # Запускаем процессор
    manager.start_processor()
    
    print("Процессор запущен. Ожидаем 15 секунд...")
    time.sleep(15)
    
    # Останавливаем процессор
    manager.stop_processor()
    
    # Получаем статистику
    stats = manager.get_statistics()
    print(f"Статистика: {stats}")