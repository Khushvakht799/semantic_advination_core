# storage\history_logger.py
"""
Логирование истории выполнения команд.
Записывает все выполненные команды с их параметрами, результатами и метаданными.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import csv


class ExecutionStatus(Enum):
    """Статусы выполнения команд."""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    AUTHORIZATION_ERROR = "authorization_error"


class LogLevel(Enum):
    """Уровни детализации логирования."""
    MINIMAL = "minimal"      # Только основные данные
    STANDARD = "standard"    # Стандартный набор данных
    VERBOSE = "verbose"      # Все возможные данные
    DEBUG = "debug"          + Отладочная информация


@dataclass
class ExecutionRecord:
    """Запись о выполнении команды."""
    
    id: Optional[int] = None
    execution_id: str = ""                    # Уникальный ID выполнения
    command_id: Optional[int] = None          # ID команды (если известен)
    command_name: str = ""                    # Имя команды
    input_text: str = ""                      # Исходный текст команды
    tokens: List[str] = None                  # Токенизированная команда
    parameters: Dict[str, Any] = None         # Параметры выполнения
    result: str = ""                          # Результат выполнения
    status: ExecutionStatus = ExecutionStatus.SUCCESS # Статус выполнения
    error_message: str = ""                   # Сообщение об ошибке
    start_time: datetime = None               # Время начала выполнения
    end_time: datetime = None                 # Время окончания выполнения
    execution_time_ms: int = 0                # Время выполнения в миллисекундах
    user_id: str = ""                         # ID пользователя
    user_role: str = "user"                   # Роль пользователя
    session_id: str = ""                      # ID сессии
    adapter_type: str = "unknown"             # Тип адаптера (CLI, API, WebSocket)
    confidence_score: float = 1.0             # Уверенность системы в команде
    metadata: Dict[str, Any] = None           # Дополнительные метаданные
    ip_address: str = ""                      # IP адрес клиента
    user_agent: str = ""                      # User-Agent клиента
    
    def __post_init__(self):
        """Инициализация значений по умолчанию."""
        if self.tokens is None:
            self.tokens = []
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.end_time is None:
            self.end_time = datetime.now()
        if not self.execution_id:
            self.execution_id = self._generate_execution_id()
    
    def _generate_execution_id(self) -> str:
        """Генерирует уникальный ID выполнения."""
        import uuid
        prefix = self.command_name.replace(' ', '_').lower()[:15]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = uuid.uuid4().hex[:6]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def calculate_execution_time(self):
        """Рассчитывает время выполнения."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.execution_time_ms = int(delta.total_seconds() * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует запись в словарь."""
        result = asdict(self)
        
        # Конвертируем специальные типы
        result['status'] = self.status.value
        
        # Конвертируем datetime в строки ISO формата
        for field in ['start_time', 'end_time']:
            if result[field]:
                result[field] = result[field].isoformat()
        
        # Удаляем None поля для чистоты
        result = {k: v for k, v in result.items() if v is not None}
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionRecord':
        """Создает запись из словаря."""
        # Конвертируем строки обратно в datetime
        datetime_fields = ['start_time', 'end_time']
        for field in datetime_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        # Конвертируем строки обратно в Enum
        if 'status' in data and data['status']:
            data['status'] = ExecutionStatus(data['status'])
        
        return cls(**data)


class HistoryLogger:
    """
    Логировщик истории выполнения команд.
    """
    
    def __init__(self, 
                 db_path: str = "data/execution_history.db",
                 log_level: LogLevel = LogLevel.STANDARD,
                 max_records: int = 100000,
                 retention_days: int = 90):
        """
        Инициализация логировщика.
        
        Args:
            db_path: Путь к файлу базы данных SQLite
            log_level: Уровень детализации логирования
            max_records: Максимальное количество записей (старые удаляются)
            retention_days: Количество дней хранения записей
        """
        self.db_path = db_path
        self.log_level = log_level
        self.max_records = max_records
        self.retention_days = retention_days
        
        self._ensure_data_dir()
        self._init_db()
        self._cleanup_old_records()
        
        self.logger = logging.getLogger(__name__)
    
    def _ensure_data_dir(self):
        """Создаёт директорию data, если её нет."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Инициализирует таблицы базы данных."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Основная таблица истории выполнения
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL UNIQUE,
                    command_id INTEGER,
                    command_name TEXT NOT NULL,
                    input_text TEXT,
                    tokens TEXT,                 -- JSON массив токенов
                    parameters TEXT,             -- JSON параметров
                    result TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    user_id TEXT,
                    user_role TEXT,
                    session_id TEXT,
                    adapter_type TEXT,
                    confidence_score REAL,
                    metadata TEXT,               -- JSON метаданных
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица статистики выполнения (агрегированные данные)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    execution_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    total_execution_time_ms INTEGER DEFAULT 0,
                    avg_execution_time_ms REAL DEFAULT 0,
                    unique_users INTEGER DEFAULT 0,
                    UNIQUE(command_name, date)
                )
            ''')
            
            # Таблица ошибок для анализа
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    command_name TEXT,
                    user_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0
                )
            ''')
            
            # Индексы для ускорения поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_command ON execution_history (command_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_user ON execution_history (user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_time ON execution_history (start_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_status ON execution_history (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stats_date ON execution_stats (date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_type ON execution_errors (error_type)')
            
            conn.commit()
    
    def log_execution(self, record: ExecutionRecord) -> str:
        """
        Логирует выполнение команды.
        
        Args:
            record: Запись о выполнении команды
            
        Returns:
            execution_id: Уникальный ID выполнения
        """
        # Рассчитываем время выполнения
        record.calculate_execution_time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Вставляем запись в историю
            cursor.execute('''
                INSERT INTO execution_history 
                (execution_id, command_id, command_name, input_text, tokens,
                 parameters, result, status, error_message, start_time, end_time,
                 execution_time_ms, user_id, user_role, session_id, adapter_type,
                 confidence_score, metadata, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.execution_id,
                record.command_id,
                record.command_name,
                record.input_text,
                json.dumps(record.tokens),
                json.dumps(record.parameters),
                record.result,
                record.status.value,
                record.error_message,
                record.start_time.isoformat(),
                record.end_time.isoformat(),
                record.execution_time_ms,
                record.user_id,
                record.user_role,
                record.session_id,
                record.adapter_type,
                record.confidence_score,
                json.dumps(record.metadata),
                record.ip_address,
                record.user_agent
            ))
            
            # Обновляем статистику
            self._update_statistics(record)
            
            # Если была ошибка, логируем её отдельно
            if record.status != ExecutionStatus.SUCCESS:
                self._log_error(record)
            
            conn.commit()
            
            # Проверяем лимит записей
            self._enforce_max_records()
            
            self.logger.info(f"Записано выполнение {record.execution_id}: {record.command_name} - {record.status.value}")
            
            return record.execution_id
    
    def _update_statistics(self, record: ExecutionRecord):
        """Обновляет агрегированную статистику."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            date_str = record.start_time.date().isoformat()
            
            # Проверяем, есть ли уже запись за этот день
            cursor.execute('''
                SELECT id, execution_count, success_count, failed_count, 
                       total_execution_time_ms, unique_users
                FROM execution_stats 
                WHERE command_name = ? AND date = ?
            ''', (record.command_name, date_str))
            
            row = cursor.fetchone()
            
            if row:
                # Обновляем существующую запись
                stat_id, exec_count, success_count, failed_count, total_time, unique_users = row
                
                exec_count += 1
                total_time += record.execution_time_ms
                
                if record.status == ExecutionStatus.SUCCESS:
                    success_count += 1
                else:
                    failed_count += 1
                
                avg_time = total_time / exec_count
                
                cursor.execute('''
                    UPDATE execution_stats 
                    SET execution_count = ?, success_count = ?, failed_count = ?,
                        total_execution_time_ms = ?, avg_execution_time_ms = ?
                    WHERE id = ?
                ''', (exec_count, success_count, failed_count, total_time, avg_time, stat_id))
            else:
                # Создаем новую запись
                success_count = 1 if record.status == ExecutionStatus.SUCCESS else 0
                failed_count = 1 - success_count
                unique_users = 1 if record.user_id else 0
                
                cursor.execute('''
                    INSERT INTO execution_stats 
                    (command_name, date, execution_count, success_count, 
                     failed_count, total_execution_time_ms, avg_execution_time_ms,
                     unique_users)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.command_name,
                    date_str,
                    1,
                    success_count,
                    failed_count,
                    record.execution_time_ms,
                    record.execution_time_ms,
                    unique_users
                ))
    
    def _log_error(self, record: ExecutionRecord):
        """Логирует ошибку выполнения."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Определяем тип ошибки
            error_type = record.status.value
            if not record.error_message and record.status != ExecutionStatus.SUCCESS:
                error_type = "unknown_error"
            
            cursor.execute('''
                INSERT INTO execution_errors 
                (execution_id, error_type, error_message, command_name, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                record.execution_id,
                error_type,
                record.error_message,
                record.command_name,
                record.user_id
            ))
    
    def _enforce_max_records(self):
        """Обеспечивает соблюдение максимального количества записей."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Считаем общее количество записей
            cursor.execute('SELECT COUNT(*) FROM execution_history')
            count = cursor.fetchone()[0]
            
            if count > self.max_records:
                # Удаляем самые старые записи
                delete_count = count - int(self.max_records * 0.9)  # Оставляем 90% лимита
                
                cursor.execute('''
                    DELETE FROM execution_history 
                    WHERE id IN (
                        SELECT id FROM execution_history 
                        ORDER BY created_at ASC 
                        LIMIT ?
                    )
                ''', (delete_count,))
                
                conn.commit()
                self.logger.info(f"Удалено {delete_count} старых записей истории")
    
    def _cleanup_old_records(self):
        """Удаляет записи старше retention_days дней."""
        cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).date().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM execution_history 
                WHERE DATE(start_time) < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                self.logger.info(f"Удалено {deleted_count} записей старше {self.retention_days} дней")
            
            conn.commit()
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionRecord]:
        """Получает запись о выполнении по ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM execution_history WHERE execution_id = ?', (execution_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_record(row)
            return None
    
    def search_executions(self,
                         command_name: str = None,
                         user_id: str = None,
                         status: ExecutionStatus = None,
                         start_date: datetime = None,
                         end_date: datetime = None,
                         limit: int = 100,
                         offset: int = 0) -> List[ExecutionRecord]:
        """
        Поиск записей выполнения по критериям.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            sql = "SELECT * FROM execution_history WHERE 1=1"
            params = []
            
            if command_name:
                sql += " AND command_name = ?"
                params.append(command_name)
            
            if user_id:
                sql += " AND user_id = ?"
                params.append(user_id)
            
            if status:
                sql += " AND status = ?"
                params.append(status.value)
            
            if start_date:
                sql += " AND start_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                sql += " AND start_time <= ?"
                params.append(end_date.isoformat())
            
            sql += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_record(row) for row in rows]
    
    def get_statistics(self,
                      command_name: str = None,
                      start_date: datetime = None,
                      end_date: datetime = None) -> Dict[str, Any]:
        """
        Возвращает статистику выполнения команд.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Базовые метрики
            stats = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'avg_execution_time_ms': 0,
                'unique_users': 0,
                'most_used_commands': [],
                'recent_errors': []
            }
            
            # Общее количество выполнений
            sql_total = "SELECT COUNT(*) FROM execution_history"
            params_total = []
            
            if command_name:
                sql_total += " WHERE command_name = ?"
                params_total.append(command_name)
            
            cursor.execute(sql_total, params_total)
            stats['total_executions'] = cursor.fetchone()[0]
            
            # Успешные выполнения
            sql_success = "SELECT COUNT(*) FROM execution_history WHERE status = 'success'"
            params_success = []
            
            if command_name:
                sql_success += " AND command_name = ?"
                params_success.append(command_name)
            
            cursor.execute(sql_success, params_success)
            stats['successful_executions'] = cursor.fetchone()[0]
            
            # Неудачные выполнения
            stats['failed_executions'] = stats['total_executions'] - stats['successful_executions']
            
            # Среднее время выполнения
            sql_avg = "SELECT AVG(execution_time_ms) FROM execution_history WHERE status = 'success'"
            params_avg = []
            
            if command_name:
                sql_avg += " AND command_name = ?"
                params_avg.append(command_name)
            
            cursor.execute(sql_avg, params_avg)
            avg_time = cursor.fetchone()[0]
            stats['avg_execution_time_ms'] = round(avg_time or 0, 2)
            
            # Уникальные пользователи
            sql_users = "SELECT COUNT(DISTINCT user_id) FROM execution_history WHERE user_id != ''"
            params_users = []
            
            if command_name:
                sql_users += " AND command_name = ?"
                params_users.append(command_name)
            
            cursor.execute(sql_users, params_users)
            stats['unique_users'] = cursor.fetchone()[0]
            
            # Самые используемые команды
            cursor.execute('''
                SELECT command_name, COUNT(*) as usage_count
                FROM execution_history
                GROUP BY command_name
                ORDER BY usage_count DESC
                LIMIT 10
            ''')
            stats['most_used_commands'] = [
                {'command': row[0], 'count': row[1]}
                for row in cursor.fetchall()
            ]
            
            # Последние ошибки
            cursor.execute('''
                SELECT command_name, error_message, timestamp
                FROM execution_errors
                WHERE resolved = 0
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            stats['recent_errors'] = [
                {'command': row[0], 'error': row[1], 'time': row[2]}
                for row in cursor.fetchall()
            ]
            
            # Ежедневная статистика (последние 7 дней)
            cursor.execute('''
                SELECT date, SUM(execution_count), SUM(success_count), SUM(failed_count)
                FROM execution_stats
                WHERE date >= DATE('now', '-7 days')
                GROUP BY date
                ORDER BY date DESC
            ''')
            
            daily_stats = []
            for row in cursor.fetchall():
                daily_stats.append({
                    'date': row[0],
                    'total': row[1],
                    'success': row[2],
                    'failed': row[3],
                    'success_rate': round(row[2] / row[1] * 100, 2) if row[1] > 0 else 0
                })
            
            stats['daily_stats_last_7_days'] = daily_stats
            
            return stats
    
    def export_to_csv(self, file_path: str, 
                     start_date: datetime = None,
                     end_date: datetime = None):
        """
        Экспортирует историю выполнения в CSV файл.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            sql = '''
                SELECT execution_id, command_name, input_text, status, 
                       execution_time_ms, user_id, start_time, end_time
                FROM execution_history
                WHERE 1=1
            '''
            params = []
            
            if start_date:
                sql += " AND start_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                sql += " AND start_time <= ?"
                params.append(end_date.isoformat())
            
            sql += " ORDER BY start_time DESC"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # Записываем в CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Заголовки
                writer.writerow([
                    'Execution ID', 'Command Name', 'Input Text', 'Status',
                    'Execution Time (ms)', 'User ID', 'Start Time', 'End Time'
                ])
                
                # Данные
                for row in rows:
                    writer.writerow(row)
            
            self.logger.info(f"Экспортировано {len(rows)} записей в {file_path}")
    
    def mark_error_resolved(self, error_id: int):
        """Отмечает ошибку как исправленную."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE execution_errors 
                SET resolved = 1 
                WHERE id = ?
            ''', (error_id,))
            
            conn.commit()
    
    def _row_to_record(self, row) -> ExecutionRecord:
        """Конвертирует строку SQLite в ExecutionRecord."""
        data = dict(row)
        
        # Парсим JSON поля
        json_fields = ['tokens', 'parameters', 'metadata']
        for field in json_fields:
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except:
                    data[field] = [] if field == 'tokens' else {}
        
        # Конвертируем строки в Enum
        if data.get('status'):
            data['status'] = ExecutionStatus(data['status'])
        
        return ExecutionRecord.from_dict(data)


# Пример использования
if __name__ == "__main__":
    import logging
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем логировщик
    logger = HistoryLogger()
    
    # Создаем тестовую запись
    record = ExecutionRecord(
        command_name="create_project",
        input_text="create new project 'test_project'",
        tokens=["create", "new", "project", "test_project"],
        parameters={"project_name": "test_project", "template": "basic"},
        result="Project 'test_project' created successfully",
        status=ExecutionStatus.SUCCESS,
        user_id="user_123",
        user_role="admin",
        adapter_type="CLI",
        confidence_score=0.95,
        metadata={"source": "manual", "environment": "development"}
    )
    
    # Логируем выполнение
    execution_id = logger.log_execution(record)
    print(f"Запись выполнена с ID: {execution_id}")
    
    # Получаем статистику
    stats = logger.get_statistics()
    print(f"Общая статистика: {stats['total_executions']} выполнений")
    
    # Поиск записей
    recent_executions = logger.search_executions(limit=5)
    print(f"Последние {len(recent_executions)} выполнений:")
    
    for exec_record in recent_executions:
        print(f"  - {exec_record.command_name} ({exec_record.status.value})")
    
    # Экспорт в CSV
    logger.export_to_csv("data/execution_history_export.csv")
    print("Экспорт завершен")