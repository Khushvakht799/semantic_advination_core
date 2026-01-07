# scripts\setup.py
"""
Скрипт начальной настройки проекта semantic_advination_core.
Создает необходимые директории, файлы конфигурации, инициализирует базы данных.
"""

import os
import sys
import json
import yaml
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, Any, List, Optional

# Добавляем родительскую директорию в путь для импорта модулей проекта
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False):
    """Настройка логирования."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('setup.log', mode='w', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


class ProjectSetup:
    """Класс для настройки проекта."""
    
    def __init__(self, project_root: Path, force: bool = False, verbose: bool = False):
        """
        Инициализация настроек проекта.
        
        Args:
            project_root: Корневая директория проекта
            force: Принудительное выполнение (перезапись существующих файлов)
            verbose: Подробный вывод
        """
        self.project_root = project_root
        self.force = force
        self.verbose = verbose
        self.logger = setup_logging(verbose)
        
        # Пути к основным директориям
        self.dirs = {
            'config': project_root / 'config',
            'data': project_root / 'data',
            'logs': project_root / 'logs',
            'scripts': project_root / 'scripts',
            'tests': project_root / 'tests',
            'interfaces': project_root / 'interfaces',
            'storage': project_root / 'storage',
            'utils': project_root / 'utils',
            'adapters': project_root / 'adapters',
            'core': project_root / 'core',
            'docs': project_root / 'docs',
        }
        
        # Пути к файлам конфигурации
        self.config_files = {
            'defaults': self.dirs['config'] / 'defaults.yaml',
            'thresholds': self.dirs['config'] / 'thresholds.py',
            'deferral_policies': self.dirs['config'] / 'deferral_policies.py',
            'env_example': project_root / 'env.example',
            'requirements': project_root / 'requirements.txt',
            'pyproject': project_root / 'pyproject.toml',
            'gitignore': project_root / '.gitignore',
        }
        
        # Пути к базам данных
        self.db_files = {
            'commands': self.dirs['data'] / 'commands.db',
            'deferred_tasks': self.dirs['data'] / 'deferred_tasks.db',
            'execution_history': self.dirs['data'] / 'execution_history.db',
        }
    
    def run_setup(self) -> bool:
        """Запускает полную настройку проекта."""
        self.logger.info("=" * 60)
        self.logger.info("Начало настройки проекта semantic_advination_core")
        self.logger.info(f"Корневая директория: {self.project_root}")
        self.logger.info("=" * 60)
        
        try:
            # 1. Создание директорий
            self.logger.info("\n1. Создание структуры директорий...")
            self.create_directories()
            
            # 2. Создание файлов конфигурации
            self.logger.info("\n2. Создание файлов конфигурации...")
            self.create_config_files()
            
            # 3. Инициализация баз данных
            self.logger.info("\n3. Инициализация баз данных...")
            self.initialize_databases()
            
            # 4. Создание скриптов
            self.logger.info("\n4. Создание вспомогательных скриптов...")
            self.create_scripts()
            
            # 5. Импорт тестовых данных
            self.logger.info("\n5. Импорт тестовых данных...")
            self.import_test_data()
            
            # 6. Проверка окружения
            self.logger.info("\n6. Проверка окружения...")
            self.check_environment()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Настройка проекта успешно завершена!")
            self.logger.info("=" * 60)
            
            self.print_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при настройке проекта: {e}")
            self.logger.exception("Детали ошибки:")
            return False
    
    def create_directories(self):
        """Создает структуру директорий проекта."""
        directories_created = 0
        directories_existed = 0
        
        for name, path in self.dirs.items():
            if path.exists():
                if self.force:
                    if self.verbose:
                        self.logger.debug(f"Пересоздание директории: {name}")
                    shutil.rmtree(path)
                    path.mkdir(parents=True, exist_ok=True)
                    directories_created += 1
                else:
                    if self.verbose:
                        self.logger.debug(f"Директория уже существует: {name}")
                    directories_existed += 1
            else:
                path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Создана директория: {name}")
                directories_created += 1
            
            # Создаем .gitkeep файлы в пустых директориях
            gitkeep_file = path / '.gitkeep'
            if not gitkeep_file.exists():
                gitkeep_file.touch()
        
        self.logger.info(f"Директории: создано {directories_created}, уже существовало {directories_existed}")
    
    def create_config_files(self):
        """Создает файлы конфигурации."""
        # 1. defaults.yaml
        self._create_defaults_yaml()
        
        # 2. thresholds.py
        self._create_thresholds_py()
        
        # 3. deferral_policies.py
        self._create_deferral_policies_py()
        
        # 4. env.example
        self._create_env_example()
        
        # 5. requirements.txt
        self._create_requirements_txt()
        
        # 6. pyproject.toml
        self._create_pyproject_toml()
        
        # 7. .gitignore
        self._create_gitignore()
    
    def _create_defaults_yaml(self):
        """Создает файл конфигурации defaults.yaml."""
        defaults_content = """# config/defaults.yaml
# Конфигурационные значения по умолчанию для semantic_advination_core

# Настройки токенизатора
tokenizer:
  stop_words: ["the", "a", "an", "and", "or", "in", "on", "at", "to", "for"]
  preserve_case: false
  split_on_punctuation: true

# Настройки схожести
similarity:
  weights:
    cosine: 0.4
    levenshtein: 0.4
    jaccard: 0.2
  threshold: 0.5  # минимальная схожесть для совпадения

# Настройки базы данных команд
database:
  path: "data/commands.db"
  backup_on_start: true
  auto_vacuum: true

# Настройки хранилища Trie
trie_storage:
  max_suggestions: 10
  cache_size: 1000

# Настройки выполнения команд
execution:
  timeout_seconds: 30
  max_retries: 3
  retry_delay_ms: 1000

# Настройки логирования
logging:
  level: "INFO"
  file: "logs/semantic_advination.log"
  max_size_mb: 10
  backup_count: 5

# Настройки отложенных задач
deferred:
  max_queue_size: 1000
  poll_interval_seconds: 5
  retry_policy: "exponential_backoff"

# Настройки адаптеров
adapters:
  shell:
    prompt: "advination> "
    history_file: "data/shell_history.txt"
    max_history: 1000
  api:
    host: "localhost"
    port: 8000
    workers: 4
    cors_origins: ["*"]

# Настройки безопасности
security:
  allowed_users: ["*"]
  require_auth: false
  api_key_header: "X-API-Key"

# Настройки мониторинга
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_path: "/health"

# Примеры команд по умолчанию
default_commands:
  - name: "help"
    tokens: ["help"]
    description: "Показывает справку по командам"
    action_type: "builtin"
    action_data: {"handler": "show_help"}
    category: "system"
  
  - name: "list_commands"
    tokens: ["list", "commands"]
    description: "Список всех доступных команд"
    action_type: "builtin"
    action_data: {"handler": "list_commands"}
    category: "system"
  
  - name: "create_project"
    tokens: ["create", "project"]
    description: "Создает новый проект"
    action_type: "shell"
    action_data: {"command": "mkdir {project_name}"}
    category: "project_management"
  
  - name: "show_stats"
    tokens: ["show", "statistics"]
    description: "Показывает статистику системы"
    action_type: "builtin"
    action_data: {"handler": "show_statistics"}
    category: "system"
"""
        self._write_config_file(self.config_files['defaults'], defaults_content)
    
    def _create_thresholds_py(self):
        """Создает файл thresholds.py."""
        thresholds_content = '''# config/thresholds.py
"""
Пороговые значения для системы semantic_advination_core.
"""

from dataclasses import dataclass

@dataclass
class Thresholds:
    """Пороговые значения системы."""
    
    # Пороги схожести
    similarity_min_match: float = 0.5          # Минимальная схожесть для совпадения команд
    similarity_high_confidence: float = 0.8    # Порог высокой уверенности
    similarity_fuzzy_match: float = 0.3        # Порог для нечёткого совпадения (предложения)
    
    # Пороги для токенизатора
    token_min_length: int = 1                  # Минимальная длина токена
    token_max_length: int = 50                 # Максимальная длина токена
    max_tokens_per_command: int = 10           # Максимальное количество токенов в команде
    
    # Пороги для хранилища Trie
    trie_max_suggestions: int = 10             # Максимальное количество предложений
    trie_cache_size: int = 1000                # Размер кэша Trie
    trie_min_prefix_length: int = 1            # Минимальная длина префикса для поиска
    
    # Пороги для выполнения команд
    execution_timeout_seconds: int = 30        # Таймаут выполнения команды
    execution_max_retries: int = 3             # Максимальное количество повторов
    execution_retry_delay_ms: int = 1000       # Задержка между повторами
    
    # Пороги для отложенных задач
    deferred_max_queue_size: int = 1000        # Максимальный размер очереди отложенных задач
    deferred_max_retries: int = 5              # Максимальное количество повторов для отложенных задач
    deferred_timeout_hours: int = 24           # Время жизни отложенной задачи
    
    # Пороги для логирования и мониторинга
    log_max_file_size_mb: int = 10             # Максимальный размер лог-файла
    log_backup_count: int = 5                  # Количество бэкапов логов
    metrics_update_interval_seconds: int = 60  # Интервал обновления метрик
    
    # Пороги для безопасности
    max_requests_per_minute: int = 100         # Ограничение скорости запросов
    session_timeout_minutes: int = 30          # Таймаут сессии
    password_min_length: int = 8               # Минимальная длина пароля
    
    # Пороги для качества работы
    confidence_min_for_auto_execute: float = 0.9  # Минимальная уверенность для авто-выполнения
    confirmation_threshold: float = 0.7           # Порог для запроса подтверждения
    learning_min_samples: int = 5                 # Минимальное количество образцов для обучения
    
    # Пороги для производительности
    cache_ttl_seconds: int = 300               # Время жизни кэша
    max_concurrent_commands: int = 10          # Максимальное количество одновременных команд
    memory_usage_limit_mb: int = 500           # Лимит использования памяти


# Глобальный экземпляр для использования по умолчанию
default_thresholds = Thresholds()
'''
        self._write_config_file(self.config_files['thresholds'], thresholds_content)
    
    def _create_deferral_policies_py(self):
        """Создает файл deferral_policies.py."""
        deferral_content = '''# config/deferral_policies.py
"""
Политики отложенного выполнения команд.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class DeferralReason(Enum):
    """Причины отложенного выполнения."""
    RESOURCE_BUSY = "resource_busy"
    SCHEDULED_TIME = "scheduled_time"
    USER_REQUEST = "user_request"
    DEPENDENCY_NOT_MET = "dependency_not_met"
    RATE_LIMITED = "rate_limited"
    LOW_PRIORITY = "low_priority"


class RetryStrategy(Enum):
    """Стратегии повторных попыток."""
    IMMEDIATE = "immediate"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    RANDOMIZED_BACKOFF = "randomized_backoff"


@dataclass
class DeferralPolicy:
    """Политика отложенного выполнения."""
    
    name: str
    description: str = ""
    enabled: bool = True
    default_delay_seconds: int = 300
    min_delay_seconds: int = 60
    max_delay_seconds: int = 86400
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    priority: int = 5
    apply_to_categories: Optional[list] = None
    
    def __post_init__(self):
        if self.apply_to_categories is None:
            self.apply_to_categories = []


# Стандартные политики
DEFAULT_POLICIES = [
    DeferralPolicy(
        name="long_running",
        description="Для длительных операций",
        default_delay_seconds=600,
        priority=3
    ),
    DeferralPolicy(
        name="low_priority",
        description="Для команд низкого приоритета",
        default_delay_seconds=1800,
        priority=2,
        apply_to_categories=["maintenance", "cleanup"]
    ),
    DeferralPolicy(
        name="resource_intensive",
        description="Для ресурсоемких операций",
        default_delay_seconds=300,
        retry_strategy=RetryStrategy.RANDOMIZED_BACKOFF,
        max_retries=5
    )
]
'''
        self._write_config_file(self.config_files['deferral_policies'], deferral_content)
    
    def _create_env_example(self):
        """Создает файл env.example."""
        env_content = """# env.example
# Копируйте этот файл в .env и настройте переменные окружения

# База данных
DATABASE_PATH=data/commands.db
DATABASE_BACKUP=true

# Логирование
LOG_LEVEL=INFO
LOG_FILE=logs/semantic_advination.log
LOG_MAX_SIZE_MB=10

# API сервер
API_HOST=localhost
API_PORT=8000
API_WORKERS=4

# Безопасность
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000

# Настройки производительности
MAX_CONCURRENT_COMMANDS=10
CACHE_TTL_SECONDS=300
MEMORY_LIMIT_MB=500

# Отладка
DEBUG=false
VERBOSE=false
"""
        self._write_config_file(self.config_files['env_example'], env_content)
    
    def _create_requirements_txt(self):
        """Создает файл requirements.txt."""
        requirements_content = """# Основные зависимости
python>=3.8

# Основные библиотеки
click>=8.0.0
pyyaml>=6.0
msgpack>=1.0.0
sqlite3

# Веб-фреймворк (опционально)
fastapi>=0.68.0
uvicorn[standard]>=0.15.0

# CLI интерфейс
rich>=10.0.0
prompt-toolkit>=3.0.0

# Утилиты
python-dotenv>=0.19.0
pytz>=2021.3
colorama>=0.4.4

# Тестирование
pytest>=6.0.0
pytest-cov>=2.0.0

# Разработка
black>=21.0.0
flake8>=4.0.0
mypy>=0.910
"""
        self._write_config_file(self.config_files['requirements'], requirements_content)
    
    def _create_pyproject_toml(self):
        """Создает файл pyproject.toml."""
        pyproject_content = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "semantic_advination_core"
version = "0.1.0"
description = "Semantic command advination system core"
readme = "README.md"
authors = [
    {name = "Development Team", email = "dev@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
requires-python = ">=3.8"
dependencies = [
    "click>=8.0.0",
    "pyyaml>=6.0",
    "msgpack>=1.0.0",
    "rich>=10.0.0",
    "python-dotenv>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "black>=21.0.0",
    "flake8>=4.0.0",
    "mypy>=0.910",
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
]
web = [
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/semantic_advination_core"
Repository = "https://github.com/yourusername/semantic_advination_core.git"
Issues = "https://github.com/yourusername/semantic_advination_core/issues"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --cov=semantic_advination_core --cov-report=term-missing"
testpaths = [
    "tests",
]
"""
        self._write_config_file(self.config_files['pyproject'], pyproject_content)
    
    def _create_gitignore(self):
        """Создает файл .gitignore."""
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/*.db
data/*.json
data/*.log
logs/*.log
*.log
*.pdf
*.pyc
__pycache__/

# Local configuration
local_settings.py
.env.local
.env.*.local

# Backup files
*.bak
*.backup
"""
        self._write_config_file(self.config_files['gitignore'], gitignore_content)
    
    def _write_config_file(self, file_path: Path, content: str):
        """Записывает конфигурационный файл."""
        if file_path.exists() and not self.force:
            self.logger.debug(f"Файл уже существует: {file_path.name}")
            return
        
        # Создаем директорию, если её нет
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.info(f"Создан файл: {file_path.name}")
    
    def initialize_databases(self):
        """Инициализирует базы данных."""
        # Создаем директорию data, если её нет
        self.dirs['data'].mkdir(parents=True, exist_ok=True)
        
        for db_name, db_path in self.db_files.items():
            if db_path.exists() and not self.force:
                self.logger.debug(f"База данных уже существует: {db_name}")
                continue
            
            self.logger.info(f"Инициализация базы данных: {db_name}")
            
            try:
                if db_name == 'commands':
                    self._init_commands_db(db_path)
                elif db_name == 'deferred_tasks':
                    self._init_deferred_tasks_db(db_path)
                elif db_name == 'execution_history':
                    self._init_execution_history_db(db_path)
                
                self.logger.info(f"База данных создана: {db_path}")
            except Exception as e:
                self.logger.error(f"Ошибка создания базы данных {db_name}: {e}")
    
    def _init_commands_db(self, db_path: Path):
        """Инициализирует базу данных команд."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Таблица команд
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                tokens TEXT NOT NULL,
                description TEXT,
                action_type TEXT DEFAULT 'function',
                action_data TEXT,
                category TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Таблица истории выполнения
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command_id INTEGER,
                input_text TEXT,
                parameters TEXT,
                result TEXT,
                success BOOLEAN,
                execution_time_ms INTEGER,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                FOREIGN KEY (command_id) REFERENCES commands (id)
            )
        ''')
        
        # Таблица синонимов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command_id INTEGER,
                alias TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (command_id) REFERENCES commands (id)
            )
        ''')
        
        # Добавляем индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_name ON commands (name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_category ON commands (category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_command_id ON execution_history (command_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_executed_at ON execution_history (executed_at)')
        
        # Добавляем тестовые команды
        test_commands = [
            ("help", '["help"]', "Показывает справку по командам", "builtin", '{"handler": "show_help"}', "system", '["help", "system"]'),
            ("list_commands", '["list", "commands"]', "Список всех доступных команд", "builtin", '{"handler": "list_commands"}', "system", '["list", "commands"]'),
            ("create_project", '["create", "project"]', "Создает новый проект", "shell", '{"command": "mkdir {project_name}"}', "project_management", '["project", "create"]'),
            ("delete_file", '["delete", "file"]', "Удаляет файл", "shell", '{"command": "rm {file_path}"}', "file_operations", '["delete", "file"]'),
            ("show_stats", '["show", "statistics"]', "Показывает статистику системы", "builtin", '{"handler": "show_statistics"}', "system", '["stats", "system"]'),
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO commands (name, tokens, description, action_type, action_data, category, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', test_commands)
        
        conn.commit()
        conn.close()
    
    def _init_deferred_tasks_db(self, db_path: Path):
        """Инициализирует базу данных отложенных задач."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deferred_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL UNIQUE,
                command_name TEXT NOT NULL,
                command_data TEXT,
                parameters TEXT,
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
                metadata TEXT
            )
        ''')
        
        # Индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_status ON deferred_tasks (status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_scheduled ON deferred_tasks (scheduled_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_priority ON deferred_tasks (priority)')
        
        conn.commit()
        conn.close()
    
    def _init_execution_history_db(self, db_path: Path):
        """Инициализирует базу данных истории выполнения."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL UNIQUE,
                command_id INTEGER,
                command_name TEXT NOT NULL,
                input_text TEXT,
                tokens TEXT,
                parameters TEXT,
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
                metadata TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_command ON execution_history (command_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_user ON execution_history (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_time ON execution_history (start_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_status ON execution_history (status)')
        
        conn.commit()
        conn.close()
    
    def create_scripts(self):
        """Создает вспомогательные скрипты."""
        # Скрипт импорта команд
        import_script = self.dirs['scripts'] / 'import_commands.py'
        import_content = '''#!/usr/bin/env python3
"""
Скрипт импорта команд из JSON/YAML файлов в базу данных.
"""

import json
import yaml
import sqlite3
import argparse
from pathlib import Path
import sys

def import_commands_from_json(file_path: Path, db_path: Path):
    """Импортирует команды из JSON файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        commands = json.load(f)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for command in commands:
        cursor.execute('''
            INSERT OR REPLACE INTO commands 
            (name, tokens, description, action_type, action_data, category, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            command['name'],
            json.dumps(command.get('tokens', [])),
            command.get('description', ''),
            command.get('action_type', 'function'),
            json.dumps(command.get('action_data', {})),
            command.get('category', ''),
            json.dumps(command.get('tags', []))
        ))
    
    conn.commit()
    conn.close()
    print(f"Импортировано {len(commands)} команд из {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Импорт команд в базу данных')
    parser.add_argument('file', type=str, help='Путь к файлу с командами (JSON/YAML)')
    parser.add_argument('--db', type=str, default='data/commands.db', help='Путь к базе данных')
    args = parser.parse_args()
    
    file_path = Path(args.file)
    db_path = Path(args.db)
    
    if not file_path.exists():
        print(f"Файл не найден: {file_path}")
        sys.exit(1)
    
    if file_path.suffix.lower() == '.json':
        import_commands_from_json(file_path, db_path)
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        print("Импорт из YAML пока не реализован")
    else:
        print(f"Неподдерживаемый формат файла: {file_path.suffix}")

if __name__ == '__main__':
    main()
'''
        self._write_config_file(import_script, import_content)
        
        # Делаем скрипт исполняемым (Unix)
        if os.name != 'nt':  # не Windows
            import_script.chmod(0o755)
    
    def import_test_data(self):
        """Импортирует тестовые данные."""
        # Создаем тестовый JSON файл с командами
        test_data_file = self.dirs['data'] / 'test_commands.json'
        test_data = [
            {
                "name": "greet",
                "tokens": ["greet", "hello"],
                "description": "Приветствует пользователя",
                "action_type": "builtin",
                "action_data": {"handler": "greet_user"},
                "category": "social",
                "tags": ["greeting", "hello"]
            },
            {
                "name": "get_time",
                "tokens": ["time", "current", "time"],
                "description": "Показывает текущее время",
                "action_type": "builtin",
                "action_data": {"handler": "get_current_time"},
                "category": "system",
                "tags": ["time", "clock"]
            },
            {
                "name": "list_files",
                "tokens": ["list", "files"],
                "description": "Показывает файлы в текущей директории",
                "action_type": "shell",
                "action_data": {"command": "ls -la"},
                "category": "file_operations",
                "tags": ["files", "list", "directory"]
            }
        ]
        
        with open(test_data_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Создан тестовый файл: {test_data_file.name}")
    
    def check_environment(self):
        """Проверяет окружение Python."""
        self.logger.info("Проверка версии Python...")
        python_version = sys.version_info
        self.logger.info(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.logger.warning("Требуется Python 3.8 или выше!")
        
        # Проверяем доступность основных модулей
        self.logger.info("Проверка доступности модулей...")
        modules_to_check = ['json', 'yaml', 'sqlite3', 'argparse', 'logging', 'pathlib']
        
        for module in modules_to_check:
            try:
                __import__(module)
                self.logger.debug(f"  ✓ {module}")
            except ImportError:
                self.logger.warning(f"  ✗ {module} не доступен")
    
    def print_summary(self):
        """Выводит краткую сводку."""
        print("\n" + "=" * 60)
        print("СВОДКА НАСТРОЙКИ ПРОЕКТА")
        print("=" * 60)
        
        # Проверяем созданные директории
        print("\nСозданные директории:")
        for name, path in self.dirs.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path}")
        
        # Проверяем созданные файлы конфигурации
        print("\nСозданные файлы конфигурации:")
        for name, path in self.config_files.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path}")
        
        # Проверяем созданные базы данных
        print("\nСозданные базы данных:")
        for name, path in self.db_files.items():
            status = "✓" if path.exists() else "✗"
            size = path.stat().st_size if path.exists() else 0
            print(f"  {status} {name}: {path} ({size} байт)")
        
        print("\n" + "=" * 60)
        print("СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Установите зависимости: pip install -r requirements.txt")
        print("2. Настройте переменные окружения: cp env.example .env")
        print("3. Отредактируйте .env файл под ваше окружение")
        print("4. Запустите тесты: pytest")
        print("5. Запустите основное приложение: python main.py")
        print("=" * 60)


def main():
    """Точка входа скрипта."""
    parser = argparse.ArgumentParser(
        description='Скрипт настройки проекта semantic_advination_core'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='Корневая директория проекта (по умолчанию: текущая директория)'
    )
    
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Принудительное создание/перезапись файлов'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод'
    )
    
    parser.add_argument(
        '--skip-db',
        action='store_true',
        help='Пропустить инициализацию баз данных'
    )
    
    args = parser.parse_args()
    
    # Определяем корневую директорию
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        print(f"Ошибка: директория не существует: {project_root}")
        sys.exit(1)
    
    # Запускаем настройку
    setup = ProjectSetup(project_root, args.force, args.verbose)
    
    if args.skip_db:
        setup.db_files = {}  # Очищаем список баз данных
    
    success = setup.run_setup()
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()