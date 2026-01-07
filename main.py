# main.py
"""
Главный CLI модуль для семантического предсказания команд.
Интегрирует все компоненты системы: TrieStorage, Adivinator, Validator, Orchestrator.
"""

import json
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Импорт компонентов системы
from storage.trie_storage import CommandTrie, Command, TrieStorage
from core.adivinator import Adivinator, AdvinationResult
from core.validator import Validator, Context, ValidationConfig
from core.orchestrator import Orchestrator, OrchestrationConfig, DialogState


class CLI:
    """CLI интерфейс для семантического предсказания команд"""
    
    def __init__(self, data_source: str = "json", data_path: Optional[str] = None):
        """
        Инициализация CLI
        
        Args:
            data_source: Источник данных ("json", "sqlite", или "memory")
            data_path: Путь к файлу данных
        """
        self.data_source = data_source
        self.data_path = data_path
        self.trie = CommandTrie()
        self.storage = None
        self.adivinator = None
        self.validator = None
        self.orchestrator = None
        self.context = Context()
        self.session_history: List[Dict[str, Any]] = []
        
        # Настройка логирования
        self._setup_logging()
        
        # Инициализация системы
        self._initialize_system()
        
    def _setup_logging(self) -> None:
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('semantic_advination.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_system(self) -> None:
        """Инициализация всех компонентов системы"""
        self.logger.info("Initializing semantic advination system...")
        
        try:
            # 1. Инициализация хранилища
            if self.data_source == "json" and self.data_path:
                self.storage = TrieStorage()
                if os.path.exists(self.data_path):
                    self.storage.load_from_json(self.data_path)
                    self.trie = self.storage.trie
                    self.logger.info(f"Loaded commands from JSON: {self.data_path}")
                else:
                    self.logger.warning(f"JSON file not found: {self.data_path}. Using empty trie.")
            elif self.data_source == "sqlite" and self.data_path:
                self.storage = TrieStorage(self.data_path)
                self.storage.load_from_sqlite()
                self.trie = self.storage.trie
                self.logger.info(f"Loaded commands from SQLite: {self.data_path}")
            else:
                # Используем тестовые данные
                self._load_test_commands()
                self.logger.info("Loaded test commands")
            
            # 2. Инициализация компонентов
            self.adivinator = Adivinator(self.trie)
            self.validator = Validator()
            
            # 3. Инициализация оркестратора
            orchestrator_config = OrchestrationConfig(
                enable_validation=True,
                enable_dialog=True,
                min_exact_confidence=0.9,
                min_partial_confidence=0.5,
                max_suggestions=5,
                log_level="INFO"
            )
            
            self.orchestrator = Orchestrator(
                self.adivinator,
                self.validator,
                orchestrator_config
            )
            
            # 4. Настройка контекста
            self.context = Context(
                domain="general",
                user_role="user",
                environment="cli",
                time=datetime.now(),
                metadata={"source": "cli"}
            )
            
            self.logger.info("System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            sys.exit(1)
    
    def _load_test_commands(self) -> None:
        """Загрузка тестовых команд в Trie"""
        test_commands = [
            Command(
                name="create_user",
                description="Создание нового пользователя",
                tokens=["create", "user", "add", "new"],
                tags=["admin", "user_management", "create"],
                metadata={"permissions": ["admin"]}
            ),
            Command(
                name="delete_user",
                description="Удаление пользователя",
                tokens=["delete", "user", "remove", "drop"],
                tags=["admin", "user_management", "delete"],
                metadata={"permissions": ["admin"], "dangerous": True}
            ),
            Command(
                name="list_users",
                description="Просмотр списка пользователей",
                tokens=["list", "users", "show", "display", "view"],
                tags=["admin", "user_management", "read"],
                metadata={"permissions": ["admin", "user"]}
            ),
            Command(
                name="create_file",
                description="Создание нового файла",
                tokens=["create", "file", "new", "touch"],
                tags=["file_operations", "create"],
                metadata={"permissions": ["admin", "user"]}
            ),
            Command(
                name="delete_file",
                description="Удаление файла",
                tokens=["delete", "file", "remove", "rm"],
                tags=["file_operations", "delete"],
                metadata={"permissions": ["admin", "user"], "dangerous": True}
            ),
            Command(
                name="search_files",
                description="Поиск файлов",
                tokens=["search", "files", "find", "locate"],
                tags=["file_operations", "search"],
                metadata={"permissions": ["admin", "user"]}
            ),
            Command(
                name="system_status",
                description="Проверка состояния системы",
                tokens=["system", "status", "health", "check"],
                tags=["system", "monitoring", "status"],
                metadata={"permissions": ["admin"]}
            ),
            Command(
                name="restart_service",
                description="Перезапуск сервиса",
                tokens=["restart", "service", "reboot", "reload"],
                tags=["system", "administration", "restart"],
                metadata={"permissions": ["admin"]}
            ),
            Command(
                name="show_logs",
                description="Просмотр логов",
                tokens=["show", "logs", "view", "display", "log"],
                tags=["system", "logs", "debugging"],
                metadata={"permissions": ["admin", "developer"]}
            ),
            Command(
                name="help",
                description="Показать справку",
                tokens=["help", "assist", "support", "guide"],
                tags=["general", "help"],
                metadata={"permissions": ["admin", "user", "guest"]}
            ),
            Command(
                name="exit",
                description="Выход из системы",
                tokens=["exit", "quit", "close", "bye"],
                tags=["general", "exit"],
                metadata={"permissions": ["admin", "user", "guest"]}
            ),
            Command(
                name="configure_system",
                description="Настройка системы",
                tokens=["configure", "system", "setup", "config"],
                tags=["system", "configuration"],
                metadata={"permissions": ["admin"]}
            ),
            Command(
                name="backup_database",
                description="Резервное копирование базы данных",
                tokens=["backup", "database", "db", "save"],
                tags=["database", "backup"],
                metadata={"permissions": ["admin"]}
            ),
            Command(
                name="restore_database",
                description="Восстановление базы данных",
                tokens=["restore", "database", "db", "recover"],
                tags=["database", "restore"],
                metadata={"permissions": ["admin"], "dangerous": True}
            ),
            Command(
                name="monitor_performance",
                description="Мониторинг производительности",
                tokens=["monitor", "performance", "stats", "metrics"],
                tags=["system", "monitoring", "performance"],
                metadata={"permissions": ["admin"]}
            )
        ]
        
        # Вставляем все тестовые команды в Trie
        for command in test_commands:
            self.trie.insert(command)
        
        self.logger.info(f"Loaded {len(test_commands)} test commands")
    
    def _save_session_history(self) -> None:
        """Сохранение истории сессии"""
        try:
            history_file = Path("session_history.json")
            if self.session_history:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.session_history, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Session history saved to {history_file}")
        except Exception as e:
            self.logger.error(f"Failed to save session history: {e}")
    
    def _print_banner(self) -> None:
        """Печать баннера"""
        banner = """
        ╔══════════════════════════════════════════════════════╗
        ║         Semantic Advination System v1.0              ║
        ║         Система семантического предсказания команд   ║
        ╚══════════════════════════════════════════════════════╝
        
        Доступные команды:
          • create_user     - Создать пользователя
          • delete_user     - Удалить пользователя  
          • list_users      - Показать пользователей
          • create_file     - Создать файл
          • delete_file     - Удалить файл
          • search_files    - Найти файлы
          • system_status   - Статус системы
          • restart_service - Перезапустить сервис
          • show_logs       - Показать логи
          • help            - Справка
          • exit            - Выход
        
        Примеры запросов:
          • "создать пользователя"
          • "удалить файл"  
          • "показать логи системы"
          • "статус"
          • "помощь"
        
        Введите запрос или 'exit' для выхода.
        """
        print(banner)
    
    def _print_result(self, result: Dict[str, Any]) -> None:
        """Печать результата в читаемом формате"""
        print("\n" + "="*60)
        print(f"РЕЗУЛЬТАТ: {result.get('outcome', 'unknown')}")
        print("="*60)
        
        # Статистика
        print(f"Запрос: {result.get('query', '')}")
        print(f"Доверие: {result.get('confidence', 0):.2%}")
        print(f"Предложений: {result.get('suggestions_count', 0)}")
        
        # Предложения
        suggestions = result.get('suggestions', [])
        if suggestions:
            print("\nПРЕДЛОЖЕНИЯ:")
            for i, suggestion in enumerate(suggestions, 1):
                confidence = suggestion.get('confidence', 0)
                confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                
                print(f"\n{i}. {suggestion.get('command_name', 'unknown')}")
                print(f"   Доверие: {confidence:.2%} [{confidence_bar}]")
                print(f"   Описание: {suggestion.get('command_description', '')}")
                
                matched_tokens = suggestion.get('matched_tokens', [])
                if matched_tokens:
                    print(f"   Совпавшие токены: {', '.join(matched_tokens)}")
                
                matched_tags = suggestion.get('matched_tags', [])
                if matched_tags:
                    print(f"   Теги: {', '.join(matched_tags)}")
        
        # Шаги диалога
        dialog_steps = result.get('dialog_steps', [])
        if dialog_steps:
            print("\nДИАЛОГ:")
            for step in dialog_steps:
                print(f"\n  • {step.get('message', '')}")
                options = step.get('options', [])
                if options:
                    print(f"    Варианты: {', '.join(options)}")
        
        # Ошибки
        error_message = result.get('error_message')
        if error_message:
            print(f"\nОШИБКА: {error_message}")
        
        # Метаданные
        metadata = result.get('metadata', {})
        if metadata:
            print(f"\nДетали:")
            for key, value in metadata.items():
                if key not in ['original_query', 'context']:
                    print(f"  {key}: {value}")
        
        print("="*60 + "\n")
    
    def _handle_special_commands(self, input_str: str) -> Optional[Dict[str, Any]]:
        """Обработка специальных команд"""
        input_lower = input_str.lower().strip()
        
        if input_lower in ["exit", "quit", "q"]:
            print("Выход из системы...")
            self._save_session_history()
            sys.exit(0)
        
        elif input_lower in ["help", "помощь", "?"]:
            self._print_banner()
            return {
                'outcome': 'help',
                'message': 'Выведена справка'
            }
        
        elif input_lower in ["history", "история"]:
            print("\nИСТОРИЯ СЕССИИ:")
            for i, entry in enumerate(self.session_history[-10:], 1):
                print(f"{i}. {entry.get('query', '')} -> {entry.get('outcome', '')}")
            return {
                'outcome': 'history',
                'count': len(self.session_history)
            }
        
        elif input_lower in ["stats", "статистика"]:
            stats = self.orchestrator.get_workflow_info() if self.orchestrator else {}
            print("\nСТАТИСТИКА СИСТЕМЫ:")
            print(f"Команд в Trie: {len(self.trie.commands)}")
            print(f"Токенов: {len(self.trie.token_to_commands)}")
            print(f"Тегов: {len(self.trie.tag_to_commands)}")
            print(f"Шагов workflow: {len(stats.get('workflow_steps', []))}")
            return {
                'outcome': 'stats',
                'command_count': len(self.trie.commands)
            }
        
        elif input_lower.startswith("context "):
            parts = input_lower.split(" ", 2)
            if len(parts) >= 3:
                key = parts[1]
                value = parts[2]
                if hasattr(self.context, key):
                    setattr(self.context, key, value)
                    print(f"Контекст обновлен: {key} = {value}")
                else:
                    self.context.metadata[key] = value
                    print(f"Метаданные контекста обновлены: {key} = {value}")
            return {
                'outcome': 'context_updated',
                'context': self.context.to_dict()
            }
        
        elif input_lower in ["clear", "очистить"]:
            os.system('cls' if os.name == 'nt' else 'clear')
            return {'outcome': 'cleared'}
        
        return None
    
    def run_interactive(self) -> None:
        """Запуск интерактивного режима"""
        self._print_banner()
        
        print("Введите запрос (или 'help' для справки, 'exit' для выхода):\n")
        
        while True:
            try:
                # Чтение ввода
                prefix = input(">>> ").strip()
                
                if not prefix:
                    continue
                
                # Проверка специальных команд
                special_result = self._handle_special_commands(prefix)
                if special_result:
                    continue
                
                # Обработка запроса
                result = self.orchestrator.process(prefix, self.context)
                
                # Преобразование в словарь
                result_dict = result.to_dict()
                result_dict['query'] = prefix
                
                # Сохранение в историю
                self.session_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': prefix,
                    'outcome': result.outcome.value,
                    'confidence': result.confidence,
                    'suggestions_count': len(result.suggestions)
                })
                
                # Печать результата
                self._print_result(result_dict)
                
                # Обработка диалога
                if result.dialog_state in [DialogState.CLARIFYING, DialogState.CONFIRMING]:
                    self._handle_dialog_continuation(result)
                
            except KeyboardInterrupt:
                print("\n\nВыход из системы...")
                self._save_session_history()
                sys.exit(0)
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки запроса: {e}")
                print(f"\nОшибка: {e}\n")
    
    def _handle_dialog_continuation(self, result: Any) -> None:
        """Обработка продолжения диалога"""
        if not result.dialog_steps:
            return
        
        last_step = result.dialog_steps[-1]
        if last_step.options:
            print("\nВыберите вариант (введите номер):")
            for i, option in enumerate(last_step.options, 1):
                print(f"  {i}. {option}")
            print("  или введите свой вариант:")
            
            try:
                user_input = input(">>> ").strip()
                
                # Проверка выбора по номеру
                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(last_step.options):
                        user_input = last_step.options[idx]
                
                # Продолжение диалога
                dialog_result = self.orchestrator.continue_dialog(
                    result.request_id,
                    user_input
                )
                
                # Обработка результата диалога
                dialog_dict = dialog_result.to_dict()
                dialog_dict['query'] = f"Диалог: {user_input}"
                
                self._print_result(dialog_dict)
                
            except Exception as e:
                self.logger.error(f"Ошибка диалога: {e}")
                print(f"\nОшибка диалога: {e}\n")
    
    def run_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Пакетная обработка запросов
        
        Args:
            queries: Список запросов
            
        Returns:
            Список результатов
        """
        results = []
        
        print(f"\nПакетная обработка {len(queries)} запросов...\n")
        
        for query in queries:
            print(f"Обработка: '{query}'")
            
            try:
                result = self.orchestrator.process(query, self.context)
                result_dict = result.to_dict()
                result_dict['query'] = query
                
                results.append(result_dict)
                
                # Краткий вывод
                print(f"  Результат: {result.outcome.value}")
                print(f"  Предложений: {len(result.suggestions)}")
                if result.suggestions:
                    best = result.get_best_suggestion()
                    print(f"  Лучшее: {best.command_name} ({best.confidence:.2%})")
                print()
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки запроса '{query}': {e}")
                results.append({
                    'query': query,
                    'outcome': 'error',
                    'error_message': str(e)
                })
                print(f"  Ошибка: {e}\n")
        
        # Сохранение результатов
        try:
            with open('batch_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Результаты сохранены в batch_results.json")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
        
        return results
    
    def test_system(self) -> None:
        """Запуск тестов системы"""
        test_queries = [
            "create user",
            "удалить файл",
            "system status",
            "показать логи",
            "help",
            "неизвестная команда"
        ]
        
        print("\n" + "="*60)
        print("ТЕСТИРОВАНИЕ СИСТЕМЫ")
        print("="*60 + "\n")
        
        self.run_batch(test_queries)
        
        print("Тестирование завершено!")
        print("Для запуска интерактивного режима введите команды напрямую.\n")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Система семантического предсказания команд',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                    # Интерактивный режим
  %(prog)s --test             # Запуск тестов
  %(prog)s --json commands.json  # Загрузка команд из JSON
  %(prog)s --batch "query1" "query2"  # Пакетная обработка
        """
    )
    
    parser.add_argument(
        '--json',
        help='Путь к JSON файлу с командами',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--sqlite',
        help='Путь к SQLite базе данных',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--test',
        help='Запустить тесты системы',
        action='store_true'
    )
    
    parser.add_argument(
        '--batch',
        help='Пакетная обработка запросов',
        nargs='+',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--verbose',
        help='Подробный вывод',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Определение источника данных
        if args.json:
            data_source = "json"
            data_path = args.json
        elif args.sqlite:
            data_source = "sqlite"
            data_path = args.sqlite
        else:
            data_source = "memory"
            data_path = None
        
        # Создание CLI
        cli = CLI(data_source, data_path)
        
        # Выбор режима работы
        if args.test:
            cli.test_system()
        elif args.batch:
            cli.run_batch(args.batch)
        else:
            cli.run_interactive()
            
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

Также создам пример JSON файла с командами:
json

# commands.json
[
  {
    "name": "create_database",
    "description": "Создание новой базы данных",
    "tokens": ["create", "database", "db", "new"],
    "tags": ["database", "admin", "create"],
    "metadata": {
      "permissions": ["admin"],
      "category": "database"
    }
  },
  {
    "name": "optimize_system",
    "description": "Оптимизация производительности системы",
    "tokens": ["optimize", "system", "performance", "tune"],
    "tags": ["system", "optimization", "performance"],
    "metadata": {
      "permissions": ["admin"],
      "category": "maintenance"
    }
  },
  {
    "name": "send_notification",
    "description": "Отправка уведомления пользователям",
    "tokens": ["send", "notification", "alert", "message"],
    "tags": ["communication", "notification"],
    "metadata": {
      "permissions": ["admin", "moderator"],
      "category": "communication"
    }
  }
]