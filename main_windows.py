# main_windows.py (альтернативная версия для Windows)
"""
Версия для Windows с поддержкой pyreadline3
"""

import sys
import os

# Попытка использовать pyreadline3 для Windows
try:
    import pyreadline3 as readline
    READLINE_AVAILABLE = True
    print("Используется pyreadline3 для автодополнения")
except ImportError:
    try:
        import readline
        READLINE_AVAILABLE = True
    except ImportError:
        READLINE_AVAILABLE = False
        print("Внимание: readline недоступен. Используется упрощённый ввод.")

# Остальной код такой же как в main.py, начиная с импортов
from storage.trie_storage import CommandTrie, Command, TrieStorage
from core.adivinator import Adivinator, create_adivinator
from core.validator import Validator, Context, create_validator
from core.orchestrator import Orchestrator, OrchestrationMode, create_orchestrator


class AutocompleteCLI:
    """CLI с поддержкой автодополнения, подобного PowerShell"""
    
    def __init__(self, data_source: str = "demo", data_path: Optional[str] = None):
        self.data_source = data_source
        self.data_path = data_path
        self.trie = CommandTrie()
        self.orchestrator = None
        self.setup_done = False
        self.history: List[str] = []
        self.history_index = -1
        
        # Настройка системы
        self._setup_system()
        
        # Настройка автодополнения только если readline доступен
        if READLINE_AVAILABLE:
            self._setup_readline()
        else:
            print("Используется упрощённый режим ввода (без readline).")
    
    def _setup_system(self):
        """Настройка всех компонентов системы"""
        print("Инициализация системы автодополнения...")
        
        try:
            # Загрузка команд
            if self.data_source == "json" and self.data_path:
                storage = TrieStorage()
                storage.load_from_json(self.data_path)
                self.trie = storage.trie
                print(f"Загружено команд из JSON: {len(self.trie.commands)}")
            elif self.data_source == "demo":
                self._load_demo_commands()
                print(f"Загружено демо-команд: {len(self.trie.commands)}")
            else:
                print("Используется пустое хранилище команд")
            
            # Создание компонентов
            adivinator = Adivinator(self.trie)
            validator = create_validator()
            self.orchestrator = create_orchestrator(adivinator, validator)
            
            self.setup_done = True
            print("Система готова к работе!")
            
        except Exception as e:
            print(f"Ошибка инициализации: {e}")
            sys.exit(1)
    
    def _load_demo_commands(self):
        """Загрузка демонстрационных команд (PowerShell-style)"""
        demo_commands = [
            # PowerShell-like commands
            Command("Get-Process", "Получить информацию о процессах", 
                   ["get", "process", "ps", "list", "show"], 
                   ["system", "process", "management"]),
            
            Command("Get-Service", "Получить информацию о сервисах", 
                   ["get", "service", "svc", "list"], 
                   ["system", "service", "management"]),
            
            Command("Get-ChildItem", "Получить содержимое директории", 
                   ["get", "child", "item", "ls", "dir", "list"], 
                   ["filesystem", "directory", "file"]),
            
            Command("Set-Location", "Сменить текущую директорию", 
                   ["set", "location", "cd", "chdir", "change"], 
                   ["filesystem", "directory", "navigation"]),
            
            Command("Copy-Item", "Копировать файл или директорию", 
                   ["copy", "item", "cp", "duplicate"], 
                   ["filesystem", "file", "copy"]),
            
            Command("Remove-Item", "Удалить файл или директорию", 
                   ["remove", "item", "rm", "delete", "del"], 
                   ["filesystem", "file", "delete"], 
                   {"dangerous": True}),
            
            Command("Write-Host", "Вывести текст в консоль", 
                   ["write", "host", "echo", "print", "output"], 
                   ["output", "console", "display"]),
            
            Command("Start-Process", "Запустить процесс", 
                   ["start", "process", "run", "execute", "launch"], 
                   ["system", "process", "execution"]),
            
            Command("Stop-Process", "Остановить процесс", 
                   ["stop", "process", "kill", "terminate", "end"], 
                   ["system", "process", "management"], 
                   {"dangerous": True}),
            
            Command("Get-Content", "Получить содержимое файла", 
                   ["get", "content", "cat", "type", "read"], 
                   ["filesystem", "file", "content"]),
            
            # Unix-like commands
            Command("ls", "Список файлов в директории", 
                   ["list", "files", "directory", "ls"], 
                   ["filesystem", "unix", "list"]),
            
            Command("cd", "Сменить директорию", 
                   ["change", "directory", "cd", "navigate"], 
                   ["filesystem", "unix", "navigation"]),
            
            Command("cp", "Копировать файлы", 
                   ["copy", "cp", "duplicate"], 
                   ["filesystem", "unix", "copy"]),
            
            Command("rm", "Удалить файлы", 
                   ["remove", "rm", "delete"], 
                   ["filesystem", "unix", "delete"], 
                   {"dangerous": True}),
            
            Command("grep", "Поиск по шаблону", 
                   ["grep", "search", "pattern", "find"], 
                   ["text", "search", "unix"]),
            
            # Application-specific commands
            Command("create-user", "Создать нового пользователя", 
                   ["create", "user", "add", "new"], 
                   ["user", "management", "create"]),
            
            Command("delete-user", "Удалить пользователя", 
                   ["delete", "user", "remove", "drop"], 
                   ["user", "management", "delete"], 
                   {"dangerous": True}),
            
            Command("list-users", "Показать список пользователей", 
                   ["list", "users", "show", "display"], 
                   ["user", "management", "list"]),
            
            Command("system-status", "Показать статус системы", 
                   ["system", "status", "health", "check"], 
                   ["system", "monitoring", "status"]),
            
            Command("show-logs", "Показать логи системы", 
                   ["show", "logs", "view", "display", "log"], 
                   ["system", "logs", "debugging"]),
            
            Command("help", "Показать справку", 
                   ["help", "assist", "support", "guide"], 
                   ["general", "help"]),
            
            Command("exit", "Выйти из системы", 
                   ["exit", "quit", "close", "bye"], 
                   ["general", "exit"]),
        ]
        
        for cmd in demo_commands:
            self.trie.insert(cmd)
    
    def _setup_readline(self):
        """Настройка readline для автодополнения (только для Unix)"""
        if not READLINE_AVAILABLE:
            return
            
        # Настройка табуляции для автодополнения
        readline.parse_and_bind('tab: complete')
        readline.set_completer(self._completer)
        readline.set_completer_delims(' \t\n;')
        
        # Загрузка истории команд
        history_file = Path.home() / '.semantic_advination_history'
        if history_file.exists():
            try:
                readline.read_history_file(str(history_file))
            except Exception:
                pass
        
        # Устанавливаем функцию для сохранения истории
        import atexit
        atexit.register(readline.write_history_file, str(history_file))
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Функция автодополнения для readline"""
        if not READLINE_AVAILABLE:
            return None
            
        if state == 0:
            # Получаем текущую строку
            line = readline.get_line_buffer()
            cursor_pos = readline.get_endidx()
            
            # Получаем предложения
            result = self.orchestrator.process(line, mode=OrchestrationMode.TAB_COMPLETION)
            
            # Формируем варианты автодополнения
            self._completion_options = []
            
            for suggestion in result.suggestions:
                if suggestion.completion_text and suggestion.completion_text.startswith(text):
                    self._completion_options.append(suggestion.completion_text)
                elif suggestion.command_name.startswith(text):
                    self._completion_options.append(suggestion.command_name[len(text):])
        
        try:
            return self._completion_options[state]
        except (IndexError, AttributeError):
            return None
    
    def run_interactive(self):
        """Запуск интерактивного режима"""
        self._print_banner()
        
        print("Введите команду для автодополнения:")
        print("  • Наберите часть команды и увидите подсказки")
        
        if READLINE_AVAILABLE:
            print("  • Нажмите Tab для автодополнения")
        else:
            print("  • Для автодополнения введите команду и нажмите Enter")
        
        print("  • Введите 'help' для справки")
        print("  • Введите 'exit' для выхода\n")
        
        while True:
            try:
                # Отображаем приглашение
                prompt = self._get_prompt()
                
                # Для Windows используем простой ввод
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Обработка специальных команд
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Выход из системы...")
                    break
                
                elif user_input.lower() in ['help', '?']:
                    self._show_help()
                    continue
                
                elif user_input.lower() in ['clear', 'cls']:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'demo':
                    self._run_demo()
                    continue
                
                # Симуляция Tab для Windows
                if user_input.endswith('\t') and not READLINE_AVAILABLE:
                    user_input = user_input.rstrip('\t')
                    self._handle_tab_windows(user_input)
                    continue
                
                # Обработка обычного ввода
                self._process_input(user_input)
                
                # Сохраняем в историю
                self._add_to_history(user_input)
                
            except KeyboardInterrupt:
                print("\n\nВыход из системы...")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Ошибка: {e}")
    
    def _handle_tab_windows(self, user_input: str):
        """Обработка Tab в Windows (симуляция)"""
        print(f"\nTab нажат для: '{user_input}'")
        
        # Получаем автодополнение
        completion = self.orchestrator.get_tab_completion(user_input)
        
        if completion:
            full_command = user_input + completion
            print(f"Автодополнение: {full_command}")
            
            # Предлагаем выполнить
            execute = input(f"Выполнить '{full_command}'? (y/N): ").strip().lower()
            if execute == 'y':
                self._execute_command(full_command)
        else:
            print("Нет вариантов для автодополнения")
    
    def _get_prompt(self) -> str:
        """Получение строки приглашения"""
        return ">>> "
    
    def _process_input(self, user_input: str):
        """Обработка ввода пользователя"""
        # Получаем предсказания
        result = self.orchestrator.process(user_input)
        
        # Отображаем подсказки (если есть)
        if result.suggestions:
            print("\n" + "="*60)
            print("ПОДСКАЗКИ:")
            print("="*60)
            
            for i, suggestion in enumerate(result.suggestions[:5], 1):
                # Форматируем отображение
                if suggestion.completion_text:
                    if sys.platform == "win32":
                        # На Windows нет цветов ANSI по умолчанию
                        display = f"{user_input}[{suggestion.completion_text}]"
                    else:
                        display = f"{user_input}\033[90m{suggestion.completion_text}\033[0m"
                else:
                    display = suggestion.command_name
                
                # Полоса уверенности
                confidence_bar = "█" * int(suggestion.confidence * 10)
                
                # Выводим
                print(f"{i:2}. {display:40} [{confidence_bar:10}] {suggestion.confidence:.0%}")
                
                if suggestion.command_description:
                    print(f"    {suggestion.command_description}")
            
            print("="*60)
            
            # Предлагаем автодополнение, если есть точное совпадение
            if len(result.suggestions) == 1 and suggestion.completion_text:
                self._offer_autocomplete(user_input, suggestion)
        
        else:
            print("  (нет вариантов для автодополнения)")
        
        print()
    
    def _offer_autocomplete(self, current_input: str, suggestion: Any):
        """Предложение автодополнения"""
        if not suggestion.completion_text:
            return
        
        full_command = current_input + suggestion.completion_text
        
        if READLINE_AVAILABLE:
            print(f"Нажмите Tab для автодополнения до '{full_command}'")
        else:
            response = input(f"Автодополнить до '{full_command}'? (y/N): ").strip().lower()
            
            if response == 'y':
                print(f"Команда: {full_command}")
                
                # Запрашиваем выполнение
                execute = input("Выполнить команду? (y/N): ").strip().lower()
                if execute == 'y':
                    self._execute_command(full_command)
    
    def _execute_command(self, command: str):
        """Выполнение команды (демо-версия)"""
        print(f"[Выполняю: {command}]")
        
        # Запись использования
        self.orchestrator.record_command_execution(command.split()[0])
        
        # Демонстрационное выполнение
        if command.startswith("Get-"):
            print(f"  Информация по команде: {command}")
        elif command.startswith("ls"):
            print("  file1.txt  file2.txt  directory/")
        elif command.startswith("cd "):
            print(f"  Переход в директорию: {command[3:]}")
        elif command == "help":
            self._show_help()
        else:
            print(f"  Команда '{command}' выполнена")
        
        print()
    
    def _add_to_history(self, command: str):
        """Добавление команды в историю"""
        if command and command not in self.history[-10:]:
            self.history.append(command)
            if len(self.history) > 100:
                self.history = self.history[-100:]
    
    def _show_help(self):
        """Отображение справки"""
        help_text = """
СИСТЕМА АВТОДОПОЛНЕНИЯ КОМАНД

Использование:
  • Вводите команды по мере набора
  • Система будет показывать подсказки
  
"""
        if READLINE_AVAILABLE:
            help_text += "  • Нажмите Tab для автодополнения\n"
        else:
            help_text += "  • Введите команду и нажмите Enter для подсказок\n"
        
        help_text += """
Примеры команд для тестирования:
  • Get-          (все Get- команды)
  • Get-P         (Get-Process)
  • ls            (список файлов)
  • create        (create-user)
  • system        (system-status)

Специальные команды:
  • help     - эта справка
  • demo     - запуск демо-сценария
  • history  - история команд
  • clear    - очистка экрана
  • exit     - выход

Система учится на ваших действиях:
  • Часто используемые команды предлагаются первыми
  • Учитывается контекст
  • Поддерживается семантический поиск
        """
        print(help_text)
    
    def _show_history(self):
        """Отображение истории команд"""
        if self.history:
            print("\nПоследние выполненные команды:")
            for i, cmd in enumerate(self.history[-10:], 1):
                print(f"  {i}. {cmd}")
        else:
            print("История команд пуста")
        print()
    
    def _run_demo(self):
        """Запуск демонстрационного сценария"""
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ АВТОДОПОЛНЕНИЯ")
        print("="*60 + "\n")
        
        demo_scenarios = [
            ("Пустой ввод (популярные команды)", ""),
            ("Начало Get- команды", "Get-"),
            ("Автодополнение процесса", "Get-P"),
            ("Семантический поиск", "показать процессы"),
            ("Частичный ввод", "cre"),
            ("Unix команда", "ls"),
            ("Опасная команда", "rm"),
        ]
        
        for description, query in demo_scenarios:
            print(f"\n{description}: '{query}'")
            print("-" * 40)
            
            result = self.orchestrator.process(query)
            
            if result.suggestions:
                for i, suggestion in enumerate(result.suggestions[:3], 1):
                    if suggestion.completion_text:
                        display = f"{query}{suggestion.completion_text}"
                    else:
                        display = suggestion.command_name
                    
                    print(f"  {i}. {display} ({suggestion.confidence:.0%})")
            else:
                print("  (нет вариантов)")
        
        print("\n" + "="*60)
        print("Демонстрация завершена")
        print("="*60 + "\n")
    
    def _print_banner(self):
        """Печать баннера"""
        banner = """
        ╔══════════════════════════════════════════════════════╗
        ║    СИСТЕМА СЕМАНТИЧЕСКОГО АВТОДОПОЛНЕНИЯ КОМАНД     ║
        ║    (подобно PowerShell и Google поиску)             ║
        ╚══════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def run_batch_mode(self, queries: List[str]):
        """Пакетный режим обработки"""
        print(f"\nПакетная обработка {len(queries)} запросов:\n")
        
        results = []
        for query in queries:
            print(f"Запрос: '{query}'")
            
            result = self.orchestrator.process(query, mode=OrchestrationMode.BATCH)
            
            if result.suggestions:
                for suggestion in result.suggestions[:3]:
                    print(f"  • {suggestion.command_name} ({suggestion.confidence:.0%})")
            else:
                print("  (нет вариантов)")
            
            print()
            results.append(result.to_dict())
        
        # Сохранение результатов
        try:
            with open('batch_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print("Результаты сохранены в batch_results.json")
        except Exception as e:
            print(f"Ошибка сохранения результатов: {e}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Система семантического автодополнения команд',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                    # Интерактивный режим
  %(prog)s --demo             # Демонстрационный режим
  %(prog)s --json commands.json  # Загрузка команд из JSON
  %(prog)s --batch "Get-" "ls"   # Пакетная обработка

Демонстрация поведения:
  • Вводите "Get-" и нажмите Enter (увидите подсказки)
  • Вводите "создать юз" и увидите "create-user"
  • Система учится на ваших действиях
        """
    )
    
    parser.add_argument(
        '--json',
        help='Путь к JSON файлу с командами',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--demo',
        help='Запустить демонстрационный сценарий',
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
    
    try:
        # Определение источника данных
        if args.json:
            data_source = "json"
            data_path = args.json
        else:
            data_source = "demo"
            data_path = None
        
        # Создание CLI
        cli = AutocompleteCLI(data_source, data_path)
        
        if not cli.setup_done:
            print("Не удалось инициализировать систему")
            sys.exit(1)
        
        # Выбор режима работы
        if args.demo:
            cli._run_demo()
        elif args.batch:
            cli.run_batch_mode(args.batch)
        else:
            cli.run_interactive()
            
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена")
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()