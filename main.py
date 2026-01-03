### 6. **main.py**
```python
#!/usr/bin/env python3
"""
Основной скрипт для демонстрации работы Semantic Advination Core.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь Python
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import ProductionOrchestrator
from storage.trie_storage import CommandTrie
from adapters.shell_adapter import ShellAdapter


def initialize_system():
    """Инициализирует систему адивинации."""
    print("⚙️  Инициализация Semantic Advination Core...")
    
    # Создаём хранилище
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    storage = CommandTrie(str(data_dir))
    
    # Добавляем тестовые команды, если хранилище пустое
    if storage.metadata["total_commands"] == 0:
        test_commands = [
            {"command": "git commit -m 'message'", "usage_count": 100},
            {"command": "git push origin main", "usage_count": 80},
            {"command": "git pull", "usage_count": 90},
            {"command": "git status", "usage_count": 120},
            {"command": "git branch", "usage_count": 60},
            {"command": "git checkout -b feature", "usage_count": 70},
            {"command": "find . -name '*.py'", "usage_count": 50},
            {"command": "ls -la", "usage_count": 200},
            {"command": "cd ~/projects", "usage_count": 150},
            {"command": "mkdir new_folder", "usage_count": 40},
            {"command": "docker ps", "usage_count": 70},
            {"command": "docker build -t myapp .", "usage_count": 30},
            {"command": "python main.py", "usage_count": 60},
            {"command": "pip install -r requirements.txt", "usage_count": 80},
            {"command": "echo 'Hello World'", "usage_count": 100},
        ]
        
        for cmd in test_commands:
            storage.insert(cmd)
        
        print(f"✅ Добавлено {len(test_commands)} тестовых команд")
    
    # Создаём оркестратор
    orchestrator = ProductionOrchestrator()
    orchestrator.adivinator.storage = storage
    
    # Создаём адаптер
    adapter = ShellAdapter(orchestrator)
    
    print(f"✅ Система инициализирована")
    print(f"   Всего команд в базе: {storage.metadata['total_commands']}")
    print(f"   Последнее обновление: {storage.metadata['last_updated']}")
    
    return adapter, storage


def demo_mode(adapter, storage):
    """Запускает демонстрационный режим."""
    print("\n" + "="*60)
    print("🎮 ДЕМО-РЕЖИМ")
    print("="*60)
    print("Доступные команды для тестирования:")
    print("  git com      - автодополнение git commit")
    print("  git pu       - автодополнение git push")
    print("  find *       - поиск файлов")
    print("  ls           - список файлов")
    print("  docker       - docker команды")
    print("  exit/quit    - выход")
    print("  help         - помощь")
    print("  stats        - статистика")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🔹 Введите команду: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("👋 До свидания!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'stats':
                stats = storage.get_stats()
                print(f"\n📊 Статистика хранилища:")
                print(f"   Всего команд: {stats['total_commands']}")
                print(f"   Всего использований: {stats.get('total_usage', 0)}")
                print(f"   Среднее использование: {stats.get('avg_usage_per_command', 0):.1f}")
                print(f"   Уникальных префиксов: {stats.get('unique_prefixes', 0)}")
                continue
            elif not user_input:
                continue
            
            # Обработка ввода
            result = adapter.process_input(user_input, {"domain": "shell"})
            
            # Вывод результата
            print_result(result)
            
            # Если начался диалог, обрабатываем его
            if result.outcome.value == "START_DIALOG":
                handle_dialog(adapter.orchestrator, result.dialog_id)
        
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def print_help():
    """Выводит справку."""
    print("\n📖 Справка:")
    print("  Система поддерживает четыре типа результатов:")
    print("  1. ✅ SUGGEST_EXACT - точное совпадение")
    print("  2. 🔄 SUGGEST_ADAPTED - адаптированное предложение")
    print("  3. 💬 START_DIALOG - требуется уточнение через диалог")
    print("  4. ⏳ DEFER - задача отложена")
    print("\n  Примеры диалога:")
    print("  Ввод: 'найди файлы'")
    print("  Система: 'В какой папке искать?'")
    print("  Ответ: 'в домашней папке'")
    print("  Результат: 'find ~/ -name \"*\"'")


def print_result(result):
    """Выводит результат обработки."""
    print(f"\n📋 Результат: {result.outcome.value}")
    
    if result.suggestions:
        print("📝 Предложения:")
        for i, suggestion in enumerate(result.suggestions[:3], 1):
            confidence = suggestion.match_score * 100
            print(f"  {i}. {suggestion.text} ({confidence:.0f}%)")
    
    if result.first_question:
        print(f"❓ Вопрос: {result.first_question}")
    
    if result.reason and result.outcome.value == "DEFER":
        print(f"⏳ Отложено: {result.reason}")
        if result.task_id:
            print(f"   ID задачи: {result.task_id}")


def handle_dialog(orchestrator, dialog_id):
    """Обрабатывает диалог с пользователем."""
    print("\n💬 Диалог композиции:")
    
    while True:
        try:
            answer = input("➤ Ваш ответ: ").strip()
            
            if answer.lower() in ['cancel', 'отмена']:
                print("Диалог отменён")
                break
            
            result = orchestrator.continue_dialog(dialog_id, answer)
            
            if result.outcome.value == "SUGGEST_EXACT":
                print(f"✅ Команда скомпонована: {result.suggestions[0].text}")
                break
            elif result.outcome.value == "START_DIALOG":
                print(f"❓ Следующий вопрос: {result.first_question}")
            elif result.outcome.value == "DEFER":
                print(f"⏳ Диалог прерван: {result.reason}")
                break
            else:
                print(f"❓ Вопрос: {result.first_question}")
        
        except KeyboardInterrupt:
            print("\nДиалог прерван")
            break
        except Exception as e:
            print(f"❌ Ошибка в диалоге: {e}")
            break


def test_scenarios(adapter):
    """Запускает тестовые сценарии."""
    print("\n🧪 ТЕСТОВЫЕ СЦЕНАРИИ")
    print("="*60)
    
    test_cases = [
        ("git com", {"domain": "git"}),
        ("git pu", {"domain": "git"}),
        ("find *", {"domain": "shell"}),
        ("ls", {"domain": "shell"}),
        ("найди файлы", {"domain": "shell"}),
        ("создай папку", {"domain": "shell"}),
        ("неизвестная команда", {"domain": "unknown"}),
    ]
    
    for prefix, context in test_cases:
        print(f"\nТест: '{prefix}'")
        result = adapter.process_input(prefix, context)
        print_result(result)
        
        # Небольшая пауза для читаемости
        import time
        time.sleep(0.5)


def main():
    """Основная функция."""
    print("="*60)
    print("🧠 SEMANTIC ADVINATION CORE")
    print("="*60)
    
    # Инициализация
    adapter, storage = initialize_system()
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_scenarios(adapter)
            return
        elif sys.argv[1] == "stats":
            stats = storage.get_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            return
        elif sys.argv[1] == "import" and len(sys.argv) > 2:
            # Импорт команд из файла
            import json
            with open(sys.argv[2], 'r', encoding='utf-8') as f:
                commands = json.load(f)
                for cmd in commands:
                    storage.insert(cmd)
            print(f"Импортировано {len(commands)} команд")
            return
    
    # Запуск демо-режима
    demo_mode(adapter, storage)
    
    # Вывод метрик при завершении
    metrics = adapter.orchestrator.get_metrics()
    print(f"\n📈 Метрики сессии:")
    print(f"   Всего запросов: {metrics.get('requests_total', 0)}")
    print(f"   Точных совпадений: {metrics.get('advination_results', {}).get('FOUND', 0)}")
    print(f"   Частичных совпадений: {metrics.get('advination_results', {}).get('PARTIAL', 0)}")
    print(f"   Диалогов: {metrics.get('outcomes', {}).get('START_DIALOG', 0)}")
    print(f"   Отложенных задач: {metrics.get('outcomes', {}).get('DEFER', 0)}")


if __name__ == "__main__":
    main()