# simple_main.py (упрощённая версия без readline)
"""
Упрощённая версия CLI без readline для Windows
"""

import sys
import os
from storage.trie_storage import CommandTrie, Command
from core.adivinator import Adivinator
from core.validator import Validator, Context
from core.orchestrator import Orchestrator, OrchestrationMode

def setup_simple_cli():
    """Настройка простого CLI"""
    # Создаём trie с демо-командами
    trie = CommandTrie()
    
    # Базовые демо-команды
    commands = [
        Command("get-process", "Показать процессы", ["get", "process", "ps"], ["system"]),
        Command("get-service", "Показать сервисы", ["get", "service", "svc"], ["system"]),
        Command("create-user", "Создать пользователя", ["create", "user", "add"], ["user"]),
        Command("delete-user", "Удалить пользователя", ["delete", "user", "remove"], ["user"]),
        Command("list-users", "Показать пользователей", ["list", "users", "show"], ["user"]),
        Command("help", "Помощь", ["help", "assist"], ["general"]),
        Command("exit", "Выход", ["exit", "quit"], ["general"]),
    ]
    
    for cmd in commands:
        trie.insert(cmd)
    
    # Создаём компоненты
    adivinator = Adivinator(trie)
    validator = Validator()
    orchestrator = Orchestrator(adivinator, validator)
    
    return orchestrator

def main():
    """Главная функция"""
    print("="*60)
    print("СИСТЕМА АВТОДОПОЛНЕНИЯ КОМАНД")
    print("="*60)
    print("\nВведите часть команды и нажмите Enter")
    print("Примеры: 'get', 'create', 'list', 'help', 'exit'")
    print("-" * 60)
    
    orchestrator = setup_simple_cli()
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Выход...")
                break
                
            if user_input.lower() == 'help':
                print("\nДоступные команды:")
                print("  get-process    - показать процессы")
                print("  get-service    - показать сервисы")
                print("  create-user    - создать пользователя")
                print("  delete-user    - удалить пользователя")
                print("  list-users     - показать пользователей")
                print("  help           - эта справка")
                print("  exit           - выход")
                continue
            
            # Получаем предсказания
            result = orchestrator.process(user_input)
            
            if result.suggestions:
                print(f"\nВарианты для '{user_input}':")
                for i, suggestion in enumerate(result.suggestions, 1):
                    confidence_bar = "█" * int(suggestion.confidence * 10)
                    print(f"{i}. {suggestion.command_name:20} [{confidence_bar:10}] {suggestion.confidence:.0%}")
                    
                    if suggestion.command_description:
                        print(f"   {suggestion.command_description}")
            else:
                print(f"Нет вариантов для '{user_input}'")
                
        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()