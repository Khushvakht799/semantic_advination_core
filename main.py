from core.orchestrator import ProductionOrchestrator
from storage.trie_storage import CommandTrie
from adapters.shell_adapter import ShellAdapter

def main():
    # Инициализация
    storage = CommandTrie("data/commands.db")
    orchestrator = ProductionOrchestrator(storage)
    adapter = ShellAdapter(orchestrator)
    
    # Демо
    print("=== Semantic Advination Core ===")
    while True:
        try:
            user_input = input("\nВведите команду (или 'exit'): ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Обработка
            result = adapter.process_input(user_input)
            
            # Вывод результата
            if result.outcome == "SUGGEST_EXACT":
                print(f"✅ Найдено: {result.suggestions[0].text}")
            elif result.outcome == "START_DIALOG":
                print(f"❓ {result.first_question}")
                # ... продолжение диалога
            elif result.outcome == "DEFER":
                print(f"⏸️  Отложено: {result.reason}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️  Ошибка: {e}")

if __name__ == "__main__":
    main()