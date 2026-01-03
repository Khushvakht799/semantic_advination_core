# build-project.ps1
# Скрипт сборки semantic_advination_core

Write-Host "=== Сборка Semantic Advination Core ===" -ForegroundColor Cyan

# Создаем все файлы
& "$PSScriptRoot\create-core-files.ps1"
& "$PSScriptRoot\create-storage-files.ps1"
& "$PSScriptRoot\create-adapters-files.ps1"
& "$PSScriptRoot\create-config-files.ps1"
& "$PSScriptRoot\create-interface-files.ps1"
& "$PSScriptRoot\create-utils-files.ps1"
& "$PSScriptRoot\create-root-files.ps1"

Write-Host "`n=== Установка зависимостей ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n=== Создание тестовой БД ===" -ForegroundColor Cyan
python -c "
from storage.trie_storage import CommandTrie
trie = CommandTrie()
# Добавляем тестовые команды
test_commands = [
    {'command': 'git commit -m', 'usage_count': 100},
    {'command': 'git push origin main', 'usage_count': 80},
    {'command': 'git pull', 'usage_count': 90},
    {'command': 'find . -name *.py', 'usage_count': 50},
    {'command': 'ls -la', 'usage_count': 200},
    {'command': 'docker ps', 'usage_count': 70},
    {'command': 'python main.py', 'usage_count': 60},
]
for cmd in test_commands:
    trie.insert(cmd)
print('✅ Тестовая БД создана с 7 командами')
"

Write-Host "`n=== Проверка сборки ===" -ForegroundColor Cyan
python -c "
try:
    from core.adivinator import Adivinator
    from storage.trie_storage import CommandTrie
    print('✅ Импорт модулей успешен')
    
    trie = CommandTrie()
    adv = Adivinator(trie)
    print('✅ Adivinator инициализирован')
    
    print('`n=== Быстрая проверка работы ===')
    result = adv.advinate('git com')
    print(f'Результат для git com: {result.result_type}')
    
except Exception as e:
    print(f'❌ Ошибка: {e}')
"

Write-Host "`n🎉 Проект собран успешно!" -ForegroundColor Green
Write-Host "Запустите: python main.py" -ForegroundColor Yellow