# Semantic Advination Core

Система семантического предсказания команд на основе префиксного дерева (Trie).

## Установка

```bash
git clone https://github.com/Khushvakht799/semantic_advination_core.git
cd semantic_advination_core
python -m pip install -e .

Использование
CLI интерфейс
# Интерактивный режим (по умолчанию)
python main.py

# С тестовыми данными из JSON
python main.py --json commands.json

# Пакетная обработка
python main.py --batch "create user" "delete file" "system status"

# Тестовый режим
python main.py --test
Пример сессии
>>> создать пользователя
РЕЗУЛЬТАТ: suggest_exact
============================================================
Запрос: создать пользователя
Доверие: 100.00%
Предложений: 1

ПРЕДЛОЖЕНИЯ:

1. create_user
   Доверие: 100.00% [██████████]
   Описание: Создание нового пользователя
   Совпавшие токены: create, user
   Теги: admin, user_management, create
============================================================
Структура проекта

    storage/ - хранилище команд на основе Trie

    core/ - основные компоненты системы

    main.py - CLI интерфейс
Компоненты системы

    CommandTrie - префиксное дерево для хранения команд

    Adivinator - семантический поиск команд

    Validator - валидация и адаптация результатов

    Orchestrator - координация workflow системы
## 7. Структура проекта после исправлений:
semantic_advination_core/
│
├── storage/
│ ├── init.py # Исправленный
│ └── trie_storage.py # Наш существующий файл
│
├── core/
│ ├── init.py # Новый файл
│ ├── adivinator.py # Наш существующий файл
│ ├── validator.py # Наш существующий файл
│ └── orchestrator.py # Наш существующий файл
│
├── init.py # Новый файл (корневой)
├── main.py # Исправленный
├── setup.py # Новый файл
├── README.md # Новый файл
└── commands.json # Пример файла с командами
Теперь система должна работать корректно. Убедитесь, что у вас есть все исправленные файлы и запустите:

```bash
python main.py
