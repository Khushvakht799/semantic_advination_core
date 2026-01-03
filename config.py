"""
Конфигурационный файл для проекта semantic_advination_core.
Здесь хранятся пути, параметры моделей и другие настройки.
"""

import os
from pathlib import Path

# ---------- ПУТИ ----------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Создание директорий при отсутствии
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# ---------- NLP МОДЕЛИ ----------
# Модель для лемматизации/морфологического анализа (русский язык)
MORPH_MODEL = "pymorphy3"  # или "pymorphy2"
MORPH_LANG = "ru"

# Модель для синтаксического разбора и NER
SPACY_MODEL = "ru_core_news_lg"  # требуется предварительно скачать: python -m spacy download ru_core_news_lg

# Модель для BERT-эмбеддингов / классификации
TRANSFORMERS_MODEL = "sberbank-ai/ruBert-large"
# Для тональности можно использовать другую:
SENTIMENT_MODEL = "blanchefort/rubert-base-cased-sentiment"

# ---------- ПАРАМЕТРЫ ИЗВЛЕЧЕНИЯ ТРИАД ----------
WINDOW_SIZE = 7  # контекстное окно для поиска отношений
MIN_SUBJECT_LENGTH = 2  # минимальная длина субъекта в словах
MIN_OBJECT_LENGTH = 1   # минимальная длина объекта в словах
RELATION_POS_TAGS = ["VERB", "ADV", "ADJ", "NOUN"]  # возможные части речи для отношения

# ---------- НАСТРОЙКИ ЛОГГИРОВАНИЯ ----------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ---------- ДОПОЛНИТЕЛЬНЫЕ ФИЛЬТРЫ ----------
STOPWORDS_LANG = "russian"
CUSTOM_STOPWORDS = ["это", "вот", "какой", "такой"]

# ---------- НАСТРОЙКИ ДЛЯ JSON ----------
JSON_INDENT = 2
JSON_ENSURE_ASCII = False

# ---------- ПРОЧЕЕ ----------
RANDOM_SEED = 42
MAX_TEXT_LENGTH = 10_000  # максимальная длина текста для обработки (символов)