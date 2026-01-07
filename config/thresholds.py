# config\thresholds.py
"""
Пороговые значения для системы semantic_advination_core.
Все числовые параметры, определяющие поведение системы.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class Thresholds:
    """
    Класс, содержащий все пороговые значения системы.
    """
    
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
    
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> 'Thresholds':
        """
        Загружает пороговые значения из YAML файла.
        Если файл не указан, использует значения по умолчанию.
        """
        thresholds = cls()
        
        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                
                if yaml_data and 'thresholds' in yaml_data:
                    # Обновляем значения из YAML
                    yaml_thresholds = yaml_data['thresholds']
                    
                    for key, value in yaml_thresholds.items():
                        if hasattr(thresholds, key):
                            setattr(thresholds, key, value)
                        else:
                            print(f"Предупреждение: неизвестный параметр в YAML: {key}")
            except Exception as e:
                print(f"Ошибка загрузки YAML конфигурации: {e}")
        
        return thresholds
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует пороговые значения в словарь."""
        return {
            'similarity': {
                'min_match': self.similarity_min_match,
                'high_confidence': self.similarity_high_confidence,
                'fuzzy_match': self.similarity_fuzzy_match
            },
            'tokenization': {
                'min_length': self.token_min_length,
                'max_length': self.token_max_length,
                'max_per_command': self.max_tokens_per_command
            },
            'trie_storage': {
                'max_suggestions': self.trie_max_suggestions,
                'cache_size': self.trie_cache_size,
                'min_prefix_length': self.trie_min_prefix_length
            },
            'execution': {
                'timeout_seconds': self.execution_timeout_seconds,
                'max_retries': self.execution_max_retries,
                'retry_delay_ms': self.execution_retry_delay_ms
            },
            'deferred_tasks': {
                'max_queue_size': self.deferred_max_queue_size,
                'max_retries': self.deferred_max_retries,
                'timeout_hours': self.deferred_timeout_hours
            },
            'logging': {
                'max_file_size_mb': self.log_max_file_size_mb,
                'backup_count': self.log_backup_count
            },
            'security': {
                'max_requests_per_minute': self.max_requests_per_minute,
                'session_timeout_minutes': self.session_timeout_minutes,
                'password_min_length': self.password_min_length
            },
            'quality': {
                'auto_execute_threshold': self.confidence_min_for_auto_execute,
                'confirmation_threshold': self.confirmation_threshold,
                'learning_min_samples': self.learning_min_samples
            },
            'performance': {
                'cache_ttl': self.cache_ttl_seconds,
                'max_concurrent': self.max_concurrent_commands,
                'memory_limit_mb': self.memory_usage_limit_mb
            }
        }
    
    def validate(self) -> bool:
        """
        Проверяет корректность всех пороговых значений.
        """
        errors = []
        
        # Проверка значений схожести
        if not 0 <= self.similarity_min_match <= 1:
            errors.append(f"similarity_min_match должен быть от 0 до 1, получено: {self.similarity_min_match}")
        
        if not 0 <= self.similarity_high_confidence <= 1:
            errors.append(f"similarity_high_confidence должен быть от 0 до 1, получено: {self.similarity_high_confidence}")
        
        if not 0 <= self.similarity_fuzzy_match <= 1:
            errors.append(f"similarity_fuzzy_match должен быть от 0 до 1, получено: {self.similarity_fuzzy_match}")
        
        if self.similarity_fuzzy_match > self.similarity_min_match:
            errors.append(f"fuzzy_match ({self.similarity_fuzzy_match}) не может быть больше min_match ({self.similarity_min_match})")
        
        # Проверка токенов
        if self.token_min_length < 1:
            errors.append(f"token_min_length должен быть >= 1, получено: {self.token_min_length}")
        
        if self.token_max_length <= self.token_min_length:
            errors.append(f"token_max_length ({self.token_max_length}) должен быть больше token_min_length ({self.token_min_length})")
        
        if self.max_tokens_per_command < 1:
            errors.append(f"max_tokens_per_command должен быть >= 1, получено: {self.max_tokens_per_command}")
        
        # Проверка таймаутов
        if self.execution_timeout_seconds < 1:
            errors.append(f"execution_timeout_seconds должен быть >= 1, получено: {self.execution_timeout_seconds}")
        
        if self.execution_max_retries < 0:
            errors.append(f"execution_max_retries должен быть >= 0, получено: {self.execution_max_retries}")
        
        # Проверка лимитов памяти
        if self.memory_usage_limit_mb < 10:
            errors.append(f"memory_usage_limit_mb должен быть >= 10, получено: {self.memory_usage_limit_mb}")
        
        if errors:
            print("Ошибки валидации пороговых значений:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def __str__(self) -> str:
        """Строковое представление пороговых значений."""
        dict_repr = self.to_dict()
        result = ["Thresholds:"]
        
        for category, values in dict_repr.items():
            result.append(f"\n{category}:")
            for key, value in values.items():
                result.append(f"  {key}: {value}")
        
        return "\n".join(result)


# Глобальный экземпляр для использования по умолчанию
default_thresholds = Thresholds()


# Пример использования
if __name__ == "__main__":
    # Создаем экземпляр с значениями по умолчанию
    thresholds = Thresholds()
    
    print("Пороговые значения по умолчанию:")
    print(thresholds)
    
    # Проверяем валидность
    is_valid = thresholds.validate()
    print(f"\nВалидность: {is_valid}")
    
    # Пример загрузки из YAML (если файл существует)
    yaml_path = "config/thresholds.yaml"
    if os.path.exists(yaml_path):
        thresholds_from_yaml = Thresholds.from_yaml(yaml_path)
        print(f"\nЗагружено из {yaml_path}:")
        print(thresholds_from_yaml)