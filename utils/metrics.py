# utils\metrics.py
"""
Модуль метрик для отслеживания производительности и качества работы системы.
Сбор, агрегация и экспорт метрик semantic_advination_core.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import json
import statistics
import logging
from contextlib import contextmanager


class MetricType(Enum):
    """Типы метрик."""
    COUNTER = "counter"          # Счетчик (только увеличивается)
    GAUGE = "gauge"              # Измеритель (может увеличиваться и уменьшаться)
    HISTOGRAM = "histogram"      # Гистограмма (распределение значений)
    TIMER = "timer"              # Таймер (измерение времени)


class AggregationMethod(Enum):
    """Методы агрегации метрик."""
    SUM = "sum"                   # Сумма значений
    AVG = "avg"                   # Среднее значение
    MIN = "min"                   # Минимальное значение
    MAX = "max"                   # Максимальное значение
    COUNT = "count"              # Количество значений
    PERCENTILE = "percentile"    # Процентиль
    RATE = "rate"                # Скорость изменения


@dataclass
class MetricDefinition:
    """Определение метрики."""
    
    name: str                    # Имя метрики
    metric_type: MetricType      # Тип метрики
    description: str = ""        # Описание метрики
    unit: str = ""               # Единица измерения (ms, count, %, и т.д.)
    labels: List[str] = field(default_factory=list)  # Метки для многомерных метрик
    aggregation_period: int = 60  # Период агрегации в секундах
    retention_days: int = 30     # Хранение данных (дней)
    
    def get_key(self, label_values: Dict[str, str] = None) -> str:
        """Генерирует ключ для метрики с метками."""
        if not self.labels or not label_values:
            return self.name
        
        # Сортируем метки для консистентности
        sorted_labels = sorted(label_values.items())
        label_str = "_".join(f"{k}:{v}" for k, v in sorted_labels)
        return f"{self.name}[{label_str}]"


@dataclass
class MetricValue:
    """Значение метрики."""
    
    metric_name: str            # Имя метрики
    value: float                # Значение
    timestamp: datetime         # Временная метка
    labels: Dict[str, str] = field(default_factory=dict)  # Метки
    metadata: Dict[str, Any] = field(default_factory=dict)  # Дополнительные данные
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'metadata': self.metadata
        }


@dataclass
class AggregatedMetric:
    """Агрегированная метрика."""
    
    metric_name: str            # Имя метрики
    aggregation_method: AggregationMethod  # Метод агрегации
    value: float                # Агрегированное значение
    timestamp: datetime         # Временная метка (начало периода)
    period_seconds: int         # Длительность периода
    samples: int = 0            # Количество образцов
    labels: Dict[str, str] = field(default_factory=dict)  # Метки
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь."""
        return {
            'metric_name': self.metric_name,
            'aggregation_method': self.aggregation_method.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'period_seconds': self.period_seconds,
            'samples': self.samples,
            'labels': self.labels
        }


class MetricsCollector:
    """
    Коллектор метрик для сбора и агрегации метрик производительности.
    """
    
    def __init__(self, namespace: str = "semantic_advination"):
        """
        Инициализация коллектора метрик.
        
        Args:
            namespace: Пространство имен для метрик
        """
        self.namespace = namespace
        self.metrics: Dict[str, MetricDefinition] = {}
        self.values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Автоматические метрики
        self._register_default_metrics()
        
        # Запуск фоновой задачи агрегации
        self.running = False
        self.aggregator_thread = None
        
        # Колбэки для экспорта метрик
        self.export_callbacks: List[Callable] = []
    
    def _register_default_metrics(self):
        """Регистрирует стандартные метрики системы."""
        default_metrics = [
            # Метрики выполнения команд
            MetricDefinition(
                name="commands_executed_total",
                metric_type=MetricType.COUNTER,
                description="Общее количество выполненных команд",
                unit="count",
                labels=["command", "status", "adapter"]
            ),
            MetricDefinition(
                name="command_execution_time",
                metric_type=MetricType.HISTOGRAM,
                description="Время выполнения команд",
                unit="ms",
                labels=["command"]
            ),
            MetricDefinition(
                name="command_confidence_score",
                metric_type=MetricType.GAUGE,
                description="Уверенность системы в команде",
                unit="percent",
                labels=["command"]
            ),
            
            # Метрики качества
            MetricDefinition(
                name="command_success_rate",
                metric_type=MetricType.GAUGE,
                description="Процент успешных выполнений",
                unit="percent"
            ),
            MetricDefinition(
                name="command_recognition_accuracy",
                metric_type=MetricType.GAUGE,
                description="Точность распознавания команд",
                unit="percent"
            ),
            
            # Метрики производительности
            MetricDefinition(
                name="system_latency",
                metric_type=MetricType.HISTOGRAM,
                description="Задержка системы",
                unit="ms"
            ),
            MetricDefinition(
                name="memory_usage",
                metric_type=MetricType.GAUGE,
                description="Использование памяти",
                unit="MB"
            ),
            MetricDefinition(
                name="cpu_usage",
                metric_type=MetricType.GAUGE,
                description="Использование CPU",
                unit="percent"
            ),
            
            # Метрики использования
            MetricDefinition(
                name="active_users",
                metric_type=MetricType.GAUGE,
                description="Активные пользователи",
                unit="count"
            ),
            MetricDefinition(
                name="requests_per_second",
                metric_type=MetricType.GAUGE,
                description="Запросов в секунду",
                unit="rps"
            ),
            
            # Метрики хранилища
            MetricDefinition(
                name="database_query_time",
                metric_type=MetricType.HISTOGRAM,
                description="Время выполнения запросов к БД",
                unit="ms"
            ),
            MetricDefinition(
                name="cache_hit_rate",
                metric_type=MetricType.GAUGE,
                description="Процент попаданий в кэш",
                unit="percent"
            ),
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def register_metric(self, metric: MetricDefinition):
        """Регистрирует новую метрику."""
        with self.lock:
            full_name = f"{self.namespace}_{metric.name}"
            metric.name = full_name  # Обновляем имя с namespace
            self.metrics[full_name] = metric
            self.logger.debug(f"Зарегистрирована метрика: {full_name}")
    
    def increment_counter(self, 
                         metric_name: str, 
                         value: float = 1.0,
                         labels: Dict[str, str] = None,
                         metadata: Dict[str, Any] = None):
        """
        Увеличивает счетчик.
        
        Args:
            metric_name: Имя метрики (без namespace)
            value: Значение для увеличения
            labels: Метки для многомерной метрики
            metadata: Дополнительные метаданные
        """
        full_name = f"{self.namespace}_{metric_name}"
        
        with self.lock:
            if full_name not in self.metrics:
                self.logger.warning(f"Метрика {full_name} не зарегистрирована")
                return
            
            metric_def = self.metrics[full_name]
            
            if metric_def.metric_type != MetricType.COUNTER:
                self.logger.warning(f"Метрика {full_name} не является счетчиком")
                return
            
            metric_value = MetricValue(
                metric_name=full_name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            key = metric_def.get_key(labels)
            self.values[key].append(metric_value)
    
    def set_gauge(self, 
                  metric_name: str, 
                  value: float,
                  labels: Dict[str, str] = None,
                  metadata: Dict[str, Any] = None):
        """
        Устанавливает значение измерителя.
        
        Args:
            metric_name: Имя метрики (без namespace)
            value: Значение измерителя
            labels: Метки для многомерной метрики
            metadata: Дополнительные метаданные
        """
        full_name = f"{self.namespace}_{metric_name}"
        
        with self.lock:
            if full_name not in self.metrics:
                self.logger.warning(f"Метрика {full_name} не зарегистрирована")
                return
            
            metric_def = self.metrics[full_name]
            
            if metric_def.metric_type != MetricType.GAUGE:
                self.logger.warning(f"Метрика {full_name} не является измерителем")
                return
            
            metric_value = MetricValue(
                metric_name=full_name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            key = metric_def.get_key(labels)
            self.values[key].append(metric_value)
    
    def record_histogram(self, 
                        metric_name: str, 
                        value: float,
                        labels: Dict[str, str] = None,
                        metadata: Dict[str, Any] = None):
        """
        Записывает значение в гистограмму.
        
        Args:
            metric_name: Имя метрики (без namespace)
            value: Значение для записи
            labels: Метки для многомерной метрики
            metadata: Дополнительные метаданные
        """
        full_name = f"{self.namespace}_{metric_name}"
        
        with self.lock:
            if full_name not in self.metrics:
                self.logger.warning(f"Метрика {full_name} не зарегистрирована")
                return
            
            metric_def = self.metrics[full_name]
            
            if metric_def.metric_type not in [MetricType.HISTOGRAM, MetricType.TIMER]:
                self.logger.warning(f"Метрика {full_name} не является гистограммой или таймером")
                return
            
            metric_value = MetricValue(
                metric_name=full_name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            key = metric_def.get_key(labels)
            self.values[key].append(metric_value)
    
    @contextmanager
    def timer(self, 
              metric_name: str,
              labels: Dict[str, str] = None,
              metadata: Dict[str, Any] = None):
        """
        Контекстный менеджер для измерения времени выполнения.
        
        Args:
            metric_name: Имя метрики (без namespace)
            labels: Метки для многомерной метрики
            metadata: Дополнительные метаданные
        
        Пример:
            with metrics.timer("command_execution_time", labels={"command": "create_project"}):
                execute_command()
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.record_histogram(metric_name, elapsed_ms, labels, metadata)
    
    def record_command_execution(self,
                                command_name: str,
                                execution_time_ms: float,
                                success: bool = True,
                                confidence_score: float = 1.0,
                                adapter_type: str = "unknown",
                                user_id: str = None):
        """
        Записывает метрики выполнения команды.
        """
        labels = {
            "command": command_name,
            "status": "success" if success else "failed",
            "adapter": adapter_type
        }
        
        # Увеличиваем счетчик выполненных команд
        self.increment_counter("commands_executed_total", labels=labels)
        
        # Записываем время выполнения
        self.record_histogram("command_execution_time", 
                             execution_time_ms, 
                             labels={"command": command_name})
        
        # Записываем уверенность
        self.set_gauge("command_confidence_score", 
                      confidence_score * 100, 
                      labels={"command": command_name})
        
        # Обновляем метрику успешности
        if success:
            self.increment_counter("successful_commands", labels={"command": command_name})
        else:
            self.increment_counter("failed_commands", labels={"command": command_name})
    
    def _aggregate_metrics(self):
        """Фоновая задача для агрегации метрик."""
        self.logger.info("Запущена агрегация метрик")
        
        while self.running:
            try:
                with self.lock:
                    current_time = datetime.now()
                    
                    for metric_name, metric_def in self.metrics.items():
                        # Для каждой комбинации меток
                        for key in list(self.values.keys()):
                            if not key.startswith(metric_name):
                                continue
                            
                            values_list = self.values[key]
                            if not values_list:
                                continue
                            
                            # Получаем метки из ключа
                            labels = {}
                            if "[" in key and "]" in key:
                                label_part = key.split("[", 1)[1].rstrip("]")
                                for pair in label_part.split("_"):
                                    if ":" in pair:
                                        k, v = pair.split(":", 1)
                                        labels[k] = v
                            
                            # Фильтруем значения за последний период
                            period_start = current_time - timedelta(seconds=metric_def.aggregation_period)
                            recent_values = [
                                v for v in values_list 
                                if v.timestamp >= period_start
                            ]
                            
                            if not recent_values:
                                continue
                            
                            # Агрегируем в зависимости от типа метрики
                            if metric_def.metric_type == MetricType.COUNTER:
                                # Для счетчиков считаем сумму
                                total = sum(v.value for v in recent_values)
                                aggregated = AggregatedMetric(
                                    metric_name=metric_name,
                                    aggregation_method=AggregationMethod.SUM,
                                    value=total,
                                    timestamp=period_start,
                                    period_seconds=metric_def.aggregation_period,
                                    samples=len(recent_values),
                                    labels=labels
                                )
                                self.aggregated[key].append(aggregated)
                            
                            elif metric_def.metric_type == MetricType.GAUGE:
                                # Для измерителей берем последнее значение
                                last_value = recent_values[-1].value
                                aggregated = AggregatedMetric(
                                    metric_name=metric_name,
                                    aggregation_method=AggregationMethod.AVG,
                                    value=last_value,
                                    timestamp=period_start,
                                    period_seconds=metric_def.aggregation_period,
                                    samples=len(recent_values),
                                    labels=labels
                                )
                                self.aggregated[key].append(aggregated)
                            
                            elif metric_def.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                                # Для гистограмм и таймеров вычисляем несколько статистик
                                values = [v.value for v in recent_values]
                                
                                # Среднее значение
                                avg_metric = AggregatedMetric(
                                    metric_name=metric_name,
                                    aggregation_method=AggregationMethod.AVG,
                                    value=statistics.mean(values) if values else 0,
                                    timestamp=period_start,
                                    period_seconds=metric_def.aggregation_period,
                                    samples=len(values),
                                    labels=labels
                                )
                                self.aggregated[f"{key}_avg"].append(avg_metric)
                                
                                # 95-й процентиль
                                if len(values) >= 5:
                                    p95 = statistics.quantiles(values, n=100)[94]
                                    percentile_metric = AggregatedMetric(
                                        metric_name=metric_name,
                                        aggregation_method=AggregationMethod.PERCENTILE,
                                        value=p95,
                                        timestamp=period_start,
                                        period_seconds=metric_def.aggregation_period,
                                        samples=len(values),
                                        labels=labels
                                    )
                                    self.aggregated[f"{key}_p95"].append(percentile_metric)
                
                # Вызываем колбэки экспорта
                for callback in self.export_callbacks:
                    try:
                        callback(self)
                    except Exception as e:
                        self.logger.error(f"Ошибка в колбэке экспорта метрик: {e}")
                
                # Очищаем старые агрегированные данные
                self._cleanup_old_data()
                
                # Пауза между агрегациями
                time.sleep(60)  # Агрегируем каждую минуту
                
            except Exception as e:
                self.logger.error(f"Ошибка при агрегации метрик: {e}")
                time.sleep(5)
    
    def _cleanup_old_data(self):
        """Очищает старые данные метрик."""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        with self.lock:
            # Очищаем агрегированные данные
            for key in list(self.aggregated.keys()):
                self.aggregated[key] = [
                    m for m in self.aggregated[key] 
                    if m.timestamp >= cutoff_time
                ]
                
                # Удаляем пустые списки
                if not self.aggregated[key]:
                    del self.aggregated[key]
    
    def start(self):
        """Запускает сбор и агрегацию метрик."""
        if self.running:
            return
        
        self.running = True
        self.aggregator_thread = threading.Thread(
            target=self._aggregate_metrics,
            daemon=True,
            name="MetricsAggregator"
        )
        self.aggregator_thread.start()
        self.logger.info("Коллектор метрик запущен")
    
    def stop(self):
        """Останавливает сбор метрик."""
        self.running = False
        if self.aggregator_thread:
            self.aggregator_thread.join(timeout=5)
        self.logger.info("Коллектор метрик остановлен")
    
    def get_metric_values(self, 
                         metric_name: str,
                         labels: Dict[str, str] = None,
                         limit: int = 100) -> List[MetricValue]:
        """
        Получает сырые значения метрики.
        """
        full_name = f"{self.namespace}_{metric_name}"
        key = self.metrics.get(full_name, MetricDefinition(name=full_name, metric_type=MetricType.COUNTER)).get_key(labels)
        
        with self.lock:
            values = list(self.values.get(key, deque()))[-limit:]
            return values
    
    def get_aggregated_metrics(self,
                              metric_name: str = None,
                              labels: Dict[str, str] = None,
                              start_time: datetime = None,
                              end_time: datetime = None,
                              aggregation_method: AggregationMethod = None) -> List[AggregatedMetric]:
        """
        Получает агрегированные метрики.
        """
        with self.lock:
            results = []
            
            for key, metrics_list in self.aggregated.items():
                # Фильтруем по имени метрики
                if metric_name and metric_name not in key:
                    continue
                
                for metric in metrics_list:
                    # Фильтруем по времени
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    
                    # Фильтруем по методу агрегации
                    if aggregation_method and metric.aggregation_method != aggregation_method:
                        continue
                    
                    # Фильтруем по меткам
                    if labels and not all(metric.labels.get(k) == v for k, v in labels.items()):
                        continue
                    
                    results.append(metric)
            
            # Сортируем по времени
            results.sort(key=lambda x: x.timestamp)
            return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Возвращает метрики здоровья системы.
        """
        with self.lock:
            health = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Проверяем основные метрики
            try:
                # Успешность выполнения команд (последние 5 минут)
                recent_success = self.get_aggregated_metrics(
                    metric_name="success_rate",
                    start_time=datetime.now() - timedelta(minutes=5)
                )
                
                if recent_success:
                    success_rate = recent_success[-1].value if recent_success else 100
                    health["metrics"]["success_rate"] = success_rate
                    
                    if success_rate < 90:  # Порог 90%
                        health["status"] = "degraded"
                        health["issues"] = [f"Низкая успешность выполнения команд: {success_rate}%"]
                
                # Задержка системы (95-й процентиль)
                latency_p95 = self.get_aggregated_metrics(
                    metric_name="system_latency_p95",
                    start_time=datetime.now() - timedelta(minutes=5)
                )
                
                if latency_p95:
                    latency = latency_p95[-1].value if latency_p95 else 0
                    health["metrics"]["latency_p95_ms"] = latency
                    
                    if latency > 1000:  # Порог 1 секунда
                        health["status"] = "degraded"
                        if "issues" not in health:
                            health["issues"] = []
                        health["issues"].append(f"Высокая задержка системы: {latency}ms")
                
                # Использование памяти
                memory_usage = self.get_metric_values("memory_usage", limit=1)
                if memory_usage:
                    memory = memory_usage[-1].value if memory_usage else 0
                    health["metrics"]["memory_usage_mb"] = memory
                    
                    if memory > 500:  # Порог 500 MB
                        health["status"] = "warning"
                
            except Exception as e:
                health["status"] = "unknown"
                health["error"] = str(e)
            
            return health
    
    def register_export_callback(self, callback: Callable):
        """
        Регистрирует колбэк для экспорта метрик.
        
        Args:
            callback: Функция, которая будет вызываться с коллектором метрик
        """
        self.export_callbacks.append(callback)
    
    def export_prometheus(self) -> str:
        """
        Экспортирует метрики в формате Prometheus.
        
        Returns:
            Строка в формате Prometheus
        """
        lines = []
        current_time = datetime.now().timestamp() * 1000
        
        with self.lock:
            # Экспортируем текущие значения
            for key, values_deque in self.values.items():
                if not values_deque:
                    continue
                
                latest_value = values_deque[-1]
                metric_def = self.metrics.get(latest_value.metric_name)
                
                if not metric_def:
                    continue
                
                # Формируем строку меток
                labels_str = ""
                if latest_value.labels:
                    labels = [f'{k}="{v}"' for k, v in sorted(latest_value.labels.items())]
                    labels_str = "{" + ",".join(labels) + "}"
                
                # Формируем строку метрики
                metric_line = f'{metric_def.name}{labels_str} {latest_value.value} {int(current_time)}'
                lines.append(metric_line)
                
                # Добавляем HELP и TYPE для первой встречи метрики
                if metric_def.description:
                    lines.insert(0, f'# HELP {metric_def.name} {metric_def.description}')
                lines.insert(1, f'# TYPE {metric_def.name} {metric_def.metric_type.value}')
        
        return "\n".join(lines)
    
    def export_json(self, include_raw: bool = False) -> Dict[str, Any]:
        """
        Экспортирует метрики в формате JSON.
        
        Args:
            include_raw: Включать ли сырые значения
            
        Returns:
            Словарь с метриками
        """
        result = {
            "namespace": self.namespace,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "aggregated": {},
            "health": self.get_system_health()
        }
        
        with self.lock:
            # Агрегированные метрики
            for key, metrics_list in self.aggregated.items():
                if metrics_list:
                    result["aggregated"][key] = [
                        m.to_dict() for m in metrics_list[-10:]  # Последние 10 значений
                    ]
            
            # Текущие значения
            if include_raw:
                for key, values_deque in self.values.items():
                    if values_deque:
                        latest = values_deque[-1]
                        result["metrics"][key] = latest.to_dict()
        
        return result


# Глобальный экземпляр для удобства использования
global_metrics = MetricsCollector()


# Декоратор для измерения времени выполнения функции
def measure_time(metric_name: str, labels: Dict[str, str] = None):
    """
    Декоратор для измерения времени выполнения функции.
    
    Пример:
        @measure_time("function_execution_time", labels={"function": "process_data"})
        def process_data():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with global_metrics.timer(metric_name, labels=labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Пример использования
if __name__ == "__main__":
    import logging
    import random
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем коллектор
    metrics = MetricsCollector()
    
    # Запускаем сбор метрик
    metrics.start()
    
    print("Коллектор метрик запущен. Симулируем выполнение команд...")
    
    # Симулируем выполнение команд
    commands = ["create_project", "delete_file", "start_server", "stop_service", "show_status"]
    
    for i in range(50):
        command = random.choice(commands)
        execution_time = random.uniform(10, 500)  # 10-500 ms
        success = random.random() > 0.1  # 90% успеха
        confidence = random.uniform(0.7, 1.0)
        
        # Записываем метрики выполнения
        metrics.record_command_execution(
            command_name=command,
            execution_time_ms=execution_time,
            success=success,
            confidence_score=confidence,
            adapter_type="CLI"
        )
        
        # Пример с таймером
        with metrics.timer("custom_operation", labels={"operation": "simulation"}):
            time.sleep(random.uniform(0.01, 0.1))
        
        # Устанавливаем измеритель
        metrics.set_gauge("simulation_progress", (i + 1) * 2)  # 2% за итерацию
        
        time.sleep(0.1)
    
    # Даем время на агрегацию
    time.sleep(2)
    
    # Получаем метрики здоровья
    health = metrics.get_system_health()
    print(f"Здоровье системы: {health['status']}")
    
    # Экспортируем в Prometheus формате
    prometheus_data = metrics.export_prometheus()
    print(f"\nPrometheus метрики ({len(prometheus_data.splitlines())} строк):")
    print(prometheus_data[:500] + "..." if len(prometheus_data) > 500 else prometheus_data)
    
    # Получаем агрегированные метрики
    aggregated = metrics.get_aggregated_metrics(
        metric_name="command_execution_time_avg",
        start_time=datetime.now() - timedelta(minutes=10)
    )
    
    print(f"\nАгрегированные метрики времени выполнения: {len(aggregated)} записей")
    
    # Останавливаем коллектор
    metrics.stop()
    print("\nКоллектор метрик остановлен")