# utils\serializers.py
"""
Модуль сериализации/десериализации данных для semantic_advination_core.
Поддержка JSON, MessagePack, Pickle и пользовательских форматов.
"""

import json
import pickle
import msgpack
import yaml
import csv
import io
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, get_type_hints
from dataclasses import dataclass, is_dataclass, asdict
from uuid import UUID
import base64
import logging
from pathlib import Path


T = TypeVar('T')

class SerializationFormat(Enum):
    """Поддерживаемые форматы сериализации."""
    JSON = "json"
    MESSAGEPACK = "msgpack"
    PICKLE = "pickle"
    YAML = "yaml"
    CSV = "csv"
    BINARY = "binary"


class SerializationError(Exception):
    """Исключение при сериализации/десериализации."""
    pass


class AdvancedJSONEncoder(json.JSONEncoder):
    """
    Расширенный JSON encoder для поддержки специальных типов.
    """
    
    def default(self, obj: Any) -> Any:
        # Обработка datetime
        if isinstance(obj, (datetime, date, time)):
            return self._serialize_datetime(obj)
        
        # Обработка Decimal
        if isinstance(obj, Decimal):
            return str(obj)
        
        # Обработка Enum
        if isinstance(obj, Enum):
            return {
                '__enum__': True,
                'type': type(obj).__name__,
                'value': obj.value
            }
        
        # Обработка UUID
        if isinstance(obj, UUID):
            return str(obj)
        
        # Обработка dataclass
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                '__dataclass__': True,
                'type': type(obj).__name__,
                'data': asdict(obj)
            }
        
        # Обработка bytes
        if isinstance(obj, bytes):
            return {
                '__bytes__': True,
                'data': base64.b64encode(obj).decode('ascii')
            }
        
        # Обработка set
        if isinstance(obj, set):
            return {'__set__': True, 'data': list(obj)}
        
        # Обработка complex
        if isinstance(obj, complex):
            return {'__complex__': True, 'real': obj.real, 'imag': obj.imag}
        
        # Обработка Path
        if isinstance(obj, Path):
            return str(obj)
        
        # Обработка классов с методом to_dict
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        
        # Для всех остальных - вызов родительского метода
        return super().default(obj)
    
    def _serialize_datetime(self, obj: Union[datetime, date, time]) -> Dict[str, Any]:
        """Сериализует объекты datetime/date/time."""
        result = {
            '__datetime__': True,
            'year': obj.year if hasattr(obj, 'year') else 0,
            'month': obj.month if hasattr(obj, 'month') else 0,
            'day': obj.day if hasattr(obj, 'day') else 0
        }
        
        if isinstance(obj, (datetime, time)):
            result.update({
                'hour': obj.hour,
                'minute': obj.minute,
                'second': obj.second,
                'microsecond': obj.microsecond
            })
        
        if isinstance(obj, datetime):
            result['tzinfo'] = str(obj.tzinfo) if obj.tzinfo else None
        
        return result


class AdvancedJSONDecoder(json.JSONDecoder):
    """
    Расширенный JSON decoder для восстановления специальных типов.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
    
    def object_hook(self, obj: Dict[str, Any]) -> Any:
        # Восстановление datetime
        if '__datetime__' in obj:
            return self._deserialize_datetime(obj)
        
        # Восстановление Enum
        if '__enum__' in obj:
            return self._deserialize_enum(obj)
        
        # Восстановление dataclass
        if '__dataclass__' in obj:
            return self._deserialize_dataclass(obj)
        
        # Восстановление bytes
        if '__bytes__' in obj:
            return base64.b64decode(obj['data'])
        
        # Восстановление set
        if '__set__' in obj:
            return set(obj['data'])
        
        # Восстановление complex
        if '__complex__' in obj:
            return complex(obj['real'], obj['imag'])
        
        return obj
    
    def _deserialize_datetime(self, obj: Dict[str, Any]) -> Union[datetime, date, time]:
        """Восстанавливает объекты datetime/date/time."""
        if 'hour' in obj:
            # Это datetime или time
            if 'year' in obj and obj['year'] > 0:
                # Это datetime
                return datetime(
                    year=obj['year'],
                    month=obj['month'],
                    day=obj['day'],
                    hour=obj['hour'],
                    minute=obj['minute'],
                    second=obj['second'],
                    microsecond=obj['microsecond']
                )
            else:
                # Это time
                return time(
                    hour=obj['hour'],
                    minute=obj['minute'],
                    second=obj['second'],
                    microsecond=obj['microsecond']
                )
        else:
            # Это date
            return date(
                year=obj['year'],
                month=obj['month'],
                day=obj['day']
            )
    
    def _deserialize_enum(self, obj: Dict[str, Any]) -> Any:
        """Восстанавливает Enum объекты."""
        # В реальном приложении нужно импортировать класс Enum
        # Здесь упрощенная версия
        return obj['value']
    
    def _deserialize_dataclass(self, obj: Dict[str, Any]) -> Any:
        """Восстанавливает dataclass объекты."""
        # В реальном приложении нужно импортировать класс dataclass
        # Здесь упрощенная версия - возвращаем словарь
        return obj['data']


class DataSerializer:
    """
    Основной класс для сериализации и десериализации данных.
    """
    
    def __init__(self, default_format: SerializationFormat = SerializationFormat.JSON):
        """
        Инициализация сериализатора.
        
        Args:
            default_format: Формат по умолчанию
        """
        self.default_format = default_format
        self.logger = logging.getLogger(__name__)
        
        # Регистрация сериализаторов для разных форматов
        self.serializers = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.MESSAGEPACK: self._serialize_msgpack,
            SerializationFormat.PICKLE: self._serialize_pickle,
            SerializationFormat.YAML: self._serialize_yaml,
            SerializationFormat.CSV: self._serialize_csv,
            SerializationFormat.BINARY: self._serialize_binary,
        }
        
        # Регистрация десериализаторов
        self.deserializers = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.MESSAGEPACK: self._deserialize_msgpack,
            SerializationFormat.PICKLE: self._deserialize_pickle,
            SerializationFormat.YAML: self._deserialize_yaml,
            SerializationFormat.CSV: self._deserialize_csv,
            SerializationFormat.BINARY: self._deserialize_binary,
        }
    
    def serialize(self, 
                  data: Any, 
                  format: SerializationFormat = None,
                  **kwargs) -> Union[str, bytes]:
        """
        Сериализует данные в указанный формат.
        
        Args:
            data: Данные для сериализации
            format: Формат сериализации (по умолчанию используется default_format)
            **kwargs: Дополнительные параметры для сериализатора
            
        Returns:
            Сериализованные данные (строка или bytes)
        """
        if format is None:
            format = self.default_format
        
        if format not in self.serializers:
            raise SerializationError(f"Неподдерживаемый формат сериализации: {format}")
        
        try:
            serializer = self.serializers[format]
            return serializer(data, **kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка сериализации в формат {format}: {e}")
            raise SerializationError(f"Ошибка сериализации: {e}")
    
    def deserialize(self, 
                    serialized_data: Union[str, bytes], 
                    format: SerializationFormat = None,
                    target_type: Type[T] = None,
                    **kwargs) -> Any:
        """
        Десериализует данные из указанного формата.
        
        Args:
            serialized_data: Сериализованные данные
            format: Формат десериализации
            target_type: Ожидаемый тип данных (для валидации)
            **kwargs: Дополнительные параметры для десериализатора
            
        Returns:
            Десериализованные данные
        """
        if format is None:
            # Пытаемся определить формат автоматически
            format = self._detect_format(serialized_data)
            if format is None:
                format = self.default_format
        
        if format not in self.deserializers:
            raise SerializationError(f"Неподдерживаемый формат десериализации: {format}")
        
        try:
            deserializer = self.deserializers[format]
            result = deserializer(serialized_data, **kwargs)
            
            # Валидация типа, если указан target_type
            if target_type is not None:
                if not isinstance(result, target_type):
                    # Попытка преобразования
                    result = self._convert_to_type(result, target_type)
            
            return result
        except Exception as e:
            self.logger.error(f"Ошибка десериализации из формата {format}: {e}")
            raise SerializationError(f"Ошибка десериализации: {e}")
    
    def _detect_format(self, data: Union[str, bytes]) -> Optional[SerializationFormat]:
        """Определяет формат данных автоматически."""
        if isinstance(data, bytes):
            # Пытаемся определить бинарные форматы
            try:
                # Проверяем MessagePack
                if len(data) > 0:
                    # MessagePack обычно начинается с определенных байт
                    if data[0] in [0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f]:
                        return SerializationFormat.MESSAGEPACK
                    
                    # Проверяем Pickle
                    if data[:2] == b'\x80\x04':
                        return SerializationFormat.PICKLE
                
                return SerializationFormat.BINARY
            except:
                pass
        
        elif isinstance(data, str):
            # Пытаемся определить текстовые форматы
            data_start = data.strip()[:100]
            
            # Проверяем JSON
            if data_start.startswith(('{', '[', '"', 'true', 'false', 'null', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-')):
                try:
                    json.loads(data_start + ('}' if data_start.startswith('{') else ']'))
                    return SerializationFormat.JSON
                except:
                    pass
            
            # Проверяем YAML
            if data_start.startswith(('---', '- ', 'key:', 'value:', '#')):
                try:
                    yaml.safe_load(data_start)
                    return SerializationFormat.YAML
                except:
                    pass
            
            # Проверяем CSV
            lines = data_start.split('\n')
            if len(lines) > 1 and ',' in lines[0] and ',' in lines[1]:
                return SerializationFormat.CSV
        
        return None
    
    def _convert_to_type(self, data: Any, target_type: Type[T]) -> T:
        """Пытается преобразовать данные к указанному типу."""
        # Если уже правильный тип
        if isinstance(data, target_type):
            return data
        
        # Для примитивных типов
        if target_type in (str, int, float, bool):
            return target_type(data)
        
        # Для списков
        if hasattr(target_type, '__origin__') and target_type.__origin__ == list:
            item_type = target_type.__args__[0]
            return [self._convert_to_type(item, item_type) for item in data]
        
        # Для словарей
        if hasattr(target_type, '__origin__') and target_type.__origin__ == dict:
            key_type, value_type = target_type.__args__
            return {
                self._convert_to_type(k, key_type): self._convert_to_type(v, value_type)
                for k, v in data.items()
            }
        
        # Для dataclass
        if is_dataclass(target_type):
            return self._dataclass_from_dict(data, target_type)
        
        # Для классов с методом from_dict
        if hasattr(target_type, 'from_dict'):
            return target_type.from_dict(data)
        
        raise SerializationError(f"Невозможно преобразовать {type(data)} в {target_type}")
    
    def _dataclass_from_dict(self, data: Dict[str, Any], dataclass_type: Type[T]) -> T:
        """Создает dataclass из словаря."""
        if not is_dataclass(dataclass_type):
            raise SerializationError(f"{dataclass_type} не является dataclass")
        
        # Получаем подсказки типов
        type_hints = get_type_hints(dataclass_type)
        
        # Подготавливаем аргументы
        kwargs = {}
        
        for field_name, field_type in type_hints.items():
            if field_name in data:
                try:
                    kwargs[field_name] = self._convert_to_type(data[field_name], field_type)
                except Exception as e:
                    self.logger.warning(f"Ошибка преобразования поля {field_name}: {e}")
                    kwargs[field_name] = data[field_name]
            else:
                # Пропускаем отсутствующие поля (будут значения по умолчанию)
                pass
        
        return dataclass_type(**kwargs)
    
    # === Реализации сериализаторов ===
    
    def _serialize_json(self, data: Any, **kwargs) -> str:
        """Сериализация в JSON."""
        default_kwargs = {
            'cls': AdvancedJSONEncoder,
            'ensure_ascii': False,
            'indent': 2,
            'sort_keys': True
        }
        default_kwargs.update(kwargs)
        
        return json.dumps(data, **default_kwargs)
    
    def _deserialize_json(self, data: Union[str, bytes], **kwargs) -> Any:
        """Десериализация из JSON."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        default_kwargs = {
            'cls': AdvancedJSONDecoder
        }
        default_kwargs.update(kwargs)
        
        return json.loads(data, **default_kwargs)
    
    def _serialize_msgpack(self, data: Any, **kwargs) -> bytes:
        """Сериализация в MessagePack."""
        # Преобразуем специальные типы перед сериализацией
        data = self._prepare_for_msgpack(data)
        return msgpack.packb(data, **kwargs)
    
    def _deserialize_msgpack(self, data: bytes, **kwargs) -> Any:
        """Десериализация из MessagePack."""
        return msgpack.unpackb(data, **kwargs)
    
    def _prepare_for_msgpack(self, data: Any) -> Any:
        """Подготавливает данные для MessagePack."""
        if isinstance(data, (datetime, date)):
            return {
                '__datetime__': True,
                'iso': data.isoformat()
            }
        elif isinstance(data, Enum):
            return {
                '__enum__': True,
                'value': data.value
            }
        elif isinstance(data, UUID):
            return str(data)
        elif isinstance(data, set):
            return list(data)
        elif is_dataclass(data) and not isinstance(data, type):
            return asdict(data)
        elif isinstance(data, dict):
            return {k: self._prepare_for_msgpack(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._prepare_for_msgpack(item) for item in data]
        else:
            return data
    
    def _serialize_pickle(self, data: Any, **kwargs) -> bytes:
        """Сериализация в Pickle."""
        default_kwargs = {
            'protocol': pickle.HIGHEST_PROTOCOL
        }
        default_kwargs.update(kwargs)
        
        return pickle.dumps(data, **default_kwargs)
    
    def _deserialize_pickle(self, data: bytes, **kwargs) -> Any:
        """Десериализация из Pickle."""
        return pickle.loads(data, **kwargs)
    
    def _serialize_yaml(self, data: Any, **kwargs) -> str:
        """Сериализация в YAML."""
        default_kwargs = {
            'default_flow_style': False,
            'allow_unicode': True,
            'encoding': 'utf-8'
        }
        default_kwargs.update(kwargs)
        
        return yaml.dump(data, **default_kwargs)
    
    def _deserialize_yaml(self, data: Union[str, bytes], **kwargs) -> Any:
        """Десериализация из YAML."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        return yaml.safe_load(data, **kwargs)
    
    def _serialize_csv(self, data: Any, **kwargs) -> str:
        """Сериализация в CSV."""
        if not isinstance(data, (list, tuple)):
            data = [data]
        
        # Преобразуем в список словарей
        if isinstance(data[0], dict):
            # Все словари должны иметь одинаковые ключи
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = list(fieldnames)
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in data:
                writer.writerow(item)
            
            return output.getvalue()
        
        # Простой список значений
        else:
            output = io.StringIO()
            writer = csv.writer(output)
            
            for item in data:
                if isinstance(item, (list, tuple)):
                    writer.writerow(item)
                else:
                    writer.writerow([item])
            
            return output.getvalue()
    
    def _deserialize_csv(self, data: Union[str, bytes], **kwargs) -> List[Dict[str, Any]]:
        """Десериализация из CSV."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        # Пытаемся определить, есть ли заголовки
        lines = data.strip().split('\n')
        if len(lines) < 2:
            return []
        
        # Проверяем первую строку на наличие заголовков
        first_line = lines[0]
        second_line = lines[1]
        
        has_header = ',' in first_line and ',' in second_line
        
        input_io = io.StringIO(data)
        
        if has_header:
            reader = csv.DictReader(input_io)
            return list(reader)
        else:
            reader = csv.reader(input_io)
            return [row for row in reader]
    
    def _serialize_binary(self, data: bytes, **kwargs) -> bytes:
        """Сериализация в бинарный формат (просто возвращает bytes)."""
        if not isinstance(data, bytes):
            raise SerializationError("Для бинарной сериализации нужны данные типа bytes")
        
        return data
    
    def _deserialize_binary(self, data: bytes, **kwargs) -> bytes:
        """Десериализация из бинарного формата."""
        return data
    
    def serialize_to_file(self, 
                         data: Any, 
                         file_path: Union[str, Path],
                         format: SerializationFormat = None,
                         **kwargs):
        """
        Сериализует данные и сохраняет в файл.
        """
        if format is None:
            # Определяем формат по расширению файла
            file_path = Path(file_path)
            ext = file_path.suffix.lower()[1:] if file_path.suffix else ''
            
            format_map = {
                'json': SerializationFormat.JSON,
                'msgpack': SerializationFormat.MESSAGEPACK,
                'mp': SerializationFormat.MESSAGEPACK,
                'pickle': SerializationFormat.PICKLE,
                'pkl': SerializationFormat.PICKLE,
                'yaml': SerializationFormat.YAML,
                'yml': SerializationFormat.YAML,
                'csv': SerializationFormat.CSV,
                'bin': SerializationFormat.BINARY,
                'dat': SerializationFormat.BINARY,
            }
            
            format = format_map.get(ext, self.default_format)
        
        serialized_data = self.serialize(data, format, **kwargs)
        
        # Создаем директорию, если её нет
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Записываем в файл
        mode = 'wb' if isinstance(serialized_data, bytes) else 'w'
        
        with open(file_path, mode, encoding='utf-8' if mode == 'w' else None) as f:
            f.write(serialized_data)
        
        self.logger.debug(f"Данные сохранены в {file_path} ({format.value})")
    
    def deserialize_from_file(self, 
                             file_path: Union[str, Path],
                             format: SerializationFormat = None,
                             target_type: Type[T] = None,
                             **kwargs) -> Any:
        """
        Загружает и десериализует данные из файла.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Определяем формат, если не указан
        if format is None:
            ext = file_path.suffix.lower()[1:] if file_path.suffix else ''
            
            format_map = {
                'json': SerializationFormat.JSON,
                'msgpack': SerializationFormat.MESSAGEPACK,
                'mp': SerializationFormat.MESSAGEPACK,
                'pickle': SerializationFormat.PICKLE,
                'pkl': SerializationFormat.PICKLE,
                'yaml': SerializationFormat.YAML,
                'yml': SerializationFormat.YAML,
                'csv': SerializationFormat.CSV,
                'bin': SerializationFormat.BINARY,
                'dat': SerializationFormat.BINARY,
            }
            
            format = format_map.get(ext, self.default_format)
        
        # Читаем файл
        mode = 'rb' if format in [SerializationFormat.MESSAGEPACK, 
                                  SerializationFormat.PICKLE, 
                                  SerializationFormat.BINARY] else 'r'
        
        with open(file_path, mode, encoding='utf-8' if mode == 'r' else None) as f:
            data = f.read()
        
        # Десериализуем
        return self.deserialize(data, format, target_type, **kwargs)


# Глобальный экземпляр для удобства использования
serializer = DataSerializer()


# Декораторы для автоматической сериализации/десериализации
def serializable(format: SerializationFormat = SerializationFormat.JSON):
    """
    Декоратор для автоматической сериализации возвращаемого значения функции.
    
    Пример:
        @serializable(SerializationFormat.JSON)
        def get_data():
            return {"key": "value"}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return serializer.serialize(result, format)
        return wrapper
    return decorator


def deserializable(format: SerializationFormat = SerializationFormat.JSON, 
                  target_type: Type = None):
    """
    Декоратор для автоматической десериализации аргумента функции.
    
    Пример:
        @deserializable(SerializationFormat.JSON, dict)
        def process_data(data):
            return data["key"]
    """
    def decorator(func):
        def wrapper(serialized_data, *args, **kwargs):
            data = serializer.deserialize(serialized_data, format, target_type)
            return func(data, *args, **kwargs)
        return wrapper
    return decorator


# Пример использования
if __name__ == "__main__":
    import logging
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Тестовые данные
    test_data = {
        "name": "Test Command",
        "tokens": ["create", "project"],
        "parameters": {"project_name": "MyProject", "template": "default"},
        "timestamp": datetime.now(),
        "count": Decimal("123.45"),
        "status": ExecutionStatus.SUCCESS,
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "binary_data": b"Hello World",
        "tags": {"python", "serialization", "test"}
    }
    
    print("Исходные данные:")
    print(json.dumps(test_data, cls=AdvancedJSONEncoder, indent=2))
    
    # Тестируем разные форматы
    formats = [
        SerializationFormat.JSON,
        SerializationFormat.YAML,
        SerializationFormat.MESSAGEPACK,
        SerializationFormat.PICKLE
    ]
    
    for fmt in formats:
        print(f"\n=== Тестирование формата: {fmt.value} ===")
        
        try:
            # Сериализация
            serialized = serializer.serialize(test_data, fmt)
            
            if isinstance(serialized, bytes):
                print(f"Размер: {len(serialized)} байт")
                print(f"Первые 100 байт: {serialized[:100]}")
            else:
                print(f"Размер: {len(serialized)} символов")
                print(f"Первые 200 символов: {serialized[:200]}...")
            
            # Десериализация
            deserialized = serializer.deserialize(serialized, fmt)
            
            print(f"Десериализация успешна: {isinstance(deserialized, dict)}")
            
            # Проверяем ключевые поля
            print(f"Восстановленное имя: {deserialized.get('name')}")
            
        except Exception as e:
            print(f"Ошибка: {e}")
    
    # Тестирование работы с файлами
    print("\n=== Тестирование работы с файлами ===")
    
    # Сохраняем в JSON файл
    serializer.serialize_to_file(
        test_data,
        "data/test_data.json",
        SerializationFormat.JSON
    )
    
    # Загружаем из JSON файла
    loaded_data = serializer.deserialize_from_file(
        "data/test_data.json",
        SerializationFormat.JSON
    )
    
    print(f"Данные загружены из файла: {loaded_data['name']}")
    
    # Тестирование dataclass
    @dataclass
    class TestCommand:
        name: str
        tokens: List[str]
        timestamp: datetime
    
    command = TestCommand(
        name="Test Command",
        tokens=["test", "command"],
        timestamp=datetime.now()
    )
    
    # Сериализация dataclass
    command_json = serializer.serialize(command, SerializationFormat.JSON)
    print(f"\nСериализованный dataclass: {command_json[:100]}...")
    
    # Десериализация с указанием типа
    restored_command = serializer.deserialize(
        command_json,
        SerializationFormat.JSON,
        target_type=TestCommand
    )
    
    print(f"Восстановленный dataclass: {restored_command}")