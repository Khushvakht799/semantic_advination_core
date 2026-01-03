import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import os


class CommandTrie:
    """
    Trie-структура для быстрого поиска команд по префиксу.
    Поддерживает инкрементальное обновление и сохранение на диск.
    """
    
    def __init__(self, data_dir: str = "data", config: Dict[str, Any] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {
            "auto_save": True,
            "save_interval_seconds": 300,
            "max_commands": 10000,
            "max_prefix_length": 100,
            "enable_fuzzy_search": True,
            "fuzzy_threshold": 0.6,
            "compression": False,
            "backup_on_save": True
        }
        
        # Основные структуры данных
        self.trie = {}  # Основная Trie-структура
        self.commands: Dict[str, Dict] = {}  # command_id -> command_data
        self.reverse_index: Dict[str, Set[str]] = {}  # command_text -> set(command_ids)
        
        # Метаданные
        self.metadata = {
            "total_commands": 0,
            "total_searches": 0,
            "total_inserts": 0,
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
            "created_at": datetime.now().isoformat()
        }
        
        # Кэши для производительности
        self.prefix_cache: Dict[str, List[str]] = {}
        self.similarity_cache: Dict[str, List[Dict]] = {}
        
        # Статистика
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "disk_reads": 0,
            "disk_writes": 0
        }
        
        # Загружаем из файла, если существует
        self._load_from_disk()
    
    def insert(self, command_data: Dict[str, Any]) -> str:
        """
        Добавляет команду в Trie.
        
        Args:
            command_data: Словарь с данными команды.
                         Должен содержать ключ 'command' (текст команды).
        
        Returns:
            ID команды в хранилище.
        """
        command_text = command_data.get("command", "").strip()
        if not command_text:
            raise ValueError("Command text cannot be empty")
        
        # Проверяем максимальную длину
        if len(command_text) > self.config["max_prefix_length"]:
            raise ValueError(f"Command too long: {len(command_text)} characters")
        
        # Проверяем максимальное количество команд
        if len(self.commands) >= self.config["max_commands"]:
            self._evict_old_commands()
        
        # Генерируем ID если нет
        if "id" not in command_data:
            command_data["id"] = self._generate_command_id(command_text)
        
        command_id = command_data["id"]
        
        # Если команда уже существует, обновляем её
        if command_id in self.commands:
            return self._update_existing_command(command_id, command_data)
        
        # Добавляем метаданные
        if "created_at" not in command_data:
            command_data["created_at"] = datetime.now().isoformat()
        if "usage_count" not in command_data:
            command_data["usage_count"] = 1
        if "last_used" not in command_data:
            command_data["last_used"] = datetime.now().isoformat()
        
        # Нормализуем текст команды для Trie
        normalized_text = self._normalize_command_text(command_text)
        
        # Сохраняем команду
        self.commands[command_id] = command_data
        
        # Добавляем в обратный индекс
        if normalized_text not in self.reverse_index:
            self.reverse_index[normalized_text] = set()
        self.reverse_index[normalized_text].add(command_id)
        
        # Добавляем в Trie
        node = self.trie
        for char in normalized_text:
            if char not in node:
                node[char] = {"__commands": set()}
            node = node[char]
            node["__commands"].add(command_id)
        
        # Помечаем конец слова
        node["__end"] = True
        
        # Добавляем n-граммы для нечеткого поиска
        if self.config["enable_fuzzy_search"]:
            self._add_ngrams(command_id, normalized_text)
        
        # Обновляем метаданные
        self.metadata["total_commands"] = len(self.commands)
        self.metadata["total_inserts"] += 1
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Очищаем кэши
        self._clear_cache_for_prefix(normalized_text[:3])
        
        # Сохраняем на диск
        if self.config["auto_save"]:
            self._schedule_save()
        
        return command_id
    
    def search_exact(self, prefix: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Ищет команды, точно начинающиеся с префикса.
        
        Args:
            prefix: Префикс для поиска.
            limit: Максимальное количество результатов.
        
        Returns:
            Список команд, отсортированных по релевантности.
        """
        if not prefix:
            return []
        
        # Проверяем кэш
        cache_key = f"exact_{prefix}_{limit}"
        if cache_key in self.prefix_cache:
            command_ids = self.prefix_cache[cache_key]
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
            
            # Ищем узел в Trie
            node = self.trie
            normalized_prefix = self._normalize_command_text(prefix)
            
            for char in normalized_prefix:
                if char not in node:
                    # Префикс не найден
                    self.prefix_cache[cache_key] = []
                    return []
                node = node[char]
            
            # Собираем все команды из поддерва
            command_ids = self._collect_command_ids(node, limit)
            
            # Сохраняем в кэш
            self.prefix_cache[cache_key] = command_ids
        
        # Получаем данные команд
        results = []
        for cmd_id in command_ids:
            if cmd_id in self.commands:
                cmd_data = self._prepare_command_result(cmd_id, prefix)
                results.append(cmd_data)
        
        # Обновляем статистику
        self.metadata["total_searches"] += 1
        
        return results[:limit]
    
    def search_similar(self,
                      query: str,
                      threshold: float = 0.3,
                      limit: int = 3) -> List[Dict[str, Any]]:
        """
        Ищет команды, похожие на запрос.
        Использует комбинацию методов для определения похожести.
        
        Args:
            query: Запрос для поиска похожих команд.
            threshold: Порог похожести (0.0 - 1.0).
            limit: Максимальное количество результатов.
        
        Returns:
            Список похожих команд с оценкой похожести.
        """
        if not query or not self.config["enable_fuzzy_search"]:
            return []
        
        # Проверяем кэш
        cache_key = f"similar_{query}_{threshold}_{limit}"
        if cache_key in self.similarity_cache:
            self.stats["cache_hits"] += 1
            return self.similarity_cache[cache_key][:limit]
        
        self.stats["cache_misses"] += 1
        
        normalized_query = self._normalize_command_text(query)
        similar_commands = []
        
        # Метод 1: Поиск по префиксу с допущениями
        prefix_results = self._fuzzy_prefix_search(normalized_query, threshold)
        similar_commands.extend(prefix_results)
        
        # Метод 2: Поиск по n-граммам
        ngram_results = self._ngram_search(normalized_query, threshold)
        similar_commands.extend(ngram_results)
        
        # Метод 3: Поиск по токенам
        token_results = self._token_search(normalized_query, threshold)
        similar_commands.extend(token_results)
        
        # Удаляем дубликаты и объединяем результаты
        unique_results = self._deduplicate_similar_results(similar_commands)
        
        # Сортируем по похожести
        unique_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Сохраняем в кэш
        self.similarity_cache[cache_key] = unique_results
        
        return unique_results[:limit]
    
    def get_command(self, command_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает команду по ID."""
        return self.commands.get(command_id)
    
    def get_command_by_text(self, command_text: str) -> Optional[Dict[str, Any]]:
        """Возвращает команду по тексту."""
        normalized = self._normalize_command_text(command_text)
        if normalized in self.reverse_index:
            for cmd_id in self.reverse_index[normalized]:
                return self.commands.get(cmd_id)
        return None
    
    def update_usage(self, command_id: str, increment: int = 1) -> bool:
        """Увеличивает счётчик использования команды."""
        if command_id in self.commands:
            current = self.commands[command_id].get("usage_count", 0)
            self.commands[command_id]["usage_count"] = current + increment
            self.commands[command_id]["last_used"] = datetime.now().isoformat()
            
            # Обновляем метаданные
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            # Сохраняем на диск
            if self.config["auto_save"]:
                self._schedule_save()
            
            return True
        return False
    
    def delete_command(self, command_id: str) -> bool:
        """Удаляет команду из хранилища."""
        if command_id not in self.commands:
            return False
        
        # Получаем текст команды
        cmd_data = self.commands[command_id]
        cmd_text = cmd_data.get("command", "")
        normalized_text = self._normalize_command_text(cmd_text)
        
        # Удаляем из Trie
        node = self.trie
        for char in normalized_text:
            if char in node:
                node = node[char]
                if "__commands" in node and command_id in node["__commands"]:
                    node["__commands"].remove(command_id)
        
        # Удаляем из обратного индекса
        if normalized_text in self.reverse_index:
            self.reverse_index[normalized_text].discard(command_id)
            if not self.reverse_index[normalized_text]:
                del self.reverse_index[normalized_text]
        
        # Удаляем n-граммы
        if self.config["enable_fuzzy_search"]:
            self._remove_ngrams(command_id, normalized_text)
        
        # Удаляем из основного хранилища
        del self.commands[command_id]
        
        # Обновляем метаданные
        self.metadata["total_commands"] = len(self.commands)
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Очищаем кэши
        self._clear_all_caches()
        
        # Сохраняем на диск
        if self.config["auto_save"]:
            self._schedule_save()
        
        return True
    
    def bulk_insert(self, commands: List[Dict[str, Any]]) -> List[str]:
        """Пакетная вставка команд."""
        ids = []
        for cmd_data in commands:
            try:
                cmd_id = self.insert(cmd_data)
                ids.append(cmd_id)
            except Exception as e:
                print(f"Warning: Failed to insert command {cmd_data.get('command', 'unknown')}: {e}")
        
        # Сохраняем на диск после пакетной вставки
        if self.config["auto_save"]:
            self.save_to_disk()
        
        return ids
    
    def search_with_filters(self,
                           prefix: str,
                           filters: Dict[str, Any],
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск с дополнительными фильтрами.
        
        Args:
            prefix: Префикс для поиска
            filters: Словарь с фильтрами
            limit: Максимальное количество результатов
        
        Returns:
            Отфильтрованный список команд
        """
        # Сначала выполняем обычный поиск
        results = self.search_exact(prefix, limit * 2)  # Берем больше, чтобы отфильтровать
        
        if not results or not filters:
            return results[:limit]
        
        filtered_results = []
        
        for cmd in results:
            include = True
            
            # Фильтр по минимальному usage_count
            if "min_usage" in filters:
                if cmd.get("usage_count", 0) < filters["min_usage"]:
                    include = False
            
            # Фильтр по категории
            if "category" in filters:
                if cmd.get("category", "") != filters["category"]:
                    include = False
            
            # Фильтр по тегам
            if "tags" in filters:
                cmd_tags = set(cmd.get("tags", []))
                filter_tags = set(filters["tags"])
                if not filter_tags.issubset(cmd_tags):
                    include = False
            
            # Фильтр по дате создания
            if "created_after" in filters:
                created_at = cmd.get("created_at", "")
                if created_at:
                    try:
                        cmd_date = datetime.fromisoformat(created_at)
                        filter_date = datetime.fromisoformat(filters["created_after"])
                        if cmd_date < filter_date:
                            include = False
                    except ValueError:
                        pass
            
            if include:
                filtered_results.append(cmd)
        
        return filtered_results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику хранилища."""
        total_usage = sum(cmd.get("usage_count", 0) for cmd in self.commands.values())
        avg_usage = total_usage / len(self.commands) if self.commands else 0
        
        # Анализ популярности
        usage_distribution = {}
        for cmd in self.commands.values():
            usage = cmd.get("usage_count", 0)
            bucket = (usage // 10) * 10  # Группируем по десяткам
            usage_distribution[bucket] = usage_distribution.get(bucket, 0) + 1
        
        return {
            **self.metadata,
            "total_usage": total_usage,
            "avg_usage_per_command": avg_usage,
            "unique_prefixes": self._count_unique_prefixes(),
            "cache_stats": self.stats.copy(),
            "memory_usage_mb": self._estimate_memory_usage(),
            "usage_distribution": dict(sorted(usage_distribution.items())),
            "top_commands": self._get_top_commands(5)
        }
    
    def save_to_disk(self, backup: bool = None):
        """Сохраняет Trie на диск."""
        if backup is None:
            backup = self.config["backup_on_save"]
        
        # Создаем backup если нужно
        if backup:
            self._create_backup()
        
        data_file = self.data_dir / "command_trie.pkl"
        try:
            save_data = {
                'trie': self.trie,
                'commands': self.commands,
                'reverse_index': self.reverse_index,
                'metadata': self.metadata,
                'config': self.config
            }
            
            with open(data_file, 'wb') as f:
                if self.config["compression"]:
                    import gzip
                    with gzip.GzipFile(fileobj=f, mode='wb') as gz_file:
                        pickle.dump(save_data, gz_file, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.stats["disk_writes"] += 1
            self.metadata["last_saved"] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"Warning: Could not save Trie to disk: {e}")
    
    def load_from_disk(self) -> bool:
        """Загружает Trie с диска."""
        return self._load_from_disk()
    
    def export_to_json(self, filepath: str, indent: int = 2):
        """Экспортирует все команды в JSON файл."""
        export_data = {
            "metadata": self.metadata,
            "config": self.config,
            "commands": list(self.commands.values())
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=False)
    
    def import_from_json(self, filepath: str, merge: bool = True):
        """Импортирует команды из JSON файла."""
        with open(filepath, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        if not merge:
            self.clear()
        
        for cmd_data in import_data.get("commands", []):
            try:
                self.insert(cmd_data)
            except Exception as e:
                print(f"Warning: Failed to import command: {e}")
        
        # Сохраняем после импорта
        if self.config["auto_save"]:
            self.save_to_disk()
    
    def clear(self):
        """Очищает все данные в хранилище."""
        self.trie = {}
        self.commands.clear()
        self.reverse_index.clear()
        self.prefix_cache.clear()
        self.similarity_cache.clear()
        
        self.metadata.update({
            "total_commands": 0,
            "total_searches": 0,
            "total_inserts": 0,
            "last_updated": datetime.now().isoformat()
        })
        
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "disk_reads": 0,
            "disk_writes": 0
        }
    
    def optimize(self):
        """Оптимизирует структуры данных."""
        # Сжимаем Trie (удаляем пустые узлы)
        self._compress_trie()
        
        # Очищаем кэши
        self._clear_all_caches()
        
        # Перестраиваем обратный индекс если нужно
        self._rebuild_reverse_index()
        
        print(f"Optimization complete. Commands: {len(self.commands)}, "
              f"Trie nodes: {self._count_trie_nodes()}")
    
    # Вспомогательные методы
    
    def _normalize_command_text(self, text: str) -> str:
        """Нормализует текст команды для Trie."""
        # Приводим к нижнему регистру и удаляем лишние пробелы
        normalized = ' '.join(text.lower().split())
        
        # Удаляем специальные символы, которые могут мешать поиску
        # (но сохраняем важные для команд символы)
        import re
        normalized = re.sub(r'[^\w\s\-\.\/\@\:]', '', normalized)
        
        return normalized
    
    def _generate_command_id(self, command_text: str) -> str:
        """Генерирует уникальный ID для команды."""
        # Используем хеш команды + timestamp для уникальности
        hash_obj = hashlib.md5(command_text.encode('utf-8'))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{hash_obj.hexdigest()[:8]}_{timestamp}"
    
    def _update_existing_command(self, command_id: str, new_data: Dict[str, Any]) -> str:
        """Обновляет существующую команду."""
        old_data = self.commands[command_id]
        
        # Обновляем usage_count
        old_usage = old_data.get("usage_count", 0)
        new_usage = new_data.get("usage_count", 1)
        old_data["usage_count"] = old_usage + new_usage
        
        # Обновляем метаданные
        old_data["last_used"] = datetime.now().isoformat()
        
        # Обновляем другие поля если предоставлены
        for key in ["description", "category", "tags", "parameters"]:
            if key in new_data:
                old_data[key] = new_data[key]
        
        # Обновляем обратный индекс если изменился текст команды
        old_text = old_data.get("command", "")
        new_text = new_data.get("command", "")
        
        if new_text and old_text != new_text:
            # Удаляем старую запись из обратного индекса
            old_normalized = self._normalize_command_text(old_text)
            if old_normalized in self.reverse_index:
                self.reverse_index[old_normalized].discard(command_id)
                if not self.reverse_index[old_normalized]:
                    del self.reverse_index[old_normalized]
            
            # Добавляем новую
            new_normalized = self._normalize_command_text(new_text)
            if new_normalized not in self.reverse_index:
                self.reverse_index[new_normalized] = set()
            self.reverse_index[new_normalized].add(command_id)
            
            # Обновляем текст в данных команды
            old_data["command"] = new_text
        
        return command_id
    
    def _collect_command_ids(self, node: Dict, limit: int) -> List[str]:
        """Собирает ID команд из поддерэва Trie."""
        command_ids = set()
        
        def collect_recursive(current_node, ids_set):
            if "__commands" in current_node:
                ids_set.update(current_node["__commands"])
            
            for key, child_node in current_node.items():
                if key not in ["__commands", "__end"]:
                    collect_recursive(child_node, ids_set)
        
        collect_recursive(node, command_ids)
        
        # Сортируем по usage_count
        sorted_ids = sorted(
            command_ids,
            key=lambda cid: self.commands.get(cid, {}).get("usage_count", 0),
            reverse=True
        )
        
        return sorted_ids[:limit]
    
    def _prepare_command_result(self, command_id: str, prefix: str) -> Dict[str, Any]:
        """Подготавливает данные команды для возврата."""
        cmd_data = self.commands[command_id].copy()
        cmd_text = cmd_data.get("command", "")
        
        # Вычисляем score релевантности
        prefix_length = len(prefix)
        cmd_length = len(cmd_text)
        
        if cmd_length > 0:
            prefix_match_score = prefix_length / cmd_length
        else:
            prefix_match_score = 0
        
        usage_score = min(1.0, cmd_data.get("usage_count", 0) / 1000)
        
        # Взвешенный score
        cmd_data["score"] = 0.7 * prefix_match_score + 0.3 * usage_score
        cmd_data["prefix_match_score"] = prefix_match_score
        cmd_data["usage_score"] = usage_score
        
        return cmd_data
    
    def _fuzzy_prefix_search(self, query: str, threshold: float) -> List[Dict[str, Any]]:
        """Нечеткий поиск по префиксу."""
        results = []
        
        # Пробуем разные варианты префикса
        for prefix_length in range(max(1, len(query) - 2), len(query) + 1):
            prefix = query[:prefix_length]
            exact_matches = self.search_exact(prefix, limit=10)
            
            for match in exact_matches:
                cmd_text = match.get("command", "").lower()
                similarity = self._calculate_similarity(query, cmd_text)
                
                if similarity >= threshold:
                    match["similarity"] = similarity
                    match["search_method"] = "fuzzy_prefix"
                    results.append(match)
        
        return results
    
    def _ngram_search(self, query: str, threshold: float) -> List[Dict[str, Any]]:
        """Поиск по n-граммам."""
        # Заглушка для реальной реализации n-gram поиска
        # В реальной системе здесь была бы сложная логика
        return []
    
    def _token_search(self, query: str, threshold: float) -> List[Dict[str, Any]]:
        """Поиск по совпадению токенов."""
        results = []
        query_tokens = set(query.split())
        
        for cmd_id, cmd_data in self.commands.items():
            cmd_text = cmd_data.get("command", "").lower()
            cmd_tokens = set(cmd_text.split())
            
            if not query_tokens or not cmd_tokens:
                continue
            
            # Вычисляем меру Жаккара
            intersection = len(query_tokens & cmd_tokens)
            union = len(query_tokens | cmd_tokens)
            
            if union > 0:
                jaccard_similarity = intersection / union
                
                if jaccard_similarity >= threshold:
                    cmd_copy = cmd_data.copy()
                    cmd_copy["similarity"] = jaccard_similarity
                    cmd_copy["search_method"] = "token"
                    cmd_copy["common_tokens"] = list(query_tokens & cmd_tokens)
                    results.append(cmd_copy)
        
        return results
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Вычисляет похожесть между двумя строками."""
        if not str1 or not str2:
            return 0.0
        
        # Простая реализация на основе расстояния Левенштейна
        # В реальной системе можно использовать более сложные алгоритмы
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _deduplicate_similar_results(self, results: List[Dict]) -> List[Dict]:
        """Удаляет дубликаты из результатов похожего поиска."""
        seen = set()
        unique = []
        
        for result in results:
            cmd_id = result.get("id")
            if cmd_id and cmd_id not in seen:
                seen.add(cmd_id)
                unique.append(result)
        
        return unique
    
    def _add_ngrams(self, command_id: str, text: str, n: int = 3):
        """Добавляет n-граммы для команды."""
        # Заглушка для реальной реализации
        pass
    
    def _remove_ngrams(self, command_id: str, text: str):
        """Удаляет n-граммы для команды."""
        # Заглушка для реальной реализации
        pass
    
    def _evict_old_commands(self):
        """Удаляет старые редко используемые команды."""
        if len(self.commands) <= self.config["max_commands"]:
            return
        
        # Сортируем команды по usage_count и последнему использованию
        sorted_commands = sorted(
            self.commands.items(),
            key=lambda x: (
                x[1].get("usage_count", 0),
                x[1].get("last_used", "1970-01-01")
            )
        )
        
        # Удаляем самые старые и редко используемые
        to_remove = sorted_commands[:len(self.commands) - self.config["max_commands"]]
        
        for cmd_id, _ in to_remove:
            self.delete_command(cmd_id)
    
    def _count_unique_prefixes(self) -> int:
        """Считает количество уникальных префиксов в Trie."""
        prefixes = set()
        
        def count_recursive(node, current_prefix=""):
            if "__commands" in node and node["__commands"]:
                prefixes.add(current_prefix)
            
            for char, child_node in node.items():
                if char not in ["__commands", "__end"]:
                    count_recursive(child_node, current_prefix + char)
        
        count_recursive(self.trie)
        return len(prefixes)
    
    def _estimate_memory_usage(self) -> float:
        """Оценивает использование памяти в мегабайтах."""
        import sys
        
        total_size = 0
        
        # Оцениваем размер Trie
        def estimate_node_size(node):
            size = sys.getsizeof(node)
            for key, value in node.items():
                size += sys.getsizeof(key)
                if isinstance(value, dict):
                    size += estimate_node_size(value)
                else:
                    size += sys.getsizeof(value)
            return size
        
        total_size += estimate_node_size(self.trie)
        
        # Оцениваем размер commands
        for cmd_id, cmd_data in self.commands.items():
            total_size += sys.getsizeof(cmd_id) + sys.getsizeof(cmd_data)
        
        # Оцениваем размер reverse_index
        for text, ids in self.reverse_index.items():
            total_size += sys.getsizeof(text) + sys.getsizeof(ids)
        
        # Конвертируем в мегабайты
        return total_size / (1024 * 1024)
    
    def _get_top_commands(self, n: int) -> List[Dict[str, Any]]:
        """Возвращает топ-N самых популярных команд."""
        sorted_commands = sorted(
            self.commands.values(),
            key=lambda x: x.get("usage_count", 0),
            reverse=True
        )
        
        top = []
        for cmd in sorted_commands[:n]:
            top.append({
                "command": cmd.get("command", ""),
                "usage_count": cmd.get("usage_count", 0),
                "last_used": cmd.get("last_used", ""),
                "category": cmd.get("category", "")
            })
        
        return top
    
    def _clear_cache_for_prefix(self, prefix: str):
        """Очищает кэш для префикса и похожих ключей."""
        keys_to_remove = []
        for key in self.prefix_cache.keys():
            if key.startswith(f"exact_{prefix}") or key.startswith(f"similar_{prefix}"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.prefix_cache[key]
        
        # Также очищаем similarity_cache
        keys_to_remove = []
        for key in self.similarity_cache.keys():
            if key.startswith(f"similar_{prefix}"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.similarity_cache[key]
    
    def _clear_all_caches(self):
        """Очищает все кэши."""
        self.prefix_cache.clear()
        self.similarity_cache.clear()
    
    def _compress_trie(self):
        """Сжимает Trie, удаляя пустые узлы."""
        def compress_recursive(node):
            if not isinstance(node, dict):
                return node
            
            # Рекурсивно сжимаем детей
            for key in list(node.keys()):
                if key not in ["__commands", "__end"]:
                    node[key] = compress_recursive(node[key])
                    # Если узел стал пустым, удаляем его
                    if not node[key]:
                        del node[key]
            
            return node
        
        self.trie = compress_recursive(self.trie)
    
    def _rebuild_reverse_index(self):
        """Перестраивает обратный индекс."""
        self.reverse_index.clear()
        
        for cmd_id, cmd_data in self.commands.items():
            cmd_text = cmd_data.get("command", "")
            normalized = self._normalize_command_text(cmd_text)
            
            if normalized not in self.reverse_index:
                self.reverse_index[normalized] = set()
            self.reverse_index[normalized].add(cmd_id)
    
    def _count_trie_nodes(self) -> int:
        """Считает количество узлов в Trie."""
        def count_recursive(node):
            count = 1  # Текущий узел
            for key, child in node.items():
                if key not in ["__commands", "__end"] and isinstance(child, dict):
                    count += count_recursive(child)
            return count
        
        return count_recursive(self.trie) if self.trie else 0
    
    def _create_backup(self):
        """Создает backup текущих данных."""
        backup_dir = self.data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"command_trie_backup_{timestamp}.pkl"
        
        try:
            save_data = {
                'trie': self.trie,
                'commands': self.commands,
                'reverse_index': self.reverse_index,
                'metadata': self.metadata
            }
            
            with open(backup_file, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Удаляем старые backup (оставляем только 5 последних)
            backups = sorted(backup_dir.glob("command_trie_backup_*.pkl"))
            for old_backup in backups[:-5]:
                old_backup.unlink()
                
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    def _load_from_disk(self) -> bool:
        """Загружает Trie с диска."""
        data_file = self.data_dir / "command_trie.pkl"
        
        if not data_file.exists():
            return False
        
        try:
            with open(data_file, 'rb') as f:
                if self.config["compression"]:
                    import gzip
                    with gzip.GzipFile(fileobj=f, mode='rb') as gz_file:
                        data = pickle.load(gz_file)
                else:
                    data = pickle.load(f)
            
            self.trie = data.get('trie', {})
            self.commands = data.get('commands', {})
            self.reverse_index = data.get('reverse_index', {})
            self.metadata = data.get('metadata', self.metadata)
            
            # Обновляем конфиг если есть в сохраненных данных
            saved_config = data.get('config')
            if saved_config:
                self.config.update(saved_config)
            
            self.stats["disk_reads"] += 1
            
            print(f"Loaded {len(self.commands)} commands from disk")
            return True
            
        except Exception as e:
            print(f"Warning: Could not load Trie from disk: {e}")
            # Создаем новые структуры при ошибке загрузки
            self.trie = {}
            self.commands = {}
            self.reverse_index = {}
            return False
    
    def _schedule_save(self):
        """Планирует сохранение на диск (реализация отложенного сохранения)."""
        # В реальной системе здесь была бы асинхронная очередь сохранения
        # Для простоты сохраняем сразу
        self.save_to_disk()


class PersistentCommandTrie(CommandTrie):
    """
    Расширенная версия CommandTrie с улучшенной персистентностью.
    """
    
    def __init__(self, data_dir: str = "data", config: Dict[str, Any] = None):
        super().__init__(data_dir, config)
        
        # Дополнительные настройки для персистентности
        self.persistence_config = {
            "auto_save_changes": 100,  # Автосохранение после N изменений
            "incremental_save": True,
            "save_queue_size": 1000,
            "recovery_mode": "auto"
        }
        
        # Отслеживание изменений
        self.change_count = 0
        self.unsaved_changes = []
        
        # Восстановление при необходимости
        if self.persistence_config["recovery_mode"] == "auto":
            self._recover_if_needed()
    
    def insert(self, command_data: Dict[str, Any]) -> str:
        """Переопределенный insert с отслеживанием изменений."""
        cmd_id = super().insert(command_data)
        
        # Отслеживаем изменения для инкрементального сохранения
        self.change_count += 1
        self.unsaved_changes.append(("insert", cmd_id, command_data))
        
        # Проверяем, нужно ли автосохранение
        if (self.persistence_config["auto_save_changes"] > 0 and
            self.change_count >= self.persistence_config["auto_save_changes"]):
            self._save_incremental()
            self.change_count = 0
            self.unsaved_changes.clear()
        
        return cmd_id
    
    def delete_command(self, command_id: str) -> bool:
        """Переопределенный delete с отслеживанием изменений."""
        if command_id not in self.commands:
            return False
        
        # Сохраняем данные для возможного восстановления
        deleted_data = self.commands[command_id].copy()
        
        success = super().delete_command(command_id)
        
        if success:
            self.change_count += 1
            self.unsaved_changes.append(("delete", command_id, deleted_data))
        
        return success
    
    def _save_incremental(self):
        """Инкрементальное сохранение изменений."""
        if not self.persistence_config["incremental_save"]:
            self.save_to_disk()
            return
        
        # В реальной системе здесь была бы сложная логика
        # инкрементального сохранения
        self.save_to_disk(backup=False)
    
    def _recover_if_needed(self):
        """Восстанавливает данные при необходимости."""
        # Проверяем наличие файла журнала изменений
        journal_file = self.data_dir / "changes.journal"
        
        if journal_file.exists():
            print("Found change journal, attempting recovery...")
            try:
                self._recover_from_journal(journal_file)
            except Exception as e:
                print(f"Recovery failed: {e}")
                # В случае ошибки используем основной файл
    
    def _recover_from_journal(self, journal_file: Path):
        """Восстанавливает данные из журнала изменений."""
        # Заглушка для реальной реализации
        pass


# Утилитарные функции
def create_trie_from_commands(commands: List[str],
                             data_dir: str = "data") -> CommandTrie:
    """
    Создает CommandTrie из списка команд.
    
    Args:
        commands: Список текстов команд
        data_dir: Директория для данных
    
    Returns:
        Инициализированный CommandTrie
    """
    trie = CommandTrie(data_dir)
    
    for cmd_text in commands:
        trie.insert({
            "command": cmd_text,
            "usage_count": 1,
            "created_at": datetime.now().isoformat()
        })
    
    return trie


def merge_tries(trie1: CommandTrie, trie2: CommandTrie) -> CommandTrie:
    """
    Объединяет два CommandTrie в один.
    
    Args:
        trie1: Первый Trie
        trie2: Второй Trie
    
    Returns:
        Новый объединенный CommandTrie
    """
    # Создаем новый Trie
    merged = CommandTrie()
    
    # Добавляем команды из первого Trie
    for cmd_id, cmd_data in trie1.commands.items():
        merged.insert(cmd_data.copy())
    
    # Добавляем команды из второго Trie
    for cmd_id, cmd_data in trie2.commands.items():
        # Обновляем usage_count для существующих команд
        existing = merged.get_command_by_text(cmd_data.get("command", ""))
        if existing:
            merged.update_usage(existing["id"], cmd_data.get("usage_count", 1))
        else:
            merged.insert(cmd_data.copy())
    
    return merged
