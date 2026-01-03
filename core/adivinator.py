from typing import List, Dict, Any, Optional
from .models import *
from storage.trie_storage import CommandTrie


class Adivinator:
    """
    Чистая адивинация.
    Только угадывает варианты команд по префиксу.
    Не принимает решений о семантике или допустимости.
    """
    
    def __init__(self, storage: CommandTrie, config: Dict[str, Any] = None):
        self.storage = storage
        self.config = config or {
            "min_prefix_length": 2,
            "partial_threshold": 0.3,
            "max_exact_results": 5,
            "max_partial_results": 3,
            "enable_fallback": True
        }
    
    def advinate(self, prefix: str, context: Dict[str, Any] = None) -> AdvinationResult:
        """
        Основной метод адивинации.
        Возвращает только факты о найденных вариантах.
        """
        context = context or {}
        
        # 1. Проверка минимальной длины префикса
        if len(prefix) < self.config["min_prefix_length"]:
            return AdvinationResult(
                result_type=AdvinationResultType.NO_MATCH,
                raw_prefix=prefix,
                raw_context=context
            )
        
        # 2. Поиск точных совпадений
        exact_matches = self.storage.search_exact(
            prefix,
            limit=self.config["max_exact_results"]
        )
        
        if exact_matches:
            suggestions = [
                CommandSuggestion(
                    text=match["command"],
                    source="exact_match",
                    match_score=1.0,
                    metadata={
                        "usage_count": match.get("usage_count", 0),
                        "storage_score": match.get("score", 1.0),
                        "command_id": match.get("id")
                    }
                )
                for match in exact_matches
            ]
            
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=suggestions,
                confidence=1.0,
                raw_prefix=prefix,
                raw_context=context
            )
        
        # 3. Поиск похожих команд (если префикс достаточно длинный)
        if len(prefix) >= 3:
            similar_matches = self.storage.search_similar(
                prefix,
                threshold=self.config["partial_threshold"],
                limit=self.config["max_partial_results"]
            )
            
            if similar_matches:
                suggestions = []
                for match in similar_matches:
                    suggestions.append(
                        CommandSuggestion(
                            text=match["command"],
                            source="partial_match",
                            match_score=match.get("similarity", 0.5),
                            metadata={
                                "similarity": match.get("similarity", 0),
                                "distance": match.get("distance", 0),
                                "usage_count": match.get("usage_count", 0),
                                "command_id": match.get("id")
                            }
                        )
                    )
                
                # Confidence - максимальное значение похожести
                max_similarity = max(match.get("similarity", 0) for match in similar_matches)
                
                return AdvinationResult(
                    result_type=AdvinationResultType.PARTIAL_FOUND,
                    suggestions=suggestions,
                    confidence=max_similarity,
                    raw_prefix=prefix,
                    raw_context=context
                )
        
        # 4. Fallback-варианты (если включены)
        if self.config["enable_fallback"]:
            fallback_suggestions = self._get_fallback_suggestions(prefix, context)
            if fallback_suggestions:
                return AdvinationResult(
                    result_type=AdvinationResultType.PARTIAL_FOUND,
                    suggestions=fallback_suggestions,
                    confidence=0.1,  # Низкая уверенность для fallback
                    raw_prefix=prefix,
                    raw_context=context
                )
        
        # 5. Ничего не найдено
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            raw_prefix=prefix,
            raw_context=context
        )
    
    def learn(self, command_text: str, context: Dict[str, Any] = None):
        """
        Обучение на лету: добавление новой команды в хранилище.
        """
        command_data = {
            "command": command_text,
            "context": context or {},
            "usage_count": 1,
            "learned_at": datetime.now().isoformat()
        }
        
        # Извлекаем метаданные из контекста, если есть
        if context:
            if "category" in context:
                command_data["category"] = context["category"]
            if "tags" in context:
                command_data["tags"] = context["tags"]
            if "description" in context:
                command_data["description"] = context["description"]
        
        self.storage.insert(command_data)
    
    def batch_learn(self, commands: List[Dict[str, Any]]):
        """
        Пакетное обучение: добавление нескольких команд.
        """
        for cmd in commands:
            self.learn(cmd.get("command", ""), cmd.get("context", {}))
    
    def update_usage(self, command_text: str, increment: int = 1) -> bool:
        """
        Обновляет счетчик использования команды.
        """
        # Ищем команду по тексту
        matches = self.storage.search_exact(command_text, limit=1)
        if matches:
            command_id = matches[0].get("id")
            if command_id:
                return self.storage.update_usage(command_id, increment)
        return False
    
    def get_command_stats(self, command_text: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает статистику команды.
        """
        matches = self.storage.search_exact(command_text, limit=1)
        if matches:
            return {
                "command": matches[0].get("command"),
                "usage_count": matches[0].get("usage_count", 0),
                "score": matches[0].get("score", 0),
                "created_at": matches[0].get("created_at")
            }
        return None
    
    def _get_fallback_suggestions(self, prefix: str, context: Dict[str, Any]) -> List[CommandSuggestion]:
        """
        Возвращает fallback-варианты для очень общих префиксов.
        """
        fallbacks = []
        
        # Простые эвристики для распространенных команд
        if len(prefix) <= 2:
            # Для очень коротких префиксов
            common_commands = [
                "ls", "cd", "pwd", "mkdir", "rm", "cp", "mv",
                "git status", "git add", "git commit",
                "python", "pip install", "docker ps"
            ]
            
            for cmd in common_commands:
                if cmd.startswith(prefix):
                    fallbacks.append(
                        CommandSuggestion(
                            text=cmd,
                            source="fallback",
                            match_score=0.1,
                            metadata={"type": "common_command"}
                        )
                    )
        
        # Контекстно-зависимые fallback
        elif context.get("domain") == "git":
            git_commands = ["git add", "git commit", "git push", "git pull", "git status"]
            for cmd in git_commands:
                if cmd.startswith(prefix):
                    fallbacks.append(
                        CommandSuggestion(
                            text=cmd,
                            source="fallback",
                            match_score=0.15,
                            metadata={"domain": "git"}
                        )
                    )
        
        elif context.get("domain") == "docker":
            docker_commands = ["docker ps", "docker build", "docker run", "docker logs"]
            for cmd in docker_commands:
                if cmd.startswith(prefix):
                    fallbacks.append(
                        CommandSuggestion(
                            text=cmd,
                            source="fallback",
                            match_score=0.15,
                            metadata={"domain": "docker"}
                        )
                    )
        
        return fallbacks[:self.config["max_exact_results"]]
    
    def search_with_filters(self,
                           prefix: str,
                           filters: Dict[str, Any] = None,
                           context: Dict[str, Any] = None) -> AdvinationResult:
        """
        Поиск с дополнительными фильтрами.
        """
        # Базовый поиск
        result = self.advinate(prefix, context)
        
        if not result.suggestions or not filters:
            return result
        
        filtered_suggestions = []
        
        for suggestion in result.suggestions:
            include = True
            
            # Фильтр по минимальному match_score
            if "min_score" in filters:
                if suggestion.match_score < filters["min_score"]:
                    include = False
            
            # Фильтр по source
            if "sources" in filters:
                if suggestion.source not in filters["sources"]:
                    include = False
            
            # Фильтр по метаданным
            if "metadata_filters" in filters:
                for key, value in filters["metadata_filters"].items():
                    if suggestion.metadata.get(key) != value:
                        include = False
                        break
            
            if include:
                filtered_suggestions.append(suggestion)
        
        # Обновляем результат с отфильтрованными предложениями
        if filtered_suggestions:
            result.suggestions = filtered_suggestions
            # Корректируем confidence на основе отфильтрованных результатов
            if filtered_suggestions:
                max_score = max(s.match_score for s in filtered_suggestions)
                result.confidence = max_score
        else:
            result.result_type = AdvinationResultType.NO_MATCH
            result.suggestions = None
        
        return result
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику хранилища.
        """
        return self.storage.get_stats()


class ConfigurableAdivinator(Adivinator):
    """
    Расширенный адивинатор с дополнительными возможностями конфигурации.
    """
    
    def __init__(self, storage: CommandTrie, config: Dict[str, Any] = None):
        super().__init__(storage, config)
        
        # Дополнительные конфигурации
        self.advanced_config = {
            "enable_semantic_expansion": config.get("enable_semantic_expansion", False),
            "expansion_threshold": config.get("expansion_threshold", 0.7),
            "max_expansion_results": config.get("max_expansion_results", 2),
            "context_boost_factor": config.get("context_boost_factor", 0.3)
        }
        
        # Загруженные семантические модели (заглушка)
        self.semantic_models = {}
    
    def advinate_with_expansion(self, prefix: str, context: Dict[str, Any] = None) -> AdvinationResult:
        """
        Адивинация с семантическим расширением.
        """
        # Базовый поиск
        base_result = super().advinate(prefix, context)
        
        if not self.advanced_config["enable_semantic_expansion"]:
            return base_result
        
        # Если есть точные совпадения, расширяем их
        if base_result.result_type == AdvinationResultType.FOUND:
            expanded = self._expand_suggestions(base_result.suggestions, context)
            if expanded:
                # Объединяем и сортируем
                all_suggestions = base_result.suggestions + expanded
                all_suggestions.sort(key=lambda x: x.match_score, reverse=True)
                
                return AdvinationResult(
                    result_type=AdvinationResultType.FOUND,
                    suggestions=all_suggestions[:self.config["max_exact_results"]],
                    confidence=base_result.confidence,
                    raw_prefix=prefix,
                    raw_context=context
                )
        
        return base_result
    
    def _expand_suggestions(self,
                           suggestions: List[CommandSuggestion],
                           context: Dict[str, Any]) -> List[CommandSuggestion]:
        """
        Расширяет список предложений семантически похожими командами.
        """
        expanded = []
        
        for suggestion in suggestions:
            # Ищем семантически похожие команды
            similar_commands = self.storage.search_similar(
                suggestion.text,
                threshold=self.advanced_config["expansion_threshold"],
                limit=self.advanced_config["max_expansion_results"]
            )
            
            for similar in similar_commands:
                # Проверяем, что эта команда еще не в предложениях
                if similar["command"] not in [s.text for s in suggestions]:
                    # Учитываем контекст
                    context_boost = 0.0
                    if context.get("domain") and context["domain"] in similar.get("category", ""):
                        context_boost = self.advanced_config["context_boost_factor"]
                    
                    expanded.append(
                        CommandSuggestion(
                            text=similar["command"],
                            source="semantic_expansion",
                            match_score=similar.get("similarity", 0.5) * (1 + context_boost),
                            metadata={
                                "original_suggestion": suggestion.text,
                                "expansion_similarity": similar.get("similarity", 0),
                                **similar.get("metadata", {})
                            }
                        )
                    )
        
        return expanded