from typing import List, Dict, Any, Optional, Set
from .models import CommandSuggestion
import re


class CommandValidator:
    """
    Валидатор и адаптер команд.
    Знает о семантике и допустимости команд в различных контекстах.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "enable_adaptation": True,
            "known_tokens": {
                "shell": {"ls", "cd", "pwd", "mkdir", "rm", "cp", "mv", "find", "grep", "cat"},
                "git": {"git", "commit", "push", "pull", "clone", "branch", "merge", "status"},
                "docker": {"docker", "ps", "build", "run", "exec", "logs", "images"},
                "python": {"python", "pip", "install", "run", "test", "import"},
                "system": {"sudo", "apt", "yum", "systemctl", "service"}
            },
            "min_confidence_for_adaptation": 0.5,
            "adaptation_rules": {
                "path_substitution": True,
                "parameter_filling": True,
                "context_aware_expansion": True
            },
            "validation_rules": {
                "check_syntax": True,
                "check_permissions": False,
                "check_parameters": True,
                "warn_destructive": True
            },
            "domain_specific_rules": {
                "git": {
                    "required_params": ["message", "branch"],
                    "common_flags": ["-m", "-a", "-v"]
                },
                "docker": {
                    "required_params": ["image", "container"],
                    "common_flags": ["-it", "-d", "-p", "-v"]
                }
            }
        }
        
        # Кэш для хранения результатов валидации
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
        self.adaptation_cache: Dict[str, List[CommandSuggestion]] = {}
    
    def can_adapt(self, suggestions: List[CommandSuggestion], context: Dict[str, Any]) -> bool:
        """
        Проверяет, можно ли адаптировать предложения под контекст.
        
        Args:
            suggestions: Список предложенных команд
            context: Контекст выполнения
            
        Returns:
            True если можно адаптировать хотя бы одно предложение
        """
        if not self.config["enable_adaptation"]:
            return False
        
        if not suggestions:
            return False
        
        # Проверяем каждое предложение
        for suggestion in suggestions:
            if self._can_adapt_single(suggestion, context):
                return True
        
        return False
    
    def adapt(self, suggestions: List[CommandSuggestion], context: Dict[str, Any]) -> List[CommandSuggestion]:
        """
        Адаптирует предложенные команды под текущий контекст.
        
        Args:
            suggestions: Список предложенных команд
            context: Контекст выполнения
            
        Returns:
            Список адаптированных команд
        """
        if not self.can_adapt(suggestions, context):
            return []
        
        adapted = []
        
        for suggestion in suggestions:
            # Проверяем, нужно ли адаптировать это предложение
            if suggestion.match_score < self.config["min_confidence_for_adaptation"]:
                continue
            
            # Пробуем адаптировать каждое предложение
            adapted_versions = self._adapt_single(suggestion, context)
            adapted.extend(adapted_versions)
        
        # Удаляем дубликаты и сортируем по confidence
        unique_adapted = self._deduplicate_suggestions(adapted)
        unique_adapted.sort(key=lambda x: x.match_score, reverse=True)
        
        return unique_adapted
    
    def validate_command(self, command_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет команду на корректность в данном контексте.
        
        Args:
            command_text: Текст команды для валидации
            context: Контекст выполнения
            
        Returns:
            Словарь с результатами валидации
        """
        # Проверяем кэш
        cache_key = f"{command_text}_{str(context)}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key].copy()
        
        issues = []
        warnings = []
        metadata = {}
        
        # Разбираем команду на токены
        tokens = command_text.split()
        if not tokens:
            issues.append("Пустая команда")
            result = self._build_validation_result(command_text, False, issues, warnings, metadata)
            self.validation_cache[cache_key] = result
            return result
        
        first_token = tokens[0].lower()
        
        # 1. Проверка синтаксиса
        if self.config["validation_rules"]["check_syntax"]:
            syntax_issues = self._check_syntax(command_text, context)
            issues.extend(syntax_issues)
        
        # 2. Проверка известности команды
        known_commands = self._get_known_commands_for_context(context)
        if first_token not in known_commands:
            warnings.append(f"Неизвестная команда: {first_token}")
            metadata["unknown_command"] = True
        
        # 3. Проверка параметров для домен-специфичных команд
        domain = context.get("domain")
        if domain and domain in self.config["domain_specific_rules"]:
            domain_rules = self.config["domain_specific_rules"][domain]
            
            # Проверка обязательных параметров
            if "required_params" in domain_rules:
                for param in domain_rules["required_params"]:
                    param_pattern = f"-{param}" if len(param) == 1 else f"--{param}"
                    if param_pattern not in command_text and f"{{{param}}}" not in command_text:
                        issues.append(f"Отсутствует обязательный параметр: {param}")
        
        # 4. Проверка деструктивных команд
        if self.config["validation_rules"]["warn_destructive"]:
            destructive_commands = {"rm", "rmdir", "format", "dd", "shred"}
            if first_token in destructive_commands:
                warnings.append(f"Деструктивная команда: {first_token}")
                metadata["destructive"] = True
        
        # 5. Проверка разрешений (заглушка для реальной системы)
        if self.config["validation_rules"]["check_permissions"]:
            sudo_commands = {"apt", "systemctl", "service", "mount"}
            if first_token in sudo_commands and not command_text.startswith("sudo"):
                warnings.append(f"Для команды {first_token} могут потребоваться права sudo")
                metadata["requires_sudo"] = True
        
        # Анализ сложности команды
        complexity_score = self._calculate_complexity(command_text)
        metadata["complexity"] = complexity_score
        metadata["token_count"] = len(tokens)
        
        if complexity_score > 0.7:
            warnings.append("Сложная команда с множеством параметров")
        
        # Проверяем наличие общих ошибок
        common_errors = self._check_common_errors(command_text)
        issues.extend(common_errors)
        
        # Строим результат
        is_valid = len(issues) == 0
        result = self._build_validation_result(command_text, is_valid, issues, warnings, metadata)
        
        # Сохраняем в кэш
        self.validation_cache[cache_key] = result.copy()
        
        return result
    
    def batch_validate(self, commands: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Пакетная валидация нескольких команд.
        """
        return [self.validate_command(cmd, context) for cmd in commands]
    
    def suggest_corrections(self,
                           command_text: str,
                           context: Dict[str, Any]) -> List[CommandSuggestion]:
        """
        Предлагает исправления для невалидной команды.
        """
        validation_result = self.validate_command(command_text, context)
        
        if validation_result["is_valid"]:
            return []
        
        corrections = []
        
        # 1. Исправление опечаток в известных командах
        typo_corrections = self._suggest_typo_corrections(command_text, context)
        corrections.extend(typo_corrections)
        
        # 2. Исправление параметров
        param_corrections = self._suggest_parameter_corrections(command_text, context)
        corrections.extend(param_corrections)
        
        # 3. Предложение альтернативных команд
        alternative_corrections = self._suggest_alternatives(command_text, context)
        corrections.extend(alternative_corrections)
        
        # Удаляем дубликаты и сортируем
        unique_corrections = self._deduplicate_suggestions(corrections)
        unique_corrections.sort(key=lambda x: x.match_score, reverse=True)
        
        return unique_corrections[:5]  # Ограничиваем количество предложений
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику валидации.
        """
        total_validations = len(self.validation_cache)
        valid_commands = sum(1 for result in self.validation_cache.values() if result["is_valid"])
        
        return {
            "total_validations": total_validations,
            "valid_commands": valid_commands,
            "invalid_commands": total_validations - valid_commands,
            "cache_hit_ratio": 0.0,  # Можно рассчитать при реальном использовании
            "most_common_issues": self._get_most_common_issues()
        }
    
    # Вспомогательные методы
    
    def _can_adapt_single(self, suggestion: CommandSuggestion, context: Dict[str, Any]) -> bool:
        """Проверяет, можно ли адаптировать одно предложение."""
        # Проверяем известность первого токена
        tokens = suggestion.text.split()
        if not tokens:
            return False
        
        first_token = tokens[0].lower()
        known_commands = self._get_known_commands_for_context(context)
        
        return first_token in known_commands
    
    def _adapt_single(self, suggestion: CommandSuggestion, context: Dict[str, Any]) -> List[CommandSuggestion]:
        """Адаптирует одно предложение под контекст."""
        adapted_versions = []
        
        # Базовая адаптация: подстановка контекстных значений
        base_adapted = self._basic_adaptation(suggestion, context)
        if base_adapted:
            adapted_versions.append(base_adapted)
        
        # Расширенная адаптация: добавление контекстных флагов
        if self.config["adaptation_rules"]["context_aware_expansion"]:
            expanded = self._context_expansion(suggestion, context)
            adapted_versions.extend(expanded)
        
        # Адаптация параметров
        if self.config["adaptation_rules"]["parameter_filling"]:
            param_filled = self._parameter_filling(suggestion, context)
            if param_filled:
                adapted_versions.append(param_filled)
        
        return adapted_versions
    
    def _basic_adaptation(self, suggestion: CommandSuggestion, context: Dict[str, Any]) -> Optional[CommandSuggestion]:
        """Базовая адаптация с подстановкой путей и параметров."""
        adapted_text = suggestion.text
        
        # Подстановка пути из контекста
        if self.config["adaptation_rules"]["path_substitution"]:
            if "current_path" in context and "." in adapted_text:
                adapted_text = adapted_text.replace(".", context["current_path"])
            
            if "home_path" in context and "~" in adapted_text:
                adapted_text = adapted_text.replace("~", context["home_path"])
        
        # Подстановка параметров из контекста
        if "params" in context:
            for key, value in context["params"].items():
                placeholder = f"{{{key}}}"
                if placeholder in adapted_text:
                    adapted_text = adapted_text.replace(placeholder, str(value))
        
        # Если текст не изменился, возвращаем None
        if adapted_text == suggestion.text:
            return None
        
        return CommandSuggestion(
            text=adapted_text,
            source="adapted",
            match_score=suggestion.match_score * 0.9,  # Чуть ниже уверенность
            metadata={
                **suggestion.metadata,
                "original": suggestion.text,
                "adapted": True,
                "adaptation_type": "basic",
                "adaptation_context": {k: v for k, v in context.items() if k != "params"}
            }
        )
    
    def _context_expansion(self, suggestion: CommandSuggestion, context: Dict[str, Any]) -> List[CommandSuggestion]:
        """Расширение команды контекстно-специфичными флагами."""
        expanded = []
        tokens = suggestion.text.split()
        
        if not tokens:
            return expanded
        
        first_token = tokens[0].lower()
        domain = context.get("domain")
        
        # Добавление флагов для разных доменов
        if domain == "git" and first_token == "git":
            if len(tokens) > 1 and tokens[1] == "commit":
                expanded_text = suggestion.text + " -m 'Update'"
                expanded.append(
                    CommandSuggestion(
                        text=expanded_text,
                        source="context_expansion",
                        match_score=suggestion.match_score * 0.85,
                        metadata={
                            **suggestion.metadata,
                            "expansion": "added_commit_message",
                            "context_domain": domain
                        }
                    )
                )
        
        elif domain == "docker" and first_token == "docker":
            if len(tokens) > 1 and tokens[1] == "run":
                expanded_text = suggestion.text + " -it --rm"
                expanded.append(
                    CommandSuggestion(
                        text=expanded_text,
                        source="context_expansion",
                        match_score=suggestion.match_score * 0.85,
                        metadata={
                            **suggestion.metadata,
                            "expansion": "added_docker_flags",
                            "context_domain": domain
                        }
                    )
                )
        
        return expanded
    
    def _parameter_filling(self, suggestion: CommandSuggestion, context: Dict[str, Any]) -> Optional[CommandSuggestion]:
        """Заполнение недостающих параметров на основе контекста."""
        domain = context.get("domain")
        
        if not domain or domain not in self.config["domain_specific_rules"]:
            return None
        
        domain_rules = self.config["domain_specific_rules"][domain]
        if "common_flags" not in domain_rules:
            return None
        
        # Проверяем, есть ли уже эти флаги в команде
        command_text = suggestion.text
        added_flags = []
        
        for flag in domain_rules["common_flags"]:
            if flag not in command_text:
                command_text += f" {flag}"
                added_flags.append(flag)
        
        if not added_flags:
            return None
        
        return CommandSuggestion(
            text=command_text,
            source="parameter_filled",
            match_score=suggestion.match_score * 0.8,
            metadata={
                **suggestion.metadata,
                "added_flags": added_flags,
                "parameter_filling": True
            }
        )
    
    def _check_syntax(self, command_text: str, context: Dict[str, Any]) -> List[str]:
        """Проверка синтаксиса команды."""
        issues = []
        
        # Проверка незакрытых кавычек
        quote_count = command_text.count('"') + command_text.count("'")
        if quote_count % 2 != 0:
            issues.append("Незакрытые кавычки")
        
        # Проверка неправильных перенаправлений
        if ">>" in command_text and ">" in command_text.replace(">>", ""):
            issues.append("Неправильное использование перенаправления вывода")
        
        # Проверка синтаксиса для специфичных доменов
        domain = context.get("domain")
        if domain == "git":
            if "git commit" in command_text and "-m" in command_text:
                # Проверяем, что после -m есть сообщение
                parts = command_text.split("-m")
                if len(parts) < 2 or not parts[1].strip():
                    issues.append("Флаг -m требует сообщения коммита")
        
        return issues
    
    def _calculate_complexity(self, command_text: str) -> float:
        """Рассчитывает сложность команды."""
        tokens = command_text.split()
        if not tokens:
            return 0.0
        
        # Факторы сложности
        factors = {
            "token_count": min(len(tokens) / 10, 1.0),
            "has_pipes": 0.2 if "|" in command_text else 0.0,
            "has_redirects": 0.2 if ">" in command_text or "<" in command_text else 0.0,
            "has_subshell": 0.3 if "$(" in command_text or "`" in command_text else 0.0,
            "has_regex": 0.2 if any(c in command_text for c in ["*", "?", "[", "]"]) else 0.0
        }
        
        # Взвешенная сумма
        weights = {
            "token_count": 0.4,
            "has_pipes": 0.2,
            "has_redirects": 0.15,
            "has_subshell": 0.15,
            "has_regex": 0.1
        }
        
        complexity = sum(factors[key] * weights[key] for key in factors)
        return min(complexity, 1.0)
    
    def _check_common_errors(self, command_text: str) -> List[str]:
        """Проверка общих ошибок."""
        issues = []
        
        # Частые опечатки
        common_typos = {
            "sl": "ls",
            "cd..": "cd ..",
            "m kdir": "mkdir",
            "git stauts": "git status",
            "git commmit": "git commit"
        }
        
        for typo, correction in common_typos.items():
            if typo in command_text:
                issues.append(f"Возможная опечатка: '{typo}' -> '{correction}'")
        
        return issues
    
    def _suggest_typo_corrections(self, command_text: str, context: Dict[str, Any]) -> List[CommandSuggestion]:
        """Предлагает исправления опечаток."""
        corrections = []
        
        # Простые замены
        replacements = {
            "sl": "ls",
            "cd..": "cd ..",
            "m kdir": "mkdir",
            "git stauts": "git status",
            "git commmit": "git commit",
            "docer": "docker",
            "pythn": "python"
        }
        
        for wrong, correct in replacements.items():
            if wrong in command_text:
                corrected = command_text.replace(wrong, correct)
                corrections.append(
                    CommandSuggestion(
                        text=corrected,
                        source="typo_correction",
                        match_score=0.8,
                        metadata={
                            "original": command_text,
                            "corrected_from": wrong,
                            "corrected_to": correct
                        }
                    )
                )
        
        return corrections
    
    def _suggest_parameter_corrections(self, command_text: str, context: Dict[str, Any]) -> List[CommandSuggestion]:
        """Предлагает исправления параметров."""
        # Заглушка для реальной реализации
        return []
    
    def _suggest_alternatives(self, command_text: str, context: Dict[str, Any]) -> List[CommandSuggestion]:
        """Предлагает альтернативные команды."""
        # Заглушка для реальной реализации
        return []
    
    def _get_known_commands_for_context(self, context: Dict[str, Any]) -> Set[str]:
        """Возвращает набор известных команд для контекста."""
        known_commands = set()
        
        # Добавляем команды из всех доменов
        for domain_commands in self.config["known_tokens"].values():
            known_commands.update(domain_commands)
        
        # Добавляем домен-специфичные команды
        domain = context.get("domain")
        if domain and domain in self.config["known_tokens"]:
            known_commands.update(self.config["known_tokens"][domain])
        
        return known_commands
    
    def _deduplicate_suggestions(self, suggestions: List[CommandSuggestion]) -> List[CommandSuggestion]:
        """Удаляет дубликаты из списка предложений."""
        seen = set()
        unique = []
        
        for suggestion in suggestions:
            key = suggestion.text
            if key not in seen:
                seen.add(key)
                unique.append(suggestion)
        
        return unique
    
    def _build_validation_result(self,
                                command_text: str,
                                is_valid: bool,
                                issues: List[str],
                                warnings: List[str],
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Строит структурированный результат валидации."""
        return {
            "command": command_text,
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "metadata": metadata,
            "has_issues": len(issues) > 0,
            "has_warnings": len(warnings) > 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_most_common_issues(self) -> List[Dict[str, Any]]:
        """Возвращает наиболее частые проблемы из кэша."""
        issue_counts = {}
        
        for result in self.validation_cache.values():
            for issue in result.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Сортируем по частоте
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{"issue": issue, "count": count} for issue, count in sorted_issues[:10]]