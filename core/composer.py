from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime
from .models import DialogContext, Command, CommandSuggestion


class CommandComposer:
    """
    Композитор команд через диалог.
    Создаёт новые команды на основе уточняющих вопросов.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "max_questions": 5,
            "max_dialog_time_minutes": 10,
            "enable_smart_defaults": True,
            "allow_skipping": False,
            "question_templates": {
                "location": "В какой папке выполнить?",
                "pattern": "Какой шаблон поиска?",
                "message": "Введите сообщение коммита:",
                "branch": "В какую ветку?",
                "filename": "Имя файла:",
                "container": "Имя контейнера:",
                "image": "Имя образа Docker:",
                "port": "Порт:",
                "service": "Имя сервиса:",
                "command": "Какую команду выполнить?",
                "arguments": "Аргументы команды:"
            },
            "command_templates": {
                "file_search": {
                    "template": "find {location} -name '{pattern}' {options}",
                    "questions": ["location", "pattern"],
                    "defaults": {
                        "location": ".",
                        "options": "-type f"
                    }
                },
                "git_commit": {
                    "template": "git commit -m '{message}' {options}",
                    "questions": ["message"],
                    "defaults": {
                        "options": ""
                    }
                },
                "docker_run": {
                    "template": "docker run {options} {image} {command}",
                    "questions": ["image", "command"],
                    "defaults": {
                        "options": "-it --rm"
                    }
                },
                "service_control": {
                    "template": "systemctl {action} {service}",
                    "questions": ["action", "service"],
                    "defaults": {}
                },
                "file_operation": {
                    "template": "{operation} {source} {destination}",
                    "questions": ["operation", "source", "destination"],
                    "defaults": {}
                }
            },
            "smart_defaults": {
                "location": {
                    "patterns": ["текущая", "здесь", "."]: ".",
                    "patterns": ["домашняя", "home", "~"]: "~",
                    "patterns": ["корневая", "root", "/"]: "/"
                },
                "operation": {
                    "patterns": ["скопируй", "копирование", "copy"]: "cp",
                    "patterns": ["перемести", "перемещение", "move"]: "mv",
                    "patterns": ["удали", "удаление", "delete"]: "rm"
                }
            }
        }
        
        self.active_dialogs: Dict[str, DialogContext] = {}
        self.completed_dialogs: Dict[str, DialogContext] = {}
        
        # Статистика композиции
        self.stats = {
            "dialogs_started": 0,
            "dialogs_completed": 0,
            "dialogs_abandoned": 0,
            "average_questions_per_dialog": 0,
            "commands_composed": 0,
            "template_usage": {}
        }
    
    def can_compose(self, prefix: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Определяет, можно ли начать диалог композиции для этого запроса.
        
        Returns:
            Словарь с решением о возможности композиции
        """
        decision = {
            "can_compose": False,
            "reason": "",
            "template": None,
            "confidence": 0.0,
            "questions": []
        }
        
        # Слишком короткий префикс
        if len(prefix.strip()) < 2:
            decision["reason"] = "Слишком общий запрос"
            return decision
        
        prefix_lower = prefix.lower()
        context_domain = context.get("domain", "").lower()
        
        # Определяем шаблон на основе префикса и контекста
        template_match = self._match_template(prefix_lower, context_domain)
        
        if template_match:
            decision.update({
                "can_compose": True,
                "reason": f"Определён шаблон: {template_match['name']}",
                "template": template_match["name"],
                "confidence": template_match["confidence"],
                "questions": template_match["questions"],
                "template_config": template_match["config"]
            })
            return decision
        
        # Общий шаблон для неизвестных команд
        decision.update({
            "can_compose": True,
            "reason": "Общая композиция",
            "template": "generic",
            "confidence": 0.3,
            "questions": ["command", "arguments"],
            "template_config": {
                "template": "{command} {arguments}",
                "questions": ["command", "arguments"],
                "defaults": {}
            }
        })
        
        return decision
    
    def start_dialog(self, prefix: str, context: Dict[str, Any]) -> DialogContext:
        """
        Начинает новый диалог композиции.
        
        Returns:
            Контекст начатого диалога
        """
        composition_decision = self.can_compose(prefix, context)
        
        if not composition_decision["can_compose"]:
            raise ValueError(f"Невозможно начать диалог: {composition_decision['reason']}")
        
        # Создаём контекст диалога
        dialog = DialogContext(
            user_intent=prefix,
            state="initial",
            collected_answers={
                "prefix": prefix,
                "context": context,
                "template": composition_decision["template"],
                "template_config": composition_decision.get("template_config", {}),
                "questions": composition_decision["questions"],
                "started_at": datetime.now().isoformat()
            }
        )
        
        # Сохраняем в активных диалогах
        self.active_dialogs[dialog.dialog_id] = dialog
        
        # Обновляем статистику
        self.stats["dialogs_started"] += 1
        template_name = composition_decision["template"]
        self.stats["template_usage"][template_name] = self.stats["template_usage"].get(template_name, 0) + 1
        
        return dialog
    
    def get_next_question(self, dialog_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает следующий вопрос для диалога.
        
        Returns:
            Словарь с информацией о вопросе или None если диалог завершён
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return None
        
        template_config = dialog.collected_answers.get("template_config", {})
        questions = dialog.collected_answers.get("questions", [])
        
        # Проверяем, все ли вопросы заданы
        if dialog.current_question_idx >= len(questions):
            # Все вопросы заданы - можно завершать диалог
            return None
        
        # Получаем следующий вопрос
        question_key = questions[dialog.current_question_idx]
        question_config = self._get_question_config(question_key, dialog)
        
        return {
            "question_id": question_key,
            "text": question_config["text"],
            "type": question_config["type"],
            "options": question_config.get("options"),
            "default": question_config.get("default"),
            "hint": question_config.get("hint"),
            "dialog_id": dialog_id,
            "step": dialog.current_question_idx + 1,
            "total_steps": len(questions),
            "required": question_config.get("required", True)
        }
    
    def process_answer(self, dialog_id: str, answer: str) -> Dict[str, Any]:
        """
        Обрабатывает ответ пользователя.
        
        Returns:
            Словарь с результатом обработки
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return {"error": "Диалог не найден", "status": "error"}
        
        # Обрабатываем специальные команды
        if answer.lower() in ["отмена", "cancel", "стоп"]:
            return self._cancel_dialog(dialog_id)
        
        if answer.lower() in ["пропустить", "skip", "далее"] and self.config["allow_skipping"]:
            return self._skip_question(dialog_id)
        
        # Нормализуем ответ
        normalized_answer = self._normalize_answer(answer, dialog)
        
        # Сохраняем ответ
        current_question_idx = dialog.current_question_idx
        questions = dialog.collected_answers.get("questions", [])
        
        if current_question_idx < len(questions):
            question_key = questions[current_question_idx]
            
            # Валидируем ответ
            validation_result = self._validate_answer(question_key, normalized_answer, dialog)
            
            if not validation_result["is_valid"]:
                return {
                    "status": "validation_error",
                    "error": validation_result["error"],
                    "hint": validation_result.get("hint"),
                    "dialog_id": dialog_id,
                    "question_key": question_key
                }
            
            # Сохраняем валидный ответ
            dialog.collected_answers[question_key] = normalized_answer
            dialog.current_question_idx += 1
            
            # Добавляем метаданные
            if "answers_metadata" not in dialog.collected_answers:
                dialog.collected_answers["answers_metadata"] = {}
            
            dialog.collected_answers["answers_metadata"][question_key] = {
                "original_answer": answer,
                "normalized_answer": normalized_answer,
                "answered_at": datetime.now().isoformat(),
                "validation_passed": True
            }
        
        # Проверяем, завершён ли диалог
        if dialog.current_question_idx >= len(questions):
            # Композиция завершена
            composed_command = self._compose_command(dialog)
            dialog.generated_command = composed_command
            dialog.state = "completed"
            
            # Перемещаем в завершённые диалоги
            self.completed_dialogs[dialog_id] = dialog
            if dialog_id in self.active_dialogs:
                del self.active_dialogs[dialog_id]
            
            # Обновляем статистику
            self.stats["dialogs_completed"] += 1
            self.stats["commands_composed"] += 1
            
            # Обновляем среднее количество вопросов
            total_questions = len(questions)
            current_avg = self.stats["average_questions_per_dialog"]
            total_completed = self.stats["dialogs_completed"]
            
            if total_completed == 1:
                self.stats["average_questions_per_dialog"] = total_questions
            else:
                self.stats["average_questions_per_dialog"] = (
                    (current_avg * (total_completed - 1) + total_questions) / total_completed
                )
            
            return {
                "status": "completed",
                "command": composed_command.text,
                "command_object": composed_command,
                "dialog_id": dialog_id,
                "metadata": {
                    "template": dialog.collected_answers.get("template"),
                    "answers": dialog.collected_answers,
                    "composed_at": datetime.now().isoformat()
                }
            }
        else:
            # Есть ещё вопросы
            return {
                "status": "continue",
                "dialog_id": dialog_id,
                "next_question": self.get_next_question(dialog_id)
            }
    
    def get_dialog_result(self, dialog_id: str) -> Optional[Command]:
        """Возвращает скомпонованную команду по ID диалога"""
        dialog = self.completed_dialogs.get(dialog_id)
        if dialog and dialog.state == "completed":
            return dialog.generated_command
        return None
    
    def resume_dialog(self, dialog_id: str) -> Optional[Dict[str, Any]]:
        """
        Возобновляет прерванный диалог.
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            dialog = self.completed_dialogs.get(dialog_id)
            if dialog:
                return {
                    "status": "already_completed",
                    "command": dialog.generated_command.text if dialog.generated_command else None,
                    "dialog_id": dialog_id
                }
            return None
        
        # Проверяем время жизни диалога
        started_at_str = dialog.collected_answers.get("started_at")
        if started_at_str:
            try:
                started_at = datetime.fromisoformat(started_at_str)
                elapsed_minutes = (datetime.now() - started_at).total_seconds() / 60
                
                if elapsed_minutes > self.config["max_dialog_time_minutes"]:
                    self._abandon_dialog(dialog_id)
                    return {
                        "status": "expired",
                        "dialog_id": dialog_id,
                        "reason": f"Диалог истёк ({elapsed_minutes:.1f} минут)"
                    }
            except ValueError:
                pass
        
        return {
            "status": "active",
            "dialog_id": dialog_id,
            "current_question": self.get_next_question(dialog_id),
            "progress": {
                "current": dialog.current_question_idx,
                "total": len(dialog.collected_answers.get("questions", []))
            }
        }
    
    def suggest_auto_complete(self, dialog_id: str, partial_answer: str) -> List[str]:
        """
        Предлагает варианты автодополнения для ответа.
        """
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return []
        
        current_question_idx = dialog.current_question_idx
        questions = dialog.collected_answers.get("questions", [])
        
        if current_question_idx >= len(questions):
            return []
        
        question_key = questions[current_question_idx]
        suggestions = self._get_auto_complete_suggestions(question_key, partial_answer, dialog)
        
        return suggestions[:5]  # Ограничиваем количество
    
    def get_dialog_summary(self, dialog_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает сводку по диалогу.
        """
        dialog = self.active_dialogs.get(dialog_id) or self.completed_dialogs.get(dialog_id)
        if not dialog:
            return None
        
        return {
            "dialog_id": dialog_id,
            "state": dialog.state,
            "user_intent": dialog.user_intent,
            "template": dialog.collected_answers.get("template"),
            "progress": {
                "current": dialog.current_question_idx,
                "total": len(dialog.collected_answers.get("questions", [])),
                "answers": list(dialog.collected_answers.keys())
            },
            "generated_command": dialog.generated_command.text if dialog.generated_command else None,
            "started_at": dialog.collected_answers.get("started_at"),
            "context": dialog.collected_answers.get("context", {})
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику композиции.
        """
        return {
            **self.stats,
            "active_dialogs": len(self.active_dialogs),
            "completed_dialogs": len(self.completed_dialogs),
            "success_rate": (
                self.stats["dialogs_completed"] / 
                max(self.stats["dialogs_started"], 1)
            )
        }
    
    # Вспомогательные методы
    
    def _match_template(self, prefix: str, context_domain: str) -> Optional[Dict[str, Any]]:
        """Сопоставляет префикс с шаблоном команды."""
        matched_templates = []
        
        # Ищем в конфигурационных шаблонах
        for template_name, template_config in self.config["command_templates"].items():
            confidence = self._calculate_template_confidence(prefix, context_domain, template_config)
            
            if confidence > 0.5:  # Порог уверенности
                matched_templates.append({
                    "name": template_name,
                    "config": template_config,
                    "confidence": confidence,
                    "questions": template_config.get("questions", [])
                })
        
        # Ищем по ключевым словам
        keyword_matches = self._match_by_keywords(prefix, context_domain)
        matched_templates.extend(keyword_matches)
        
        if not matched_templates:
            return None
        
        # Выбираем шаблон с наибольшей уверенностью
        matched_templates.sort(key=lambda x: x["confidence"], reverse=True)
        return matched_templates[0]
    
    def _calculate_template_confidence(self, prefix: str, context_domain: str, template_config: Dict) -> float:
        """Рассчитывает уверенность в совпадении с шаблоном."""
        confidence = 0.0
        
        # Анализ шаблона
        template_str = str(template_config.get("template", "")).lower()
        
        # Проверяем наличие ключевых слов из шаблона в префиксе
        keywords = self._extract_keywords(template_str)
        for keyword in keywords:
            if keyword in prefix:
                confidence += 0.2
        
        # Учитываем домен контекста
        if context_domain:
            # Простая эвристика: если домен совпадает с предполагаемым
            if context_domain in template_str:
                confidence += 0.3
        
        # Ограничиваем максимальное значение
        return min(confidence, 1.0)
    
    def _match_by_keywords(self, prefix: str, context_domain: str) -> List[Dict[str, Any]]:
        """Сопоставляет по ключевым словам."""
        matches = []
        
        keyword_patterns = {
            "найди": {"template": "file_search", "confidence": 0.8},
            "поиск": {"template": "file_search", "confidence": 0.8},
            "find": {"template": "file_search", "confidence": 0.8},
            "search": {"template": "file_search", "confidence": 0.8},
            "git": {"template": "git_commit", "confidence": 0.7},
            "коммит": {"template": "git_commit", "confidence": 0.7},
            "docker": {"template": "docker_run", "confidence": 0.7},
            "контейнер": {"template": "docker_run", "confidence": 0.7},
            "запусти": {"template": "service_control", "confidence": 0.6},
            "останови": {"template": "service_control", "confidence": 0.6},
            "скопируй": {"template": "file_operation", "confidence": 0.6},
            "удали": {"template": "file_operation", "confidence": 0.6}
        }
        
        for keyword, template_info in keyword_patterns.items():
            if keyword in prefix:
                template_name = template_info["template"]
                if template_name in self.config["command_templates"]:
                    matches.append({
                        "name": template_name,
                        "config": self.config["command_templates"][template_name],
                        "confidence": template_info["confidence"],
                        "questions": self.config["command_templates"][template_name].get("questions", [])
                    })
        
        return matches
    
    def _get_question_config(self, question_key: str, dialog: DialogContext) -> Dict[str, Any]:
        """Возвращает конфигурацию вопроса."""
        base_config = {
            "text": self.config["question_templates"].get(
                question_key, 
                f"Уточните {question_key.replace('_', ' ')}:"
            ),
            "type": "text",
            "required": True
        }
        
        # Добавляем умные подсказки и значения по умолчанию
        if self.config["enable_smart_defaults"]:
            smart_default = self._get_smart_default(question_key, dialog)
            if smart_default:
                base_config["default"] = smart_default["value"]
                base_config["hint"] = smart_default.get("hint")
        
        # Для определённых типов вопросов меняем конфигурацию
        if question_key in ["operation", "action"]:
            base_config["type"] = "choice"
            base_config["options"] = self._get_options_for_question(question_key, dialog)
        
        elif question_key in ["location"]:
            base_config["type"] = "path"
            base_config["hint"] = "Можно использовать абсолютные или относительные пути"
        
        elif question_key in ["port"]:
            base_config["type"] = "number"
            base_config["hint"] = "Введите номер порта (1-65535)"
        
        return base_config
    
    def _get_smart_default(self, question_key: str, dialog: DialogContext) -> Optional[Dict[str, Any]]:
        """Возвращает умное значение по умолчанию для вопроса."""
        if question_key not in self.config["smart_defaults"]:
            return None
        
        smart_config = self.config["smart_defaults"][question_key]
        
        # Проверяем контекст диалога
        context = dialog.collected_answers.get("context", {})
        user_intent = dialog.user_intent.lower()
        
        # Ищем совпадения с паттернами
        for pattern_key, pattern_config in smart_config.items():
            if "patterns" in pattern_config:
                for pattern in pattern_config["patterns"]:
                    if pattern in user_intent:
                        return {
                            "value": pattern_config.get("value", ""),
                            "hint": pattern_config.get("hint", f"На основе: '{pattern}'")
                        }
        
        # Проверяем контекстные значения
        if question_key in context:
            return {
                "value": context[question_key],
                "hint": "Из контекста"
            }
        
        # Возвращаем глобальные умолчания из шаблона
        template_config = dialog.collected_answers.get("template_config", {})
        defaults = template_config.get("defaults", {})
        
        if question_key in defaults:
            return {
                "value": defaults[question_key],
                "hint": "Значение по умолчанию для шаблона"
            }
        
        return None
    
    def _get_options_for_question(self, question_key: str, dialog: DialogContext) -> List[str]:
        """Возвращает варианты выбора для вопроса."""
        if question_key == "operation":
            return ["cp", "mv", "rm", "mkdir", "touch"]
        elif question_key == "action":
            return ["start", "stop", "restart", "status", "enable", "disable"]
        
        return []
    
    def _normalize_answer(self, answer: str, dialog: DialogContext) -> str:
        """Нормализует ответ пользователя."""
        normalized = answer.strip()
        
        # Для путей: заменяем тильду на домашнюю директорию
        current_question_idx = dialog.current_question_idx
        questions = dialog.collected_answers.get("questions", [])
        
        if current_question_idx < len(questions):
            question_key = questions[current_question_idx]
            
            if question_key == "location" and normalized.startswith("~"):
                # В реальной системе здесь была бы замена на абсолютный путь
                pass
        
        # Удаляем лишние кавычки
        if (normalized.startswith('"') and normalized.endswith('"')) or \
           (normalized.startswith("'") and normalized.endswith("'")):
            normalized = normalized[1:-1]
        
        return normalized
    
    def _validate_answer(self, question_key: str, answer: str, dialog: DialogContext) -> Dict[str, Any]:
        """Валидирует ответ на вопрос."""
        result = {
            "is_valid": True,
            "error": None,
            "hint": None
        }
        
        # Проверка на пустоту для обязательных вопросов
        question_config = self._get_question_config(question_key, dialog)
        if question_config.get("required", True) and not answer:
            result.update({
                "is_valid": False,
                "error": "Ответ не может быть пустым",
                "hint": "Пожалуйста, введите значение"
            })
            return result
        
        # Специфичные проверки для разных типов вопросов
        if question_key == "port":
            try:
                port = int(answer)
                if not (1 <= port <= 65535):
                    result.update({
                        "is_valid": False,
                        "error": f"Некорректный порт: {port}",
                        "hint": "Порт должен быть в диапазоне 1-65535"
                    })
            except ValueError:
                result.update({
                    "is_valid": False,
                    "error": f"Нечисловое значение: {answer}",
                    "hint": "Введите число от 1 до 65535"
                })
        
        elif question_key == "location":
            # Простая проверка пути
            if ".." in answer and answer.count("..") > 3:
                result.update({
                    "is_valid": False,
                    "error": "Слишком много '..' в пути",
                    "hint": "Проверьте корректность пути"
                })
        
        elif question_key == "filename":
            # Проверка имени файла
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in invalid_chars:
                if char in answer:
                    result.update({
                        "is_valid": False,
                        "error": f"Недопустимый символ в имени файла: {char}",
                        "hint": "Имя файла не должно содержать: / \\ : * ? \" < > |"
                    })
                    break
        
        return result
    
    def _compose_command(self, dialog: DialogContext) -> Command:
        """Создаёт команду на основе ответов."""
        template_config = dialog.collected_answers.get("template_config", {})
        template = template_config.get("template", "{command}")
        
        # Начинаем с шаблона
        command_text = template
        
        # Заменяем плейсхолдеры
        for key, value in dialog.collected_answers.items():
            if key not in ["template", "context", "questions", "template_config", 
                          "prefix", "started_at", "answers_metadata"]:
                placeholder = f"{{{key}}}"
                if placeholder in command_text:
                    command_text = command_text.replace(placeholder, str(value))
        
        # Применяем умолчания из шаблона
        defaults = template_config.get("defaults", {})
        for key, value in defaults.items():
            placeholder = f"{{{key}}}"
            if placeholder in command_text:
                command_text = command_text.replace(placeholder, str(value))
        
        # Убираем лишние пробелы
        command_text = " ".join(command_text.split())
        
        # Создаём объект команды
        return Command(
            text=command_text,
            category=dialog.collected_answers.get("template", "generic"),
            status="generated",
            parameters={
                k: v for k, v in dialog.collected_answers.items()
                if k not in ["template", "context", "questions", "template_config", 
                            "prefix", "started_at", "answers_metadata"]
            },
            description=f"Скомпонована из диалога {dialog.dialog_id}",
            tags=["composed", dialog.collected_answers.get("template", "generic")]
        )
    
    def _cancel_dialog(self, dialog_id: str) -> Dict[str, Any]:
        """Отменяет диалог."""
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return {"status": "error", "error": "Диалог не найден"}
        
        # Удаляем из активных
        if dialog_id in self.active_dialogs:
            del self.active_dialogs[dialog_id]
        
        # Обновляем статистику
        self.stats["dialogs_abandoned"] += 1
        
        return {
            "status": "cancelled",
            "dialog_id": dialog_id,
            "reason": "Отменено пользователем"
        }
    
    def _skip_question(self, dialog_id: str) -> Dict[str, Any]:
        """Пропускает текущий вопрос."""
        dialog = self.active_dialogs.get(dialog_id)
        if not dialog:
            return {"status": "error", "error": "Диалог не найден"}
        
        # Используем умолчание или пустое значение
        current_question_idx = dialog.current_question_idx
        questions = dialog.collected_answers.get("questions", [])
        
        if current_question_idx < len(questions):
            question_key = questions[current_question_idx]
            question_config = self._get_question_config(question_key, dialog)
            
            # Используем значение по умолчанию или пустую строку
            default_value = question_config.get("default", "")
            dialog.collected_answers[question_key] = default_value
            dialog.current_question_idx += 1
            
            # Помечаем как пропущенный
            if "answers_metadata" not in dialog.collected_answers:
                dialog.collected_answers["answers_metadata"] = {}
            
            dialog.collected_answers["answers_metadata"][question_key] = {
                "skipped": True,
                "used_default": bool(default_value),
                "default_value": default_value,
                "skipped_at": datetime.now().isoformat()
            }
        
        return {
            "status": "continue",
            "dialog_id": dialog_id,
            "next_question": self.get_next_question(dialog_id),
            "skipped": True
        }
    
    def _abandon_dialog(self, dialog_id: str):
        """Отмечает диалог как брошенный."""
        if dialog_id in self.active_dialogs:
            del self.active_dialogs[dialog_id]
            self.stats["dialogs_abandoned"] += 1
    
    def _get_auto_complete_suggestions(self, question_key: str, partial: str, dialog: DialogContext) -> List[str]:
        """Возвращает предложения для автодополнения."""
        suggestions = []
        
        if question_key == "location":
            # Простые подсказки для путей
            common_paths = [".", "..", "~", "/", "/tmp", "/var", "/etc"]
            for path in common_paths:
                if path.startswith(partial):
                    suggestions.append(path)
        
        elif question_key == "branch":
            # Подсказки для веток git
            common_branches = ["main", "master", "develop", "feature", "hotfix"]
            for branch in common_branches:
                if branch.startswith(partial):
                    suggestions.append(branch)
        
        return suggestions
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Извлекает ключевые слова из текста."""
        # Простая реализация - разбиваем на слова и фильтруем
        words = text.lower().split()
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but"}
        
        keywords = []
        for word in words:
            # Убираем специальные символы
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                keywords.append(clean_word)
        
        return list(set(keywords))  # Убираем дубликаты