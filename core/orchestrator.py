from typing import Dict, Any, Optional
import time
from datetime import datetime, timedelta
import uuid

from .models import *
from .adivinator import Adivinator
from .validator import CommandValidator
from .composer import CommandComposer


class DeferralStrategy:
    """Стратегия откладывания задач"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "base_delay_hours": 24,
            "max_retries": 3,
            "exponential_backoff": True,
            "priority_delays": {1: 1.0, 2: 0.5, 3: 0.25}  # множители задержки
        }
    
    def calculate_retry_time(self, attempt: int, priority: int = 1) -> Optional[datetime]:
        """Рассчитывает время следующей попытки"""
        if attempt >= self.config["max_retries"]:
            return None
        
        if self.config["exponential_backoff"]:
            delay_hours = self.config["base_delay_hours"] * (2 ** attempt)
        else:
            delay_hours = self.config["base_delay_hours"]
        
        # Корректируем по приоритету
        priority_factor = self.config["priority_delays"].get(priority, 1.0)
        delay_hours *= priority_factor
        
        return datetime.now() + timedelta(hours=delay_hours)


class OrchestrationMetrics:
    """Метрики оркестрации"""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "advination_results": {"FOUND": 0, "PARTIAL": 0, "NO_MATCH": 0},
            "outcomes": {},
            "processing_times": [],
            "errors": 0
        }
    
    def record_advination(self, result_type: AdvinationResultType):
        self.metrics["advination_results"][result_type.value] += 1
    
    def record_orchestration(self, outcome: OrchestrationOutcome, processing_time: float):
        self.metrics["requests_total"] += 1
        self.metrics["outcomes"][outcome.value] = self.metrics["outcomes"].get(outcome.value, 0) + 1
        self.metrics["processing_times"].append(processing_time)
    
    def record_error(self):
        self.metrics["errors"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку метрик"""
        times = self.metrics["processing_times"]
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            **self.metrics,
            "avg_processing_time": avg_time,
            "success_rate": (self.metrics["requests_total"] - self.metrics["errors"]) / 
                           max(self.metrics["requests_total"], 1)
        }


class ProductionOrchestrator:
    """
    Основной оркестратор workflow.
    Управляет всем процессом от запроса до результата.
    """
    
    def __init__(self,
                 advinator: Adivinator = None,
                 validator: CommandValidator = None,
                 composer: CommandComposer = None,
                 config: Dict[str, Any] = None):
        
        self.config = {
            "enable_adaptation": True,
            "enable_composition": True,
            "enable_deferral": True,
            "deferral_strategy": DeferralStrategy(),
            **config or {}
        }
        
        self.adivinator = advinator or Adivinator(None)
        self.validator = validator or CommandValidator()
        self.composer = composer or CommandComposer()
        self.metrics = OrchestrationMetrics()
        self.deferred_tasks: Dict[str, Dict] = {}
    
    def process_request(self,
                       prefix: str,
                       context: Dict[str, Any] = None,
                       user_id: str = None) -> OrchestrationResult:
        """
        Основной метод обработки запроса.
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # 1. Чистая адивинация
            adv_result = self.adivinator.advinate(prefix, context)
            self.metrics.record_advination(adv_result.result_type)
            
            # 2. Обработка результатов адивинации
            if adv_result.result_type == AdvinationResultType.FOUND:
                result = self._handle_found(adv_result, context)
            elif adv_result.result_type == AdvinationResultType.PARTIAL_FOUND:
                result = self._handle_partial(adv_result, context)
            else:  # NO_MATCH
                result = self._handle_no_match(prefix, context)
            
            # 3. Записываем метрики
            elapsed = time.time() - start_time
            self.metrics.record_orchestration(result.outcome, elapsed)
            
            # 4. Добавляем метаданные
            result.metadata.update({
                "processing_time_ms": elapsed * 1000,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.metrics.record_error()
            
            # При ошибке - откладываем
            if self.config["enable_deferral"]:
                task_id = self._defer_task(
                    prefix, context,
                    f"Ошибка обработки: {str(e)}",
                    priority=3
                )
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.DEFER,
                    task_id=task_id,
                    reason="Внутренняя ошибка системы",
                    retry_after=datetime.now() + timedelta(hours=1),
                    priority=3,
                    metadata={"error": str(e)}
                )
            else:
                # Если откладывание отключено, возвращаем ошибку
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.DEFER,
                    reason=f"Ошибка: {str(e)}",
                    metadata={"error": str(e), "deferral_disabled": True}
                )
    
    def _handle_found(self, adv_result: AdvinationResult, context: Dict) -> OrchestrationResult:
        """Обработка точных совпадений"""
        return OrchestrationResult(
            outcome=OrchestrationOutcome.SUGGEST_EXACT,
            suggestions=adv_result.suggestions,
            metadata={"source": "exact_match", "confidence": adv_result.confidence}
        )
    
    def _handle_partial(self, adv_result: AdvinationResult, context: Dict) -> OrchestrationResult:
        """Обработка частичных совпадений"""
        if self.config["enable_adaptation"]:
            # Пытаемся адаптировать
            adapted = self.validator.adapt(adv_result.suggestions, context)
            if adapted:
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.SUGGEST_ADAPTED,
                    suggestions=adapted,
                    metadata={"source": "adapted", "original_confidence": adv_result.confidence}
                )
        
        # Если адаптация невозможна - пробуем композицию
        if self.config["enable_composition"]:
            return self._try_composition(adv_result.raw_prefix, context)
        else:
            # Ничего не можем предложить
            return self._defer_or_fail(adv_result.raw_prefix, context,
                                     "Не удалось адаптировать частичные совпадения")
    
    def _handle_no_match(self, prefix: str, context: Dict) -> OrchestrationResult:
        """Обработка отсутствия совпадений"""
        if self.config["enable_composition"]:
            return self._try_composition(prefix, context)
        else:
            return self._defer_or_fail(prefix, context,
                                     "Нет совпадений и композиция отключена")
    
    def _try_composition(self, prefix: str, context: Dict) -> OrchestrationResult:
        """Попытка композиции новой команды"""
        composition_decision = self.composer.can_compose(prefix, context)
        
        if composition_decision["can_compose"]:
            # Начинаем диалог композиции
            dialog = self.composer.start_dialog(prefix, context)
            next_question = self.composer.get_next_question(dialog.dialog_id)
            
            if next_question:
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.START_DIALOG,
                    dialog_id=dialog.dialog_id,
                    first_question=next_question["text"],
                    question_type="text",
                    metadata={
                        "template": composition_decision["template"],
                        "reason": composition_decision["reason"]
                    }
                )
        
        # Не можем скомпоновать - откладываем
        return self._defer_or_fail(
            prefix, context,
            f"Невозможно скомпоновать: {composition_decision.get('reason', 'неизвестно')}"
        )
    
    def _defer_or_fail(self, prefix: str, context: Dict, reason: str) -> OrchestrationResult:
        """Откладывает задачу или возвращает ошибку"""
        if self.config["enable_deferral"]:
            task_id = self._defer_task(prefix, context, reason, priority=2)
            return OrchestrationResult(
                outcome=OrchestrationOutcome.DEFER,
                task_id=task_id,
                reason=reason,
                retry_after=datetime.now() + timedelta(hours=24),
                priority=2
            )
        else:
            # Если откладывание отключено, возвращаем пустой результат
            return OrchestrationResult(
                outcome=OrchestrationOutcome.DEFER,
                reason=reason,
                metadata={"deferral_disabled": True}
            )
    
    def _defer_task(self, prefix: str, context: Dict, reason: str, priority: int = 1) -> str:
        """Создаёт отложенную задачу"""
        task_id = str(uuid.uuid4())
        strategy = self.config["deferral_strategy"]
        retry_after = strategy.calculate_retry_time(0, priority)
        
        task = {
            "task_id": task_id,
            "prefix": prefix,
            "context": context,
            "reason": reason,
            "created_at": datetime.now(),
            "retry_after": retry_after,
            "attempt": 0,
            "priority": priority,
            "status": "deferred"
        }
        
        self.deferred_tasks[task_id] = task
        
        # Здесь можно сохранять задачи в файл или БД
        # self._save_deferred_task(task)
        
        return task_id
    
    def continue_dialog(self, dialog_id: str, answer: str) -> OrchestrationResult:
        """Продолжение диалога композиции"""
        try:
            result = self.composer.process_answer(dialog_id, answer)
            
            if result["status"] == "completed":
                # Команда скомпонована
                command = self.composer.get_dialog_result(dialog_id)
                
                if command:
                    # Сохраняем новую команду
                    self.adivinator.learn(command.text, {})
                    return OrchestrationResult(
                        outcome=OrchestrationOutcome.SUGGEST_EXACT,
                        suggestions=[
                            CommandSuggestion(
                                text=command.text,
                                source="composed",
                                match_score=0.9,
                                metadata={"dialog_id": dialog_id, "composed": True}
                            )
                        ],
                        metadata={"dialog_completed": True, "command_id": command.command_id}
                    )
                    
            elif result["status"] == "continue":
                # Продолжаем диалог
                next_q = result["next_question"]
                return OrchestrationResult(
                    outcome=OrchestrationOutcome.START_DIALOG,
                    dialog_id=dialog_id,
                    first_question=next_q["text"],
                    question_type="text",
                    metadata={"step": next_q["step"], "total_steps": next_q["total_steps"]}
                )
                
        except Exception as e:
            return OrchestrationResult(
                outcome=OrchestrationOutcome.DEFER,
                reason=f"Ошибка в диалоге: {str(e)}",
                metadata={"error": str(e), "dialog_id": dialog_id}
            )
        
        return OrchestrationResult(
            outcome=OrchestrationOutcome.DEFER,
            reason="Неизвестное состояние диалога",
            metadata={"dialog_id": dialog_id}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает текущие метрики"""
        return self.metrics.get_summary()
    
    def get_deferred_tasks(self) -> List[Dict[str, Any]]:
        """Возвращает список отложенных задач"""
        return list(self.deferred_tasks.values())
    
    def retry_deferred_task(self, task_id: str) -> bool:
        """Повторно обрабатывает отложенную задачу"""
        if task_id not in self.deferred_tasks:
            return False
        
        task = self.deferred_tasks[task_id]
        task["attempt"] += 1
        task["last_retry"] = datetime.now()
        
        try:
            # Пытаемся обработать задачу снова
            result = self.process_request(
                task["prefix"],
                task["context"],
                task.get("user_id")
            )
            
            if result.outcome != OrchestrationOutcome.DEFER:
                # Задача успешно обработана
                task["status"] = "resolved"
                task["resolved_at"] = datetime.now()
                task["resolution_result"] = result.to_dict()
                return True
            else:
                # Задача всё ещё не может быть обработана
                task["status"] = "still_deferred"
                return False
                
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            return False