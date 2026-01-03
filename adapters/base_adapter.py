"""
Base adapter interface for semantic advination system.
Defines the contract for all platform-specific adapters.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import json
import uuid

from core.models import (
    OrchestrationResult,
    CommandSuggestion,
    AdvinationResult,
    Command,
)
from core.orchestrator import ProductionOrchestrator


class BaseAdapter(ABC):
    """
    Abstract base class for all platform adapters.
    Defines the interface between platform-specific input/output
    and the core advination system.
    """
    
    def __init__(self, 
                 orchestrator: Optional[ProductionOrchestrator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.
        
        Args:
            orchestrator: Core orchestrator instance
            config: Adapter-specific configuration
        """
        self.orchestrator = orchestrator
        self.config = self._merge_default_config(config or {})
        self.adapter_id = str(uuid.uuid4())[:8]
        
        # State management
        self._state = {
            "initialized": False,
            "active_sessions": {},
            "processed_requests": 0,
            "successful_responses": 0,
            "errors": 0,
            "started_at": datetime.now().isoformat(),
        }
        
        # Callback registry
        self._callbacks = {
            "on_request": [],
            "on_response": [],
            "on_error": [],
            "on_session_start": [],
            "on_session_end": [],
        }
        
        # Platform-specific context
        self._platform_context = self._initialize_platform_context()
        
        # Initialize
        self._initialize()
    
    def _merge_default_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with default adapter config."""
        default_config = {
            "enable_learning": True,
            "enable_history": True,
            "max_history_size": 1000,
            "response_timeout_seconds": 30,
            "auto_confirm_threshold": 0.9,
            "retry_on_error": True,
            "max_retries": 3,
            "log_level": "INFO",
            "platform_specific": {},
        }
        
        # Deep merge
        merged = default_config.copy()
        self._deep_update(merged, user_config)
        return merged
    
    def _deep_update(self, target: Dict, source: Dict):
        """Recursively update dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    @abstractmethod
    def _initialize_platform_context(self) -> Dict[str, Any]:
        """
        Initialize platform-specific context.
        Must be implemented by subclasses.
        
        Returns:
            Platform context dictionary
        """
        pass
    
    def _initialize(self):
        """Initialize adapter internals."""
        # Initialize history storage if enabled
        if self.config["enable_history"]:
            self._history = []
            self._session_history = {}
        
        # Initialize learning cache if enabled
        if self.config["enable_learning"]:
            self._learning_cache = []
        
        # Mark as initialized
        self._state["initialized"] = True
        self._state["initialized_at"] = datetime.now().isoformat()
    
    # Core adapter interface
    @abstractmethod
    def normalize_input(self, raw_input: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize platform-specific input to standard format.
        
        Args:
            raw_input: Raw input from platform
            context: Additional context
            
        Returns:
            Normalized input dictionary with keys:
            - prefix: str - the text to advinate
            - context: Dict - normalized context
            - metadata: Dict - input metadata
        """
        pass
    
    @abstractmethod
    def format_response(self, 
                       result: OrchestrationResult,
                       original_input: Any) -> Any:
        """
        Format orchestration result for platform.
        
        Args:
            result: Orchestration result
            original_input: Original platform input
            
        Returns:
            Platform-specific response
        """
        pass
    
    @abstractmethod
    def create_request_context(self, 
                              raw_input: Any,
                              session_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create context for advination request.
        
        Args:
            raw_input: Raw platform input
            session_data: Session data if available
            
        Returns:
            Context dictionary
        """
        pass
    
    # Main processing methods
    def process(self, raw_input: Any, **kwargs) -> Any:
        """
        Main processing method - entry point for all requests.
        
        Args:
            raw_input: Raw input from platform
            **kwargs: Additional parameters
            
        Returns:
            Platform-formatted response
        """
        self._state["processed_requests"] += 1
        
        # Call request callbacks
        self._call_callbacks("on_request", raw_input, kwargs)
        
        try:
            # Normalize input
            normalized = self.normalize_input(raw_input, kwargs)
            prefix = normalized.get("prefix", "")
            input_context = normalized.get("context", {})
            metadata = normalized.get("metadata", {})
            
            # Create request context
            request_context = self.create_request_context(raw_input, {
                **input_context,
                "metadata": metadata,
                "adapter_id": self.adapter_id,
                "platform": self.get_platform_name(),
            })
            
            # Merge with adapter platform context
            full_context = {
                **self._platform_context,
                **request_context,
                "adapter_config": self.config,
            }
            
            # Add session ID if available
            session_id = kwargs.get("session_id")
            if session_id:
                full_context["session_id"] = session_id
                self._track_session(session_id, prefix, full_context)
            
            # Process through orchestrator
            result = self.orchestrator.process_request(
                prefix=prefix,
                context=full_context,
                user_id=kwargs.get("user_id"),
            )
            
            # Handle different outcomes
            response = self._handle_orchestration_result(
                result, raw_input, full_context, kwargs
            )
            
            # Learn from successful processing
            if self.config["enable_learning"] and result.outcome.value in ["SUGGEST_EXACT", "SUGGEST_ADAPTED"]:
                self._learn_from_result(prefix, result, full_context)
            
            # Store in history
            if self.config["enable_history"]:
                self._store_in_history(prefix, result, full_context, metadata)
            
            # Call response callbacks
            self._call_callbacks("on_response", result, response)
            
            self._state["successful_responses"] += 1
            return response
            
        except Exception as e:
            self._state["errors"] += 1
            error_response = self._handle_error(e, raw_input, kwargs)
            
            # Call error callbacks
            self._call_callbacks("on_error", e, raw_input, error_response)
            
            return error_response
    
    def _handle_orchestration_result(self,
                                    result: OrchestrationResult,
                                    raw_input: Any,
                                    context: Dict[str, Any],
                                    kwargs: Dict[str, Any]) -> Any:
        """
        Handle different orchestration outcomes.
        
        Args:
            result: Orchestration result
            raw_input: Original input
            context: Request context
            kwargs: Additional parameters
            
        Returns:
            Formatted response
        """
        # Handle dialog continuation
        if result.outcome.value == "START_DIALOG" and result.dialog_id:
            # Store dialog context for continuation
            self._store_dialog_context(
                result.dialog_id,
                raw_input,
                context,
                kwargs.get("session_id")
            )
        
        # Handle deferred tasks
        elif result.outcome.value == "DEFER" and result.task_id:
            self._store_deferred_task(
                result.task_id,
                result,
                context,
                kwargs.get("session_id")
            )
        
        # Format the response for platform
        return self.format_response(result, raw_input)
    
    def _handle_error(self, 
                     error: Exception,
                     raw_input: Any,
                     kwargs: Dict[str, Any]) -> Any:
        """
        Handle processing errors.
        
        Args:
            error: Exception that occurred
            raw_input: Original input
            kwargs: Additional parameters
            
        Returns:
            Error response formatted for platform
        """
        # Create error result
        error_result = OrchestrationResult(
            outcome="DEFER",  # Using string literal for simplicity
            reason=f"Adapter error: {str(error)}",
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "adapter_id": self.adapter_id,
                "platform": self.get_platform_name(),
                "timestamp": datetime.now().isoformat(),
                "input": str(raw_input)[:100],  # Truncate for safety
            }
        )
        
        # Try to format error response
        try:
            return self.format_response(error_result, raw_input)
        except Exception as format_error:
            # Fallback error response
            return self._create_fallback_error_response(error, format_error)
    
    def _create_fallback_error_response(self,
                                       original_error: Exception,
                                       format_error: Exception) -> Dict[str, Any]:
        """Create fallback error response when formatting fails."""
        return {
            "error": True,
            "message": f"Processing failed: {str(original_error)}",
            "details": {
                "original_error": str(original_error),
                "format_error": str(format_error),
                "adapter_id": self.adapter_id,
                "timestamp": datetime.now().isoformat(),
            },
            "suggestions": [],
            "suggestion_type": "error",
        }
    
    # Dialog handling
    def continue_dialog(self,
                       dialog_id: str,
                       user_response: Any,
                       **kwargs) -> Any:
        """
        Continue an existing dialog.
        
        Args:
            dialog_id: Dialog identifier
            user_response: User's response
            **kwargs: Additional parameters
            
        Returns:
            Dialog continuation response
        """
        try:
            # Get dialog context
            dialog_context = self._get_dialog_context(dialog_id)
            if not dialog_context:
                raise ValueError(f"Dialog {dialog_id} not found or expired")
            
            # Normalize response
            normalized_response = self._normalize_dialog_response(
                user_response,
                dialog_context
            )
            
            # Continue dialog in orchestrator
            result = self.orchestrator.continue_dialog(
                dialog_id,
                normalized_response
            )
            
            # Handle result
            original_input = dialog_context.get("original_input", "")
            response = self._handle_orchestration_result(
                result,
                original_input,
                dialog_context.get("context", {}),
                kwargs
            )
            
            # Clean up if dialog completed
            if result.outcome.value != "START_DIALOG":
                self._cleanup_dialog_context(dialog_id)
            
            return response
            
        except Exception as e:
            return self._handle_error(e, user_response, kwargs)
    
    # Learning and history
    def _learn_from_result(self,
                          prefix: str,
                          result: OrchestrationResult,
                          context: Dict[str, Any]):
        """Learn from successful advination result."""
        if not self.config["enable_learning"]:
            return
        
        learning_entry = {
            "prefix": prefix,
            "result": result.to_dict(),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "adapter_id": self.adapter_id,
        }
        
        self._learning_cache.append(learning_entry)
        
        # Flush cache if it gets too large
        if len(self._learning_cache) >= 100:
            self._flush_learning_cache()
    
    def _flush_learning_cache(self):
        """Flush learning cache to orchestrator."""
        if not hasattr(self, '_learning_cache') or not self._learning_cache:
            return
        
        for entry in self._learning_cache:
            try:
                # Extract the selected command if available
                if (entry["result"].get("suggestions") and 
                    len(entry["result"]["suggestions"]) > 0):
                    
                    selected_suggestion = entry["result"]["suggestions"][0]
                    command_text = selected_suggestion.get("text", "")
                    
                    if command_text:
                        # Learn the command
                        self.orchestrator.adivinator.learn(
                            command_text,
                            entry["context"]
                        )
            except Exception as e:
                print(f"Warning: Failed to learn from entry: {e}")
        
        # Clear cache
        self._learning_cache.clear()
    
    def _store_in_history(self,
                         prefix: str,
                         result: OrchestrationResult,
                         context: Dict[str, Any],
                         metadata: Dict[str, Any]):
        """Store request/response in history."""
        if not self.config["enable_history"]:
            return
        
        history_entry = {
            "id": str(uuid.uuid4()),
            "prefix": prefix,
            "result": result.to_dict(),
            "context": {k: v for k, v in context.items() 
                       if k not in ["adapter_config"]},  # Exclude large config
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "adapter_id": self.adapter_id,
            "platform": self.get_platform_name(),
        }
        
        self._history.append(history_entry)
        
        # Trim history if too large
        if len(self._history) > self.config["max_history_size"]:
            self._history = self._history[-self.config["max_history_size"]:]
        
        # Store in session history if session ID available
        session_id = context.get("session_id")
        if session_id:
            if session_id not in self._session_history:
                self._session_history[session_id] = []
            
            self._session_history[session_id].append(history_entry)
    
    # Session management
    def _track_session(self,
                      session_id: str,
                      prefix: str,
                      context: Dict[str, Any]):
        """Track session activity."""
        if session_id not in self._state["active_sessions"]:
            self._state["active_sessions"][session_id] = {
                "started_at": datetime.now().isoformat(),
                "requests": [],
                "context": context,
            }
            
            # Call session start callbacks
            self._call_callbacks("on_session_start", session_id, context)
        
        # Add request to session
        self._state["active_sessions"][session_id]["requests"].append({
            "prefix": prefix,
            "timestamp": datetime.now().isoformat(),
            "context": {k: v for k, v in context.items() 
                       if k not in ["adapter_config"]},
        })
    
    def end_session(self, session_id: str, **kwargs):
        """
        End a session.
        
        Args:
            session_id: Session identifier
            **kwargs: Additional parameters
        """
        if session_id in self._state["active_sessions"]:
            session_data = self._state["active_sessions"][session_id]
            session_data["ended_at"] = datetime.now().isoformat()
            session_data["end_reason"] = kwargs.get("reason", "manual")
            
            # Call session end callbacks
            self._call_callbacks("on_session_end", session_id, session_data)
            
            # Remove from active sessions
            del self._state["active_sessions"][session_id]
    
    # Dialog context management
    def _store_dialog_context(self,
                             dialog_id: str,
                             raw_input: Any,
                             context: Dict[str, Any],
                             session_id: Optional[str] = None):
        """Store dialog context for continuation."""
        # In a real implementation, this would use persistent storage
        # For now, store in memory
        if not hasattr(self, '_dialog_contexts'):
            self._dialog_contexts = {}
        
        self._dialog_contexts[dialog_id] = {
            "dialog_id": dialog_id,
            "original_input": raw_input,
            "context": context,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        }
    
    def _get_dialog_context(self, dialog_id: str) -> Optional[Dict[str, Any]]:
        """Get dialog context."""
        if not hasattr(self, '_dialog_contexts'):
            return None
        
        context = self._dialog_contexts.get(dialog_id)
        if context:
            context["last_accessed"] = datetime.now().isoformat()
        
        return context
    
    def _cleanup_dialog_context(self, dialog_id: str):
        """Clean up dialog context."""
        if hasattr(self, '_dialog_contexts') and dialog_id in self._dialog_contexts:
            del self._dialog_contexts[dialog_id]
    
    # Deferred task management
    def _store_deferred_task(self,
                            task_id: str,
                            result: OrchestrationResult,
                            context: Dict[str, Any],
                            session_id: Optional[str] = None):
        """Store deferred task information."""
        # In a real implementation, this would use persistent storage
        if not hasattr(self, '_deferred_tasks'):
            self._deferred_tasks = {}
        
        self._deferred_tasks[task_id] = {
            "task_id": task_id,
            "result": result.to_dict(),
            "context": context,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }
    
    # Callback system
    def register_callback(self,
                         event_type: str,
                         callback: Callable[[Any, ...], None]):
        """
        Register a callback for adapter events.
        
        Args:
            event_type: Event type (on_request, on_response, on_error, 
                        on_session_start, on_session_end)
            callback: Callback function
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}. "
                           f"Available: {list(self._callbacks.keys())}")
    
    def _call_callbacks(self, event_type: str, *args, **kwargs):
        """Call registered callbacks for an event."""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    print(f"Warning: Callback failed for event {event_type}: {e}")
    
    # Platform information
    @abstractmethod
    def get_platform_name(self) -> str:
        """
        Get platform name.
        
        Returns:
            Platform name string
        """
        pass
    
    @abstractmethod
    def get_platform_version(self) -> str:
        """
        Get platform version.
        
        Returns:
            Platform version string
        """
        pass
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get adapter information.
        
        Returns:
            Adapter information dictionary
        """
        return {
            "adapter_id": self.adapter_id,
            "platform": self.get_platform_name(),
            "platform_version": self.get_platform_version(),
            "orchestrator_available": self.orchestrator is not None,
            "config": {k: v for k, v in self.config.items() 
                      if k not in ["platform_specific"]},  # Exclude sensitive
            "state": self._state.copy(),
            "initialized": self._state["initialized"],
            "started_at": self._state["started_at"],
        }
    
    # Utility methods
    def _normalize_dialog_response(self,
                                  user_response: Any,
                                  dialog_context: Dict[str, Any]) -> str:
        """
        Normalize dialog response.
        
        Args:
            user_response: Raw user response
            dialog_context: Dialog context
            
        Returns:
            Normalized response string
        """
        # Default implementation - convert to string
        if isinstance(user_response, str):
            return user_response.strip()
        else:
            return str(user_response)
    
    def get_history(self,
                   limit: int = 50,
                   session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get processing history.
        
        Args:
            limit: Maximum number of entries
            session_id: Optional session filter
            
        Returns:
            List of history entries
        """
        if not self.config["enable_history"]:
            return []
        
        if session_id:
            entries = self._session_history.get(session_id, [])
        else:
            entries = self._history
        
        return entries[-limit:] if entries else []
    
    def clear_history(self, session_id: Optional[str] = None):
        """
        Clear processing history.
        
        Args:
            session_id: Optional session filter
        """
        if session_id:
            if session_id in self._session_history:
                self._session_history[session_id].clear()
        else:
            self._history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get adapter statistics.
        
        Returns:
            Statistics dictionary
        """
        history_size = len(self._history) if hasattr(self, '_history') else 0
        session_count = len(self._session_history) if hasattr(self, '_session_history') else 0
        active_sessions = len(self._state["active_sessions"])
        
        return {
            **self._state,
            "history_size": history_size,
            "session_count": session_count,
            "active_sessions": active_sessions,
            "dialog_contexts": len(getattr(self, '_dialog_contexts', {})),
            "deferred_tasks": len(getattr(self, '_deferred_tasks', {})),
            "callbacks_registered": {
                event: len(callbacks) 
                for event, callbacks in self._callbacks.items()
            },
        }
    
    # Configuration management
    def update_config(self, new_config: Dict[str, Any], merge: bool = True):
        """
        Update adapter configuration.
        
        Args:
            new_config: New configuration values
            merge: Whether to merge with existing config
        """
        if merge:
            self._deep_update(self.config, new_config)
        else:
            self.config = self._merge_default_config(new_config)
    
    def get_config(self, key: Optional[str] = None) -> Any:
        """
        Get configuration value(s).
        
        Args:
            key: Optional specific key
            
        Returns:
            Configuration value or dictionary
        """
        if key:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None
            return value
        else:
            return self.config.copy()
    
    # Lifecycle management
    def start(self):
        """Start adapter (if needed)."""
        self._state["started_at"] = datetime.now().isoformat()
        print(f"Adapter {self.adapter_id} started for {self.get_platform_name()}")
    
    def stop(self):
        """Stop adapter and clean up resources."""
        # Flush learning cache
        if hasattr(self, '_learning_cache'):
            self._flush_learning_cache()
        
        # End all active sessions
        for session_id in list(self._state["active_sessions"].keys()):
            self.end_session(session_id, reason="adapter_stopped")
        
        # Clear caches
        if hasattr(self, '_dialog_contexts'):
            self._dialog_contexts.clear()
        
        if hasattr(self, '_deferred_tasks'):
            self._deferred_tasks.clear()
        
        print(f"Adapter {self.adapter_id} stopped")
    
    def reset(self):
        """Reset adapter state."""
        self.stop()
        self._initialize()
        self.start()


# Error classes
class AdapterError(Exception):
    """Base exception for adapter errors."""
    pass


class ConfigurationError(AdapterError):
    """Exception for configuration errors."""
    pass


class ProcessingError(AdapterError):
    """Exception for processing errors."""
    pass


class PlatformError(AdapterError):
    """Exception for platform-specific errors."""
    pass


# Utility functions
def create_adapter(adapter_class: type,
                  orchestrator: ProductionOrchestrator,
                  config: Optional[Dict[str, Any]] = None) -> BaseAdapter:
    """
    Create and initialize an adapter.
    
    Args:
        adapter_class: Adapter class to instantiate
        orchestrator: Orchestrator instance
        config: Adapter configuration
        
    Returns:
        Initialized adapter instance
    """
    if not issubclass(adapter_class, BaseAdapter):
        raise ValueError(f"Class {adapter_class.__name__} must inherit from BaseAdapter")
    
    adapter = adapter_class(orchestrator, config)
    adapter.start()
    return adapter


def validate_adapter_config(config: Dict[str, Any],
                           adapter_type: str = None) -> Dict[str, Any]:
    """
    Validate adapter configuration.
    
    Args:
        config: Configuration to validate
        adapter_type: Optional adapter type for specific validation
        
    Returns:
        Validation results
    """
    errors = []
    warnings = []
    
    # Common validations
    if "enable_learning" in config and not isinstance(config["enable_learning"], bool):
        errors.append("enable_learning must be a boolean")
    
    if "enable_history" in config and not isinstance(config["enable_history"], bool):
        errors.append("enable_history must be a boolean")
    
    if "max_history_size" in config:
        max_size = config["max_history_size"]
        if not isinstance(max_size, int) or max_size < 0:
            errors.append("max_history_size must be a non-negative integer")
        elif max_size > 100000:
            warnings.append(f"Large max_history_size: {max_size}")
    
    if "response_timeout_seconds" in config:
        timeout = config["response_timeout_seconds"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append("response_timeout_seconds must be a positive number")
        elif timeout > 300:
            warnings.append(f"Long response timeout: {timeout} seconds")
    
    if "auto_confirm_threshold" in config:
        threshold = config["auto_confirm_threshold"]
        if not isinstance(threshold, (int, float)):
            errors.append("auto_confirm_threshold must be a number")
        elif threshold < 0 or threshold > 1:
            errors.append("auto_confirm_threshold must be between 0 and 1")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "config": config,
    }


# Module-level adapter registry
class AdapterRegistry:
    """Registry for adapter types."""
    
    _adapters: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, adapter_class: type):
        """
        Register an adapter class.
        
        Args:
            name: Adapter name
            adapter_class: Adapter class
        """
        if not issubclass(adapter_class, BaseAdapter):
            raise ValueError(f"Class {adapter_class.__name__} must inherit from BaseAdapter")
        
        cls._adapters[name] = adapter_class
    
    @classmethod
    def create(cls,
               adapter_type: str,
               orchestrator: ProductionOrchestrator,
               config: Optional[Dict[str, Any]] = None) -> BaseAdapter:
        """
        Create an adapter instance.
        
        Args:
            adapter_type: Type of adapter to create
            orchestrator: Orchestrator instance
            config: Adapter configuration
            
        Returns:
            Adapter instance
        """
        if adapter_type not in cls._adapters:
            raise ValueError(f"Unknown adapter type: {adapter_type}. "
                           f"Available: {list(cls._adapters.keys())}")
        
        adapter_class = cls._adapters[adapter_type]
        return create_adapter(adapter_class, orchestrator, config)
    
    @classmethod
    def list_adapters(cls) -> Dict[str, str]:
        """List all registered adapters."""
        return {
            name: adapter_class.__name__
            for name, adapter_class in cls._adapters.items()
        }