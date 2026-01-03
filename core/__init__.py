"""
Core module for semantic advination system.
Pure logic for command guessing, validation, composition, and orchestration.
"""

__version__ = "1.0.0"
__author__ = "Semantic Advination Core Team"
__license__ = "MIT"

from .models import (
    # Enums
    AdvinationResultType,
    OrchestrationOutcome,
    CommandStatus,
    
    # Data classes
    CommandSuggestion,
    AdvinationResult,
    OrchestrationResult,
    Command,
    DialogContext,
    ClarificationQuestion,
    LearningEvent,
    
    # Type aliases (for convenience)
    CommandDict,
    ContextDict,
    SuggestionList,
    ValidationResult,
)

from .adivinator import (
    Adivinator,
    ConfigurableAdivinator,
)

from .validator import (
    CommandValidator,
)

from .composer import (
    CommandComposer,
)

from .orchestrator import (
    ProductionOrchestrator,
    DeferralStrategy,
    OrchestrationMetrics,
)

# Re-export commonly used types for convenience
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Models
    'AdvinationResultType',
    'OrchestrationOutcome',
    'CommandStatus',
    'CommandSuggestion',
    'AdvinationResult',
    'OrchestrationResult',
    'Command',
    'DialogContext',
    'ClarificationQuestion',
    'LearningEvent',
    
    # Core components
    'Adivinator',
    'ConfigurableAdivinator',
    'CommandValidator',
    'CommandComposer',
    'ProductionOrchestrator',
    
    # Utilities
    'DeferralStrategy',
    'OrchestrationMetrics',
    
    # Type aliases (defined below)
    'CommandDict',
    'ContextDict',
    'SuggestionList',
    'ValidationResult',
]

# Type aliases for better code readability
from typing import Dict, List, Any, Optional, Union

CommandDict = Dict[str, Any]
"""Type alias for command dictionary representation."""

ContextDict = Dict[str, Any]
"""Type alias for context dictionary."""

SuggestionList = List[CommandSuggestion]
"""Type alias for list of command suggestions."""

ValidationResult = Dict[str, Any]
"""Type alias for command validation result."""

# Utility functions for common operations
def create_advination_result(
    result_type: AdvinationResultType,
    suggestions: Optional[List[CommandSuggestion]] = None,
    confidence: float = 1.0,
    prefix: str = "",
    context: Optional[Dict[str, Any]] = None
) -> AdvinationResult:
    """
    Helper function to create an AdvinationResult.
    
    Args:
        result_type: Type of the result
        suggestions: List of command suggestions
        confidence: Confidence score (0.0-1.0)
        prefix: Original prefix
        context: Context dictionary
        
    Returns:
        AdvinationResult object
    """
    return AdvinationResult(
        result_type=result_type,
        suggestions=suggestions,
        confidence=confidence,
        raw_prefix=prefix,
        raw_context=context or {}
    )


def create_command_suggestion(
    text: str,
    source: str = "unknown",
    match_score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> CommandSuggestion:
    """
    Helper function to create a CommandSuggestion.
    
    Args:
        text: Command text
        source: Source of the suggestion
        match_score: Match score (0.0-1.0)
        metadata: Additional metadata
        
    Returns:
        CommandSuggestion object
    """
    return CommandSuggestion(
        text=text,
        source=source,
        match_score=match_score,
        metadata=metadata or {}
    )


def create_orchestration_result(
    outcome: OrchestrationOutcome,
    suggestions: Optional[List[CommandSuggestion]] = None,
    dialog_id: Optional[str] = None,
    first_question: Optional[str] = None,
    task_id: Optional[str] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> OrchestrationResult:
    """
    Helper function to create an OrchestrationResult.
    
    Args:
        outcome: Outcome of orchestration
        suggestions: List of command suggestions
        dialog_id: ID of the dialog (if started)
        first_question: First question of the dialog
        task_id: ID of the deferred task
        reason: Reason for deferral
        metadata: Additional metadata
        
    Returns:
        OrchestrationResult object
    """
    return OrchestrationResult(
        outcome=outcome,
        suggestions=suggestions,
        dialog_id=dialog_id,
        first_question=first_question,
        task_id=task_id,
        reason=reason,
        metadata=metadata or {}
    )


def create_command(
    text: str,
    description: str = "",
    category: str = "general",
    tags: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    usage_count: int = 0,
    status: CommandStatus = CommandStatus.CONFIRMED
) -> Command:
    """
    Helper function to create a Command.
    
    Args:
        text: Command text
        description: Command description
        category: Command category
        tags: List of tags
        parameters: Command parameters
        usage_count: Usage count
        status: Command status
        
    Returns:
        Command object
    """
    return Command(
        text=text,
        description=description,
        category=category,
        tags=tags or [],
        parameters=parameters or {},
        usage_count=usage_count,
        status=status
    )


# Factory function for creating pre-configured systems
def create_advination_system(
    storage_backend: Any = None,
    config: Optional[Dict[str, Any]] = None
) -> ProductionOrchestrator:
    """
    Creates a fully configured advination system.
    
    Args:
        storage_backend: Storage backend (e.g., CommandTrie)
        config: Configuration dictionary
        
    Returns:
        ProductionOrchestrator instance
    """
    # Import here to avoid circular imports
    from storage.trie_storage import CommandTrie
    
    # Use provided storage or create default
    storage = storage_backend or CommandTrie()
    
    # Merge with default config
    system_config = {
        "enable_adaptation": True,
        "enable_composition": True,
        "enable_deferral": True,
        "max_dialog_questions": 5,
        "partial_match_threshold": 0.3,
        **(config or {})
    }
    
    # Create components
    adivinator_config = {
        "min_prefix_length": 2,
        "partial_threshold": system_config.get("partial_match_threshold", 0.3),
        "max_exact_results": 5,
        "max_partial_results": 3
    }
    
    adivinator = Adivinator(storage, adivinator_config)
    validator = CommandValidator()
    composer = CommandComposer({
        "max_questions": system_config.get("max_dialog_questions", 5),
        "enable_smart_defaults": True
    })
    
    # Create orchestrator
    orchestrator = ProductionOrchestrator(
        advinator=adivinator,
        validator=validator,
        composer=composer,
        config=system_config
    )
    
    return orchestrator


def quick_advinate(
    prefix: str,
    context: Optional[Dict[str, Any]] = None,
    system: Optional[ProductionOrchestrator] = None
) -> OrchestrationResult:
    """
    Quick utility function for advination.
    
    Args:
        prefix: Prefix to advinate
        context: Context dictionary
        system: Optional pre-configured system
        
    Returns:
        OrchestrationResult
    """
    if system is None:
        system = create_advination_system()
    
    return system.process_request(prefix, context or {})


# Diagnostic utilities
def print_advination_result(result: AdvinationResult, indent: int = 0):
    """
    Pretty prints an AdvinationResult.
    
    Args:
        result: Result to print
        indent: Indentation level
    """
    indent_str = " " * indent
    print(f"{indent_str}Advination Result:")
    print(f"{indent_str}  Type: {result.result_type.value}")
    print(f"{indent_str}  Confidence: {result.confidence:.2f}")
    print(f"{indent_str}  Prefix: '{result.raw_prefix}'")
    
    if result.suggestions:
        print(f"{indent_str}  Suggestions ({len(result.suggestions)}):")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"{indent_str}    {i}. '{suggestion.text}' "
                  f"(score: {suggestion.match_score:.2f}, source: {suggestion.source})")


def print_orchestration_result(result: OrchestrationResult, indent: int = 0):
    """
    Pretty prints an OrchestrationResult.
    
    Args:
        result: Result to print
        indent: Indentation level
    """
    indent_str = " " * indent
    print(f"{indent_str}Orchestration Result:")
    print(f"{indent_str}  Outcome: {result.outcome.value}")
    
    if result.suggestions:
        print(f"{indent_str}  Suggestions ({len(result.suggestions)}):")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"{indent_str}    {i}. '{suggestion.text}'")
    
    if result.dialog_id:
        print(f"{indent_str}  Dialog ID: {result.dialog_id}")
        if result.first_question:
            print(f"{indent_str}  First question: {result.first_question}")
    
    if result.task_id:
        print(f"{indent_str}  Task ID: {result.task_id}")
        if result.reason:
            print(f"{indent_str}  Reason: {result.reason}")


# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitor for the core module."""
    
    def __init__(self):
        self.operations = []
        self.start_time = None
    
    def start_operation(self, name: str):
        """Starts timing an operation."""
        import time
        self.operations.append({
            "name": name,
            "start_time": time.time(),
            "end_time": None,
            "duration": None
        })
    
    def end_operation(self, name: str):
        """Ends timing an operation."""
        import time
        for op in self.operations:
            if op["name"] == name and op["end_time"] is None:
                op["end_time"] = time.time()
                op["duration"] = op["end_time"] - op["start_time"]
                break
    
    def get_report(self) -> Dict[str, Any]:
        """Returns a performance report."""
        import time
        total_duration = 0
        completed_ops = []
        
        for op in self.operations:
            if op["duration"] is not None:
                total_duration += op["duration"]
                completed_ops.append(op)
        
        return {
            "total_operations": len(self.operations),
            "completed_operations": len(completed_ops),
            "total_duration": total_duration,
            "operations": completed_ops,
            "average_duration": total_duration / len(completed_ops) if completed_ops else 0
        }


# Module-level performance monitor
_performance_monitor = PerformanceMonitor()


def monitor_performance(func):
    """Decorator for monitoring function performance."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _performance_monitor.start_operation(func.__name__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            _performance_monitor.end_operation(func.__name__)
            duration = time.time() - start_time
    
    return wrapper


def get_performance_stats() -> Dict[str, Any]:
    """Returns performance statistics for the core module."""
    return _performance_monitor.get_report()


# Error handling utilities
class AdvinationError(Exception):
    """Base exception for advination errors."""
    pass


class ValidationError(AdvinationError):
    """Exception for validation errors."""
    pass


class CompositionError(AdvinationError):
    """Exception for composition errors."""
    pass


class OrchestrationError(AdvinationError):
    """Exception for orchestration errors."""
    pass


def handle_advination_error(error: Exception, prefix: str, context: Dict[str, Any]) -> OrchestrationResult:
    """
    Handles advination errors and returns appropriate result.
    
    Args:
        error: The exception that occurred
        prefix: Original prefix
        context: Context dictionary
        
    Returns:
        OrchestrationResult with DEFER outcome
    """
    from datetime import datetime, timedelta
    
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Log the error (in a real system, this would go to a logger)
    print(f"Advination error: {error_type}: {error_msg} for prefix '{prefix}'")
    
    # Return deferred result
    return create_orchestration_result(
        outcome=OrchestrationOutcome.DEFER,
        task_id=f"error_{datetime.now().timestamp()}",
        reason=f"{error_type}: {error_msg}",
        metadata={
            "error_type": error_type,
            "error_message": error_msg,
            "prefix": prefix,
            "context": context,
            "handled_at": datetime.now().isoformat(),
            "retry_after": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    )


# Module initialization
def initialize_module(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initializes the core module.
    
    Args:
        config: Optional configuration
        
    Returns:
        Status dictionary
    """
    init_config = config or {}
    
    # Perform any initialization tasks here
    # For now, just return status
    
    return {
        "status": "initialized",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
        "config": init_config,
        "components_available": {
            "Adivinator": True,
            "CommandValidator": True,
            "CommandComposer": True,
            "ProductionOrchestrator": True
        }
    }


# Global module state
_module_state = {
    "initialized": False,
    "initialization_time": None,
    "default_system": None
}


def get_module_state() -> Dict[str, Any]:
    """Returns the current module state."""
    return _module_state.copy()


# Auto-initialize when imported
try:
    init_result = initialize_module()
    _module_state.update({
        "initialized": True,
        "initialization_time": datetime.now(),
        "init_result": init_result
    })
except Exception as e:
    _module_state.update({
        "initialized": False,
        "init_error": str(e),
        "init_error_type": type(e).__name__
    })
    print(f"Warning: Core module initialization failed: {e}")

# Export the monitor for external access
performance_monitor = _performance_monitor