"""
Storage module for semantic advination system.
Persistent and in-memory storage solutions for command history and models.
"""

__version__ = "1.0.0"
__author__ = "Semantic Advination Core Team"
__license__ = "MIT"

from .trie_storage import (
    CommandTrie,
    PersistentCommandTrie,
    create_trie_from_commands,
    merge_tries,
)

# Re-export for convenience
__all__ = [
    'CommandTrie',
    'PersistentCommandTrie',
    'create_trie_from_commands',
    'merge_tries',
]

# Storage configuration types
from typing import Dict, Any, Optional, Union
from pathlib import Path

StorageConfig = Dict[str, Any]
"""Type alias for storage configuration."""

StorageBackend = Union[CommandTrie, 'SQLiteStorage', 'MemoryStorage']
"""Type alias for storage backend types."""

# Storage factory and registry
class StorageRegistry:
    """
    Registry for storage backends.
    Allows dynamic registration and creation of storage implementations.
    """
    
    _backends: Dict[str, type] = {}
    _default_backend = "trie"
    
    @classmethod
    def register(cls, name: str, backend_class: type):
        """
        Register a storage backend.
        
        Args:
            name: Backend name (e.g., 'trie', 'sqlite', 'memory')
            backend_class: Storage backend class
        """
        cls._backends[name] = backend_class
    
    @classmethod
    def create(cls,
               backend_type: str = None,
               config: Optional[StorageConfig] = None,
               **kwargs) -> StorageBackend:
        """
        Create a storage backend instance.
        
        Args:
            backend_type: Type of backend to create
            config: Configuration dictionary
            **kwargs: Additional arguments for backend constructor
            
        Returns:
            Storage backend instance
        """
        backend_type = backend_type or cls._default_backend
        
        if backend_type not in cls._backends:
            raise ValueError(f"Unknown storage backend: {backend_type}. "
                           f"Available: {list(cls._backends.keys())}")
        
        backend_class = cls._backends[backend_type]
        merged_config = config or {}
        merged_config.update(kwargs)
        
        return backend_class(**merged_config)
    
    @classmethod
    def list_backends(cls) -> Dict[str, str]:
        """List all registered storage backends."""
        return {
            name: backend_class.__name__
            for name, backend_class in cls._backends.items()
        }


# Register default backends
StorageRegistry.register("trie", CommandTrie)
StorageRegistry.register("persistent_trie", PersistentCommandTrie)

# Storage configuration utilities
def create_default_config(backend_type: str = "trie") -> StorageConfig:
    """
    Create default configuration for a storage backend.
    
    Args:
        backend_type: Type of storage backend
        
    Returns:
        Default configuration dictionary
    """
    base_config = {
        "auto_save": True,
        "max_commands": 10000,
        "data_dir": "data",
        "backup_on_save": True,
    }
    
    if backend_type == "trie":
        config = {
            **base_config,
            "enable_fuzzy_search": True,
            "fuzzy_threshold": 0.6,
            "compression": False,
        }
    elif backend_type == "persistent_trie":
        config = {
            **base_config,
            "enable_fuzzy_search": True,
            "auto_save_changes": 100,
            "incremental_save": True,
        }
    else:
        config = base_config
    
    return config


def validate_config(config: StorageConfig, backend_type: str = None) -> Dict[str, Any]:
    """
    Validate storage configuration.
    
    Args:
        config: Configuration to validate
        backend_type: Optional backend type for type-specific validation
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Common validations
    if "max_commands" in config:
        max_cmds = config["max_commands"]
        if not isinstance(max_cmds, int) or max_cmds <= 0:
            errors.append("max_commands must be a positive integer")
        elif max_cmds > 1000000:
            warnings.append(f"Large max_commands value: {max_cmds}")
    
    if "data_dir" in config:
        data_dir = config["data_dir"]
        if not isinstance(data_dir, (str, Path)):
            errors.append("data_dir must be a string or Path")
    
    if "auto_save" in config and not isinstance(config["auto_save"], bool):
        errors.append("auto_save must be a boolean")
    
    # Backend-specific validations
    if backend_type == "trie":
        if "fuzzy_threshold" in config:
            threshold = config["fuzzy_threshold"]
            if not isinstance(threshold, (int, float)):
                errors.append("fuzzy_threshold must be a number")
            elif threshold < 0 or threshold > 1:
                errors.append("fuzzy_threshold must be between 0 and 1")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "config": config,
    }


# Storage statistics and monitoring
class StorageMonitor:
    """
    Monitor for storage operations and performance.
    """
    
    def __init__(self):
        self.operations = []
        self.start_time = None
        self.total_operations = 0
    
    def start_monitoring(self):
        """Start monitoring storage operations."""
        import time
        self.start_time = time.time()
        self.operations = []
        self.total_operations = 0
    
    def record_operation(self,
                        operation_type: str,
                        duration: float,
                        success: bool = True,
                        details: Optional[Dict[str, Any]] = None):
        """
        Record a storage operation.
        
        Args:
            operation_type: Type of operation (insert, search, delete, etc.)
            duration: Operation duration in seconds
            success: Whether operation succeeded
            details: Additional operation details
        """
        self.total_operations += 1
        self.operations.append({
            "type": operation_type,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        import time
        
        if not self.operations:
            return {"total_operations": 0}
        
        durations = [op["duration"] for op in self.operations]
        success_count = sum(1 for op in self.operations if op["success"])
        
        operation_types = {}
        for op in self.operations:
            op_type = op["type"]
            operation_types[op_type] = operation_types.get(op_type, 0) + 1
        
        total_duration = sum(durations)
        
        return {
            "total_operations": self.total_operations,
            "success_rate": success_count / len(self.operations),
            "avg_duration": total_duration / len(self.operations),
            "total_duration": total_duration,
            "operation_types": operation_types,
            "monitoring_since": self.start_time,
            "monitoring_duration": time.time() - self.start_time if self.start_time else 0,
        }
    
    def clear(self):
        """Clear monitoring data."""
        self.operations.clear()
        self.total_operations = 0


# Global storage monitor
_storage_monitor = StorageMonitor()


def get_storage_monitor() -> StorageMonitor:
    """
    Get the global storage monitor.
    
    Returns:
        StorageMonitor instance
    """
    return _storage_monitor


# Storage migration utilities
class StorageMigrator:
    """
    Utility for migrating between different storage backends.
    """
    
    @staticmethod
    def migrate(source: StorageBackend,
                target: StorageBackend,
                progress_callback=None) -> Dict[str, Any]:
        """
        Migrate data from source to target storage.
        
        Args:
            source: Source storage backend
            target: Target storage backend
            progress_callback: Optional callback for progress updates
            
        Returns:
            Migration statistics
        """
        import time
        
        start_time = time.time()
        migrated_count = 0
        error_count = 0
        skipped_count = 0
        
        # Get commands from source
        if hasattr(source, 'commands'):
            commands = list(source.commands.values())
        else:
            # Fallback for other storage types
            commands = StorageMigrator._extract_commands_from_source(source)
        
        total_commands = len(commands)
        
        # Migrate each command
        for i, cmd_data in enumerate(commands):
            try:
                # Skip if command already exists in target
                cmd_text = cmd_data.get("command", "")
                if hasattr(target, 'get_command_by_text'):
                    existing = target.get_command_by_text(cmd_text)
                    if existing:
                        skipped_count += 1
                        continue
                
                # Insert into target
                target.insert(cmd_data.copy())
                migrated_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress = (i + 1) / total_commands
                    progress_callback(progress, migrated_count, total_commands)
                    
            except Exception as e:
                error_count += 1
                print(f"Warning: Failed to migrate command '{cmd_text}': {e}")
        
        end_time = time.time()
        
        return {
            "total_commands": total_commands,
            "migrated": migrated_count,
            "errors": error_count,
            "skipped": skipped_count,
            "duration_seconds": end_time - start_time,
            "success_rate": migrated_count / total_commands if total_commands > 0 else 0,
        }
    
    @staticmethod
    def _extract_commands_from_source(source) -> List[Dict[str, Any]]:
        """
        Extract commands from source storage.
        
        Args:
            source: Source storage backend
            
        Returns:
            List of command dictionaries
        """
        # This is a fallback method - should be overridden for specific backends
        if hasattr(source, 'export_to_json'):
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                source.export_to_json(f.name)
                with open(f.name, 'r') as json_file:
                    data = json.load(json_file)
                    return data.get("commands", [])
        
        # Default: try to access commands attribute
        if hasattr(source, 'commands'):
            return list(source.commands.values())
        
        raise ValueError(f"Cannot extract commands from source of type {type(source)}")


# Storage backup and recovery
def create_backup(storage: StorageBackend,
                  backup_dir: str = "backups",
                  include_metadata: bool = True) -> str:
    """
    Create a backup of storage data.
    
    Args:
        storage: Storage backend to backup
        backup_dir: Directory for backups
        include_metadata: Whether to include metadata
        
    Returns:
        Path to backup file
    """
    from pathlib import Path
    import json
    from datetime import datetime
    
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_path / f"storage_backup_{timestamp}.json"
    
    # Collect data for backup
    backup_data = {
        "backup_timestamp": timestamp,
        "backup_created": datetime.now().isoformat(),
        "storage_type": type(storage).__name__,
    }
    
    if include_metadata:
        if hasattr(storage, 'metadata'):
            backup_data["metadata"] = storage.metadata
        if hasattr(storage, 'stats'):
            backup_data["stats"] = storage.stats
    
    # Add commands
    if hasattr(storage, 'commands'):
        commands = []
        for cmd_id, cmd_data in storage.commands.items():
            # Create a serializable copy
            cmd_copy = cmd_data.copy()
            # Ensure all values are JSON serializable
            for key, value in cmd_copy.items():
                if isinstance(value, datetime):
                    cmd_copy[key] = value.isoformat()
            commands.append(cmd_copy)
        
        backup_data["commands"] = commands
        backup_data["command_count"] = len(commands)
    
    # Write backup file
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    print(f"Backup created: {backup_file} ({backup_data.get('command_count', 0)} commands)")
    return str(backup_file)


def restore_from_backup(backup_file: str,
                        target_storage: StorageBackend = None,
                        merge: bool = False) -> StorageBackend:
    """
    Restore storage from backup file.
    
    Args:
        backup_file: Path to backup file
        target_storage: Target storage backend (creates new if None)
        merge: Whether to merge with existing data
        
    Returns:
        Restored storage backend
    """
    import json
    from pathlib import Path
    
    backup_path = Path(backup_file)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file}")
    
    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_data = json.load(f)
    
    # Create or use target storage
    if target_storage is None:
        storage_type = backup_data.get("storage_type", "CommandTrie")
        if storage_type == "CommandTrie":
            target_storage = CommandTrie()
        else:
            raise ValueError(f"Unsupported storage type in backup: {storage_type}")
    elif not merge:
        target_storage.clear()
    
    # Restore commands
    commands = backup_data.get("commands", [])
    restored_count = 0
    error_count = 0
    
    for cmd_data in commands:
        try:
            target_storage.insert(cmd_data)
            restored_count += 1
        except Exception as e:
            error_count += 1
            print(f"Warning: Failed to restore command: {e}")
    
    # Restore metadata if available
    if hasattr(target_storage, 'metadata'):
        metadata = backup_data.get("metadata", {})
        target_storage.metadata.update(metadata)
    
    print(f"Restored {restored_count} commands from backup ({error_count} errors)")
    return target_storage


# Storage benchmarking
def benchmark_storage(storage: StorageBackend,
                     operations: int = 1000,
                     warmup: int = 100) -> Dict[str, Any]:
    """
    Benchmark storage performance.
    
    Args:
        storage: Storage backend to benchmark
        operations: Number of operations to perform
        warmup: Number of warmup operations
        
    Returns:
        Benchmark results
    """
    import time
    import random
    import string
    
    results = {
        "insert": {"durations": [], "success": 0, "errors": 0},
        "search": {"durations": [], "success": 0, "errors": 0},
        "delete": {"durations": [], "success": 0, "errors": 0},
    }
    
    # Generate test data
    def generate_random_command(length: int = 10) -> str:
        words = ['git', 'ls', 'cd', 'docker', 'python', 'find', 'grep', 'cat']
        return f"{random.choice(words)} {''.join(random.choices(string.ascii_lowercase, k=length))}"
    
    # Warmup phase
    print(f"Warming up with {warmup} operations...")
    for _ in range(warmup):
        cmd = generate_random_command(5)
        try:
            storage.insert({"command": cmd})
            storage.search_exact(cmd[:3], limit=1)
        except:
            pass
    
    # Benchmark phase
    print(f"Benchmarking with {operations} operations...")
    
    inserted_commands = []
    
    for i in range(operations):
        op_type = random.choice(["insert", "search", "delete"])
        cmd = generate_random_command(random.randint(5, 20))
        
        start_time = time.time()
        
        try:
            if op_type == "insert":
                storage.insert({"command": cmd, "usage_count": 1})
                inserted_commands.append(cmd)
                results["insert"]["success"] += 1
                
            elif op_type == "search":
                # Search for existing or random command
                if inserted_commands and random.random() > 0.3:
                    search_cmd = random.choice(inserted_commands)
                else:
                    search_cmd = cmd
                
                prefix = search_cmd[:random.randint(1, min(5, len(search_cmd)))]
                storage.search_exact(prefix, limit=5)
                results["search"]["success"] += 1
                
            elif op_type == "delete":
                # Delete existing command if available
                if inserted_commands:
                    delete_cmd = random.choice(inserted_commands)
                    if hasattr(storage, 'get_command_by_text'):
                        cmd_data = storage.get_command_by_text(delete_cmd)
                        if cmd_data:
                            storage.delete_command(cmd_data.get("id"))
                            inserted_commands.remove(delete_cmd)
                    results["delete"]["success"] += 1
                else:
                    results["delete"]["errors"] += 1
            
        except Exception as e:
            results[op_type]["errors"] += 1
        
        duration = time.time() - start_time
        results[op_type]["durations"].append(duration)
    
    # Calculate statistics
    for op_type in results:
        durations = results[op_type]["durations"]
        if durations:
            results[op_type]["avg_duration"] = sum(durations) / len(durations)
            results[op_type]["min_duration"] = min(durations)
            results[op_type]["max_duration"] = max(durations)
            results[op_type]["total_duration"] = sum(durations)
        else:
            results[op_type]["avg_duration"] = 0
            results[op_type]["min_duration"] = 0
            results[op_type]["max_duration"] = 0
            results[op_type]["total_duration"] = 0
    
    # Get storage stats if available
    if hasattr(storage, 'get_stats'):
        results["storage_stats"] = storage.get_stats()
    
    return results


# Storage utilities
def estimate_storage_size(storage: StorageBackend) -> Dict[str, Any]:
    """
    Estimate storage size in memory and on disk.
    
    Args:
        storage: Storage backend
        
    Returns:
        Size estimates in bytes and MB
    """
    import sys
    from pathlib import Path
    
    estimates = {
        "memory_bytes": 0,
        "memory_mb": 0,
        "disk_bytes": 0,
        "disk_mb": 0,
    }
    
    # Estimate memory usage
    try:
        if hasattr(storage, 'commands'):
            total_size = sys.getsizeof(storage)
            
            # Add size of commands dictionary
            total_size += sum(sys.getsizeof(k) + sys.getsizeof(v)
                            for k, v in storage.commands.items())
            
            estimates["memory_bytes"] = total_size
            estimates["memory_mb"] = total_size / (1024 * 1024)
    except:
        pass
    
    # Estimate disk usage
    try:
        if hasattr(storage, 'data_dir'):
            data_dir = Path(storage.data_dir)
            if data_dir.exists():
                total_disk_size = 0
                for file in data_dir.rglob("*"):
                    if file.is_file():
                        total_disk_size += file.stat().st_size
                
                estimates["disk_bytes"] = total_disk_size
                estimates["disk_mb"] = total_disk_size / (1024 * 1024)
    except:
        pass
    
    return estimates


def compact_storage(storage: StorageBackend) -> Dict[str, Any]:
    """
    Compact storage to save space.
    
    Args:
        storage: Storage backend to compact
        
    Returns:
        Compaction results
    """
    results = {
        "before_size": estimate_storage_size(storage),
        "removed_count": 0,
        "errors": 0,
    }
    
    # Remove duplicate commands if supported
    if hasattr(storage, 'commands') and hasattr(storage, 'reverse_index'):
        duplicates = []
        seen_texts = set()
        
        for cmd_id, cmd_data in list(storage.commands.items()):
            cmd_text = cmd_data.get("command", "")
            if cmd_text in seen_texts:
                duplicates.append(cmd_id)
            else:
                seen_texts.add(cmd_text)
        
        # Remove duplicates
        for cmd_id in duplicates:
            try:
                storage.delete_command(cmd_id)
                results["removed_count"] += 1
            except Exception as e:
                results["errors"] += 1
                print(f"Warning: Failed to remove duplicate command {cmd_id}: {e}")
    
    # Optimize storage if supported
    if hasattr(storage, 'optimize'):
        try:
            storage.optimize()
            results["optimized"] = True
        except Exception as e:
            results["optimized"] = False
            results["optimize_error"] = str(e)
    
    results["after_size"] = estimate_storage_size(storage)
    
    return results


# Module initialization
from datetime import datetime

def initialize_storage_module(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the storage module.
    
    Args:
        config: Optional configuration
        
    Returns:
        Initialization status
    """
    init_config = config or {}
    
    # Start monitoring
    _storage_monitor.start_monitoring()
    
    return {
        "status": "initialized",
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
        "registered_backends": StorageRegistry.list_backends(),
        "monitoring_started": True,
        "config": init_config,
    }


# Global module state
_storage_module_state = {
    "initialized": False,
    "default_backend": "trie",
    "backends_available": list(StorageRegistry._backends.keys()),
}


def get_storage_module_state() -> Dict[str, Any]:
    """Get current storage module state."""
    return {
        **_storage_module_state,
        "monitor_active": _storage_monitor.start_time is not None,
        "default_config": create_default_config(),
    }


# Auto-initialize
try:
    init_result = initialize_storage_module()
    _storage_module_state.update({
        "initialized": True,
        "initialization_time": datetime.now(),
        "init_result": init_result,
    })
except Exception as e:
    _storage_module_state.update({
        "initialized": False,
        "init_error": str(e),
        "init_error_type": type(e).__name__,
    })
    print(f"Warning: Storage module initialization failed: {e}")


# Export commonly used utilities
create_storage = StorageRegistry.create
"""Alias for StorageRegistry.create for convenience."""

# Example usage:
# storage = create_storage("trie", data_dir="my_data", max_commands=5000)