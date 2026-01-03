"""
Shell adapter for semantic advination system.
Provides integration with command-line interfaces (bash, zsh, PowerShell, etc.).
"""

import os
import sys
import re
import shlex
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import subprocess
import readline  # For input history and completion

from .base_adapter import BaseAdapter, AdapterError, create_adapter
from core.models import OrchestrationResult, CommandSuggestion
from core.orchestrator import ProductionOrchestrator


class ShellAdapter(BaseAdapter):
    """
    Adapter for command-line shell environments.
    Supports bash, zsh, PowerShell, and other Unix-like shells.
    """
    
    def __init__(self, 
                 orchestrator: Optional[ProductionOrchestrator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize shell adapter.
        
        Args:
            orchestrator: Core orchestrator instance
            config: Shell-specific configuration
        """
        # Default shell config
        default_shell_config = {
            "shell_type": self._detect_shell_type(),
            "enable_tab_completion": True,
            "enable_history_integration": True,
            "max_suggestions_display": 5,
            "show_confidence": False,
            "auto_execute_threshold": 0.95,
            "require_confirmation": True,
            "confirmation_timeout": 5,
            "color_output": True,
            "prompt_format": "{platform}> ",
            "working_directory": os.getcwd(),
            "shell_specific": {
                "bash": {"history_file": "~/.bash_history"},
                "zsh": {"history_file": "~/.zsh_history"},
                "powershell": {"history_file": None},
                "cmd": {"history_file": None},
            },
        }
        
        # Merge with user config
        merged_config = default_shell_config.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(orchestrator, merged_config)
        
        # Shell-specific initialization
        self._shell_state = {
            "current_directory": os.getcwd(),
            "environment": dict(os.environ),
            "command_history": [],
            "last_command": None,
            "session_start_time": datetime.now().isoformat(),
        }
        
        # Initialize readline for better input handling if available
        self._init_readline()
        
        # Load shell history if enabled
        if self.config["enable_history_integration"]:
            self._load_shell_history()
    
    def _detect_shell_type(self) -> str:
        """Detect the current shell type."""
        shell = os.environ.get("SHELL", "")
        if "bash" in shell.lower():
            return "bash"
        elif "zsh" in shell.lower():
            return "zsh"
        elif "powershell" in shell.lower() or "ps" in shell.lower():
            return "powershell"
        elif "cmd" in shell.lower() or "command" in shell.lower():
            return "cmd"
        elif "fish" in shell.lower():
            return "fish"
        else:
            # Default to bash-like for unknown shells
            return "bash"
    
    def _initialize_platform_context(self) -> Dict[str, Any]:
        """
        Initialize shell-specific context.
        
        Returns:
            Shell context dictionary
        """
        shell_type = self.config["shell_type"]
        
        # Get user info
        user_info = {
            "username": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
            "home_directory": os.path.expanduser("~"),
        }
        
        # Get system info
        try:
            hostname = os.uname().nodename
        except AttributeError:
            hostname = os.environ.get("COMPUTERNAME", "unknown")
        
        # Get current directory and its contents
        current_dir = os.getcwd()
        try:
            dir_contents = os.listdir(current_dir)
        except Exception:
            dir_contents = []
        
        return {
            "shell_type": shell_type,
            "user": user_info,
            "hostname": hostname,
            "current_directory": current_dir,
            "directory_contents": dir_contents,
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if not k.startswith(("_", ".")) and len(k) < 50
            },
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
        }
    
    def _init_readline(self):
        """Initialize readline for enhanced input handling."""
        try:
            # Configure readline for tab completion
            readline.parse_and_bind("tab: complete")
            
            # Set completer function
            readline.set_completer(self._readline_completer)
            
            # Configure history
            readline.set_history_length(1000)
            
            # Load existing history if any
            history_file = os.path.expanduser("~/.advination_history")
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
                
        except (ImportError, AttributeError):
            # readline not available (e.g., on Windows without pyreadline)
            self.config["enable_tab_completion"] = False
    
    def _readline_completer(self, text: str, state: int) -> Optional[str]:
        """
        Readline completer function for tab completion.
        
        Args:
            text: Text to complete
            state: Completion state
            
        Returns:
            Completion suggestion or None
        """
        if not self.config["enable_tab_completion"] or not self.orchestrator:
            return None
        
        # Get completions on first call
        if state == 0:
            try:
                result = self.orchestrator.process_request(
                    prefix=text,
                    context=self.create_request_context(text)
                )
                
                if result.suggestions:
                    self._completion_cache = [
                        suggestion.text for suggestion in result.suggestions
                    ]
                else:
                    self._completion_cache = []
            except Exception:
                self._completion_cache = []
        
        # Return cached completion
        if state < len(self._completion_cache):
            return self._completion_cache[state]
        
        return None
    
    def _load_shell_history(self):
        """Load shell command history."""
        shell_type = self.config["shell_type"]
        history_config = self.config["shell_specific"].get(shell_type, {})
        history_file = history_config.get("history_file")
        
        if not history_file:
            return
        
        history_path = os.path.expanduser(history_file)
        if not os.path.exists(history_path):
            return
        
        try:
            with open(history_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                # Parse history based on shell type
                if shell_type == "bash":
                    self._parse_bash_history(lines)
                elif shell_type == "zsh":
                    self._parse_zsh_history(lines)
                elif shell_type == "powershell":
                    self._parse_powershell_history(lines)
                
        except Exception as e:
            print(f"Warning: Could not load shell history: {e}")
    
    def _parse_bash_history(self, lines: List[str]):
        """Parse bash history file."""
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Simple parsing - in reality bash history is more complex
                self._shell_state["command_history"].append({
                    "command": line,
                    "timestamp": datetime.now().isoformat(),
                    "source": "bash_history",
                })
    
    def _parse_zsh_history(self, lines: List[str]):
        """Parse zsh history file."""
        for line in lines:
            line = line.strip()
            if line and ';' in line:
                # zsh history format: ': timestamp:duration;command'
                parts = line.split(';', 1)
                if len(parts) == 2:
                    command = parts[1].strip()
                    self._shell_state["command_history"].append({
                        "command": command,
                        "timestamp": datetime.now().isoformat(),
                        "source": "zsh_history",
                    })
    
    def _parse_powershell_history(self, lines: List[str]):
        """Parse PowerShell history."""
        # PowerShell history is more complex, using simplified parsing
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                self._shell_state["command_history"].append({
                    "command": line,
                    "timestamp": datetime.now().isoformat(),
                    "source": "powershell_history",
                })
    
    # Core adapter interface implementation
    def normalize_input(self, raw_input: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize shell input to standard format.
        
        Args:
            raw_input: Raw shell input (string)
            context: Additional context
            
        Returns:
            Normalized input dictionary
        """
        if not isinstance(raw_input, str):
            raw_input = str(raw_input)
        
        # Clean and normalize the input
        normalized_input = raw_input.strip()
        
        # Extract potential command parts
        parts = shlex.split(normalized_input, posix=True)
        
        # Detect if this looks like a partial command
        is_partial = normalized_input.endswith(' ') or not normalized_input
        
        # Create metadata
        metadata = {
            "raw_input": raw_input,
            "normalized_input": normalized_input,
            "parts": parts,
            "is_partial": is_partial,
            "input_length": len(normalized_input),
            "word_count": len(parts),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Update shell state
        self._shell_state["last_input"] = normalized_input
        
        return {
            "prefix": normalized_input,
            "context": context or {},
            "metadata": metadata,
        }
    
    def create_request_context(self, 
                              raw_input: Any,
                              session_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create context for shell advination request.
        
        Args:
            raw_input: Raw shell input
            session_data: Session data if available
            
        Returns:
            Context dictionary
        """
        # Update current directory
        try:
            current_dir = os.getcwd()
            self._shell_state["current_directory"] = current_dir
            
            # Get directory contents
            dir_contents = os.listdir(current_dir)
            self._platform_context["current_directory"] = current_dir
            self._platform_context["directory_contents"] = dir_contents
            
        except Exception:
            current_dir = self._shell_state["current_directory"]
            dir_contents = self._platform_context.get("directory_contents", [])
        
        # Build context from shell state
        context = {
            "shell": {
                "type": self.config["shell_type"],
                "current_directory": current_dir,
                "directory_contents": dir_contents,
                "environment": self._shell_state["environment"],
                "history_size": len(self._shell_state["command_history"]),
                "last_command": self._shell_state.get("last_command"),
            },
            "user": self._platform_context["user"],
            "system": {
                "hostname": self._platform_context["hostname"],
                "platform": self._platform_context["platform"],
            },
            "session": session_data or {},
            "input_type": "shell_command",
        }
        
        # Add recent command history as context
        if self._shell_state["command_history"]:
            recent_history = self._shell_state["command_history"][-10:]  # Last 10 commands
            context["shell"]["recent_history"] = [
                cmd["command"] for cmd in recent_history
            ]
        
        return context
    
    def format_response(self, 
                       result: OrchestrationResult,
                       original_input: Any) -> Dict[str, Any]:
        """
        Format orchestration result for shell output.
        
        Args:
            result: Orchestration result
            original_input: Original shell input
            
        Returns:
            Shell-formatted response dictionary
        """
        # Update shell state with result
        self._shell_state["last_result"] = {
            "input": str(original_input),
            "outcome": result.outcome.value,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Build base response
        response = {
            "success": result.outcome.value in ["SUGGEST_EXACT", "SUGGEST_ADAPTED"],
            "outcome": result.outcome.value,
            "input": str(original_input),
            "timestamp": datetime.now().isoformat(),
            "adapter_id": self.adapter_id,
            "shell_type": self.config["shell_type"],
        }
        
        # Format based on outcome
        if result.outcome.value == "SUGGEST_EXACT":
            response.update(self._format_suggestions(result, original_input))
        
        elif result.outcome.value == "SUGGEST_ADAPTED":
            response.update(self._format_suggestions(result, original_input))
            response["adapted"] = True
        
        elif result.outcome.value == "START_DIALOG":
            response.update(self._format_dialog(result, original_input))
        
        elif result.outcome.value == "DEFER":
            response.update(self._format_deferred(result, original_input))
        
        # Add metadata
        if result.metadata:
            response["metadata"] = {
                k: v for k, v in result.metadata.items()
                if not isinstance(v, (dict, list)) or k in ["processing_time_ms"]
            }
        
        # Generate text representation for display
        response["text"] = self._generate_response_text(response)
        
        return response
    
    def _format_suggestions(self, 
                           result: OrchestrationResult,
                           original_input: Any) -> Dict[str, Any]:
        """Format suggestion results."""
        max_display = self.config["max_suggestions_display"]
        suggestions = result.suggestions[:max_display] if result.suggestions else []
        
        formatted = {
            "suggestions": [
                {
                    "text": suggestion.text,
                    "confidence": suggestion.match_score,
                    "source": suggestion.source,
                    "metadata": suggestion.metadata,
                }
                for suggestion in suggestions
            ],
            "suggestion_count": len(result.suggestions) if result.suggestions else 0,
            "display_count": len(suggestions),
        }
        
        # Auto-execute logic
        if (suggestions and 
            suggestions[0].match_score >= self.config["auto_execute_threshold"] and
            not self.config["require_confirmation"]):
            formatted["auto_execute"] = True
            formatted["selected_suggestion"] = suggestions[0].text
        
        return formatted
    
    def _format_dialog(self,
                      result: OrchestrationResult,
                      original_input: Any) -> Dict[str, Any]:
        """Format dialog start result."""
        return {
            "dialog_started": True,
            "dialog_id": result.dialog_id,
            "question": result.first_question,
            "question_type": result.question_type or "text",
            "instructions": "Type your answer to continue the dialog.",
        }
    
    def _format_deferred(self,
                        result: OrchestrationResult,
                        original_input: Any) -> Dict[str, Any]:
        """Format deferred task result."""
        return {
            "deferred": True,
            "task_id": result.task_id,
            "reason": result.reason,
            "retry_after": result.retry_after.isoformat() if result.retry_after else None,
            "message": f"Command deferred: {result.reason}",
        }
    
    def _generate_response_text(self, response: Dict[str, Any]) -> str:
        """Generate human-readable text representation of response."""
        lines = []
        
        if self.config["color_output"]:
            # ANSI color codes
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            BLUE = '\033[94m'
            RED = '\033[91m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
        else:
            GREEN = YELLOW = BLUE = RED = BOLD = RESET = ""
        
        outcome = response["outcome"]
        
        if outcome == "SUGGEST_EXACT":
            lines.append(f"{GREEN}✓ Found exact matches:{RESET}")
            for i, suggestion in enumerate(response.get("suggestions", []), 1):
                conf = suggestion["confidence"]
                conf_str = f" [{conf:.0%}]" if self.config["show_confidence"] else ""
                lines.append(f"  {i}. {BOLD}{suggestion['text']}{RESET}{conf_str}")
            
            if response.get("auto_execute"):
                lines.append(f"{YELLOW}→ Auto-executing: {response['selected_suggestion']}{RESET}")
        
        elif outcome == "SUGGEST_ADAPTED":
            lines.append(f"{YELLOW}⚠ Found adapted matches:{RESET}")
            for i, suggestion in enumerate(response.get("suggestions", []), 1):
                conf = suggestion["confidence"]
                conf_str = f" [{conf:.0%}]" if self.config["show_confidence"] else ""
                lines.append(f"  {i}. {suggestion['text']}{conf_str}")
        
        elif outcome == "START_DIALOG":
            lines.append(f"{BLUE}? {response.get('question', 'Please provide more information:')}{RESET}")
            lines.append(f"{BLUE}  (Dialog ID: {response.get('dialog_id', 'unknown')}){RESET}")
        
        elif outcome == "DEFER":
            lines.append(f"{RED}⏸ Deferred: {response.get('reason', 'Unknown reason')}{RESET}")
            if response.get("task_id"):
                lines.append(f"{RED}  Task ID: {response['task_id']}{RESET}")
        
        # Add metadata if available
        if "metadata" in response and "processing_time_ms" in response["metadata"]:
            time_ms = response["metadata"]["processing_time_ms"]
            lines.append(f"{BLUE}[{time_ms:.1f} ms]{RESET}")
        
        return "\n".join(lines)
    
    # Shell-specific methods
    def get_platform_name(self) -> str:
        """Get platform name."""
        return f"shell_{self.config['shell_type']}"
    
    def get_platform_version(self) -> str:
        """Get platform version."""
        # Try to get shell version
        shell_type = self.config["shell_type"]
        
        if shell_type == "bash":
            try:
                result = subprocess.run(
                    ["bash", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    # Extract version from first line
                    lines = result.stdout.split('\n')
                    if lines:
                        return lines[0].strip()
            except Exception:
                pass
        
        elif shell_type == "zsh":
            try:
                result = subprocess.run(
                    ["zsh", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
        
        elif shell_type == "powershell":
            try:
                result = subprocess.run(
                    ["powershell", "-Command", "$PSVersionTable.PSVersion"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return f"PowerShell {result.stdout.strip()}"
            except Exception:
                pass
        
        return f"Unknown {shell_type} version"
    
    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            **kwargs: Additional parameters
            
        Returns:
            Execution results
        """
        try:
            # Update shell state
            self._shell_state["last_command"] = command
            
            # Add to command history
            self._shell_state["command_history"].append({
                "command": command,
                "timestamp": datetime.now().isoformat(),
                "executed": True,
            })
            
            # Execute command
            timeout = kwargs.get("timeout", 30)
            cwd = kwargs.get("cwd", self._shell_state["current_directory"])
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=self._shell_state["environment"],
            )
            
            # Update current directory if it changed
            if command.startswith("cd "):
                try:
                    new_dir = os.getcwd()
                    self._shell_state["current_directory"] = new_dir
                except Exception:
                    pass
            
            execution_result = {
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "execution_time": None,  # Would need timing
                "timestamp": datetime.now().isoformat(),
            }
            
            # Learn from execution if successful
            if (result.returncode == 0 and 
                self.config["enable_learning"] and 
                self.orchestrator):
                
                # Create learning context
                context = self.create_request_context(command)
                self.orchestrator.adivinator.learn(command, context)
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "error": "Command timed out",
                "timeout_seconds": timeout,
                "success": False,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            return {
                "command": command,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat(),
            }
    
    def get_shell_info(self) -> Dict[str, Any]:
        """Get detailed shell information."""
        info = {
            **self._platform_context,
            "state": self._shell_state,
            "config": {
                k: v for k, v in self.config.items()
                if k not in ["shell_specific"]
            },
            "adapter_info": self.get_adapter_info(),
        }
        
        # Add process info
        info["process"] = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        }
        
        return info
    
    def change_directory(self, path: str) -> Dict[str, Any]:
        """
        Change current directory.
        
        Args:
            path: Path to change to
            
        Returns:
            Result of directory change
        """
        try:
            new_dir = os.path.expanduser(path)
            os.chdir(new_dir)
            
            # Update state
            self._shell_state["current_directory"] = new_dir
            self._platform_context["current_directory"] = new_dir
            
            # Update directory contents
            try:
                dir_contents = os.listdir(new_dir)
                self._platform_context["directory_contents"] = dir_contents
            except Exception:
                self._platform_context["directory_contents"] = []
            
            return {
                "success": True,
                "old_directory": self._shell_state.get("previous_directory"),
                "new_directory": new_dir,
                "message": f"Changed directory to {new_dir}",
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to change directory: {e}",
            }
    
    def get_command_suggestions(self, 
                               prefix: str,
                               limit: int = None) -> List[Dict[str, Any]]:
        """
        Get command suggestions for a prefix.
        
        Args:
            prefix: Command prefix
            limit: Maximum number of suggestions
            
        Returns:
            List of command suggestions
        """
        if not self.orchestrator:
            return []
        
        result = self.orchestrator.process_request(
            prefix=prefix,
            context=self.create_request_context(prefix)
        )
        
        if not result.suggestions:
            return []
        
        max_limit = limit or self.config["max_suggestions_display"]
        suggestions = result.suggestions[:max_limit]
        
        return [
            {
                "text": suggestion.text,
                "confidence": suggestion.match_score,
                "source": suggestion.source,
                "metadata": suggestion.metadata,
            }
            for suggestion in suggestions
        ]
    
    def interactive_session(self, 
                           prompt: str = None,
                           exit_commands: List[str] = None):
        """
        Start an interactive shell session.
        
        Args:
            prompt: Custom prompt
            exit_commands: Commands to exit the session
        """
        if not prompt:
            platform = self.get_platform_name()
            prompt = self.config["prompt_format"].format(platform=platform)
        
        if not exit_commands:
            exit_commands = ["exit", "quit", "q"]
        
        print(f"Starting interactive {self.get_platform_name()} session")
        print("Type 'help' for available commands, or an exit command to quit.")
        print()
        
        session_id = f"interactive_{datetime.now().timestamp()}"
        
        while True:
            try:
                # Read input
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in exit_commands:
                    print("Exiting interactive session.")
                    break
                
                # Check for special commands
                if self._handle_special_command(user_input, session_id):
                    continue
                
                # Process through adapter
                response = self.process(user_input, session_id=session_id)
                
                # Display response
                if "text" in response:
                    print(response["text"])
                
                # Handle auto-execution
                if (response.get("auto_execute") and 
                    response.get("selected_suggestion")):
                    
                    command = response["selected_suggestion"]
                    print(f"Auto-executing: {command}")
                    
                    exec_result = self.execute_command(command)
                    if exec_result.get("stdout"):
                        print(exec_result["stdout"])
                    if exec_result.get("stderr"):
                        print(f"Error: {exec_result['stderr']}")
                
                # Handle dialog continuation
                elif (response.get("dialog_started") and 
                      response.get("dialog_id")):
                    
                    self._handle_interactive_dialog(response["dialog_id"])
                
            except KeyboardInterrupt:
                print("\nInterrupted. Use 'exit' to quit.")
            except EOFError:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _handle_special_command(self, command: str, session_id: str) -> bool:
        """Handle special interactive commands."""
        command_lower = command.lower()
        
        if command_lower in ["help", "?"]:
            self._show_help()
            return True
        
        elif command_lower in ["clear", "cls"]:
            os.system("cls" if os.name == "nt" else "clear")
            return True
        
        elif command_lower.startswith("cd "):
            path = command[3:].strip()
            result = self.change_directory(path)
            print(result.get("message", "Directory changed"))
            return True
        
        elif command_lower == "pwd":
            print(self._shell_state["current_directory"])
            return True
        
        elif command_lower == "history":
            self._show_history()
            return True
        
        elif command_lower == "info":
            info = self.get_shell_info()
            print(json.dumps(info, indent=2, default=str))
            return True
        
        elif command_lower.startswith("config "):
            parts = command.split()
            if len(parts) == 2:
                value = self.get_config(parts[1])
                print(f"{parts[1]} = {value}")
            elif len(parts) == 3:
                # Simple config update
                key = parts[1]
                try:
                    value = eval(parts[2])  # Simple eval for bool/numbers
                except:
                    value = parts[2]
                
                self.update_config({key: value})
                print(f"Updated {key} = {value}")
            return True
        
        return False
    
    def _show_help(self):
        """Show help for interactive session."""
        help_text = """
Available commands:
  <command>           - Get suggestions for a command
  cd <path>          - Change directory
  pwd                - Print current directory
  clear/cls          - Clear screen
  history            - Show command history
  info               - Show shell information
  config <key>       - Get configuration value
  config <key> <value> - Set configuration value
  help/?             - Show this help
  exit/quit/q        - Exit interactive session
  
Special features:
  - Tab completion for commands
  - Auto-execution for high-confidence matches
  - Command history with up/down arrows
  - Dialog-based command composition
        """
        print(help_text)
    
    def _show_history(self):
        """Show command history."""
        history = self._shell_state["command_history"][-20:]  # Last 20 commands
        if not history:
            print("No command history")
            return
        
        for i, entry in enumerate(history, 1):
            timestamp = entry.get("timestamp", "unknown")
            command = entry.get("command", "")
            source = entry.get("source", "interactive")
            print(f"{i:3}. [{timestamp}] {command} ({source})")
    
    def _handle_interactive_dialog(self, dialog_id: str):
        """Handle interactive dialog continuation."""
        print("Dialog started. Answer questions to compose command.")
        print("Type 'cancel' to cancel the dialog.")
        print()
        
        while True:
            try:
                # Get current dialog state
                dialog_context = self._get_dialog_context(dialog_id)
                if not dialog_context:
                    print("Dialog not found or expired.")
                    break
                
                # Get next question
                result = self.orchestrator.continue_dialog(dialog_id, "")
                if result.outcome.value != "START_DIALOG":
                    # Dialog completed
                    if result.suggestions:
                        suggestion = result.suggestions[0]
                        print(f"\nComposed command: {suggestion.text}")
                        
                        # Ask for execution
                        execute = input("Execute this command? (y/N): ").lower()
                        if execute in ["y", "yes"]:
                            exec_result = self.execute_command(suggestion.text)
                            if exec_result.get("stdout"):
                                print(exec_result["stdout"])
                    break
                
                # Show question and get answer
                question = result.first_question
                user_answer = input(f"{question}: ").strip()
                
                if user_answer.lower() == "cancel":
                    print("Dialog cancelled.")
                    self._cleanup_dialog_context(dialog_id)
                    break
                
                # Continue dialog with answer
                result = self.orchestrator.continue_dialog(dialog_id, user_answer)
                
                # Check if dialog completed
                if result.outcome.value != "START_DIALOG":
                    if result.suggestions:
                        suggestion = result.suggestions[0]
                        print(f"\nComposed command: {suggestion.text}")
                        
                        # Ask for execution
                        execute = input("Execute this command? (y/N): ").lower()
                        if execute in ["y", "yes"]:
                            exec_result = self.execute_command(suggestion.text)
                            if exec_result.get("stdout"):
                                print(exec_result["stdout"])
                    break
                
            except KeyboardInterrupt:
                print("\nDialog interrupted.")
                break
            except Exception as e:
                print(f"Dialog error: {e}")
                break
    
    # Override base methods for shell-specific behavior
    def start(self):
        """Start shell adapter."""
        super().start()
        
        # Save readline history on exit
        import atexit
        
        def save_readline_history():
            try:
                history_file = os.path.expanduser("~/.advination_history")
                readline.write_history_file(history_file)
            except Exception:
                pass
        
        atexit.register(save_readline_history)
        
        print(f"Shell adapter started for {self.config['shell_type']}")
        print(f"Working directory: {self._shell_state['current_directory']}")
    
    def stop(self):
        """Stop shell adapter."""
        # Save shell history
        if self.config["enable_history_integration"]:
            self._save_shell_history()
        
        super().stop()
    
    def _save_shell_history(self):
        """Save shell command history."""
        # In a real implementation, this would update the shell's history file
        # For now, just save to our own history file
        history_file = os.path.expanduser("~/.advination_shell_history")
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                for entry in self._shell_state["command_history"]:
                    f.write(f"{entry['timestamp']}|{entry['command']}\n")
        except Exception as e:
            print(f"Warning: Could not save shell history: {e}")


# Shell-specific utility functions
def create_shell_adapter(orchestrator: ProductionOrchestrator = None,
                        shell_type: str = None,
                        config: Dict[str, Any] = None) -> ShellAdapter:
    """
    Create and configure a shell adapter.
    
    Args:
        orchestrator: Orchestrator instance
        shell_type: Specific shell type
        config: Additional configuration
        
    Returns:
        Configured ShellAdapter instance
    """
    adapter_config = config or {}
    if shell_type:
        adapter_config["shell_type"] = shell_type
    
    adapter = ShellAdapter(orchestrator, adapter_config)
    adapter.start()
    return adapter


def detect_available_shells() -> List[Dict[str, Any]]:
    """
    Detect available shells on the system.
    
    Returns:
        List of available shells with information
    """
    shells = []
    common_shells = [
        ("bash", ["bash", "--version"]),
        ("zsh", ["zsh", "--version"]),
        ("powershell", ["powershell", "-Command", "$PSVersionTable.PSVersion"]),
        ("fish", ["fish", "--version"]),
        ("dash", ["dash", "--version"]),
        ("ksh", ["ksh", "--version"]),
        ("tcsh", ["tcsh", "--version"]),
    ]
    
    for shell_name, test_command in common_shells:
        try:
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                timeout=2,
                shell=False,
            )
            
            if result.returncode == 0:
                shells.append({
                    "name": shell_name,
                    "available": True,
                    "version": result.stdout.strip()[:100],  # Truncate
                    "path": shutil.which(shell_name),
                })
            else:
                shells.append({
                    "name": shell_name,
                    "available": False,
                    "path": shutil.which(shell_name),
                })
                
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            shells.append({
                "name": shell_name,
                "available": False,
            })
    
    return shells


# Register with adapter registry
try:
    from .base_adapter import AdapterRegistry
    AdapterRegistry.register("shell", ShellAdapter)
    AdapterRegistry.register("bash", ShellAdapter)  # Alias
    AdapterRegistry.register("powershell", ShellAdapter)  # Alias
    AdapterRegistry.register("zsh", ShellAdapter)  # Alias
except ImportError:
    pass


# Import for type hints at the end to avoid circular imports
import json
import shutil