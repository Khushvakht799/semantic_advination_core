"""
API adapter for semantic advination system.
Provides REST API and WebSocket interfaces for remote clients.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime
import uuid
from enum import Enum
import logging

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for type hints
    class FastAPI: pass
    class HTTPException(Exception): pass
    class WebSocket: pass
    class BaseModel: pass
    class Field: pass

from .base_adapter import BaseAdapter, AdapterError, create_adapter
from core.models import OrchestrationResult, CommandSuggestion
from core.orchestrator import ProductionOrchestrator


class APIAdapter(BaseAdapter):
    """
    Adapter for API-based access to advination system.
    Provides REST endpoints and WebSocket connections.
    """
    
    def __init__(self, 
                 orchestrator: Optional[ProductionOrchestrator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize API adapter.
        
        Args:
            orchestrator: Core orchestrator instance
            config: API-specific configuration
        """
        # Default API config
        default_config = {
            "api_version": "v1",
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "enable_cors": True,
            "cors_origins": ["*"],
            "enable_websocket": True,
            "websocket_heartbeat": 30,
            "max_connections": 100,
            "rate_limit_per_minute": 60,
            "auth_required": False,
            "auth_tokens": [],
            "enable_docs": True,
            "docs_url": "/docs",
            "enable_metrics": True,
            "metrics_url": "/metrics",
            "request_timeout": 30,
            "max_request_size": 1048576,  # 1MB
            "log_level": "INFO",
            "enable_ssl": False,
            "ssl_cert": None,
            "ssl_key": None,
        }
        
        # Merge with user config
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        
        super().__init__(orchestrator, merged_config)
        
        # API state
        self._api_state = {
            "server_started": False,
            "active_connections": 0,
            "websocket_connections": {},
            "total_requests": 0,
            "failed_requests": 0,
            "start_time": datetime.now().isoformat(),
        }
        
        # Rate limiting
        self._rate_limits = {}
        
        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logging.warning("FastAPI not available. API adapter will be limited.")
        
        # Initialize WebSocket manager
        self._websocket_manager = WebSocketManager(self) if merged_config["enable_websocket"] else None
        
        # Initialize request handlers
        self._setup_request_handlers()
    
    def _initialize_platform_context(self) -> Dict[str, Any]:
        """
        Initialize API-specific context.
        
        Returns:
            API context dictionary
        """
        return {
            "api_version": self.config["api_version"],
            "host": self.config["host"],
            "port": self.config["port"],
            "protocol": "https" if self.config["enable_ssl"] else "http",
            "environment": "development" if self.config["debug"] else "production",
            "features": {
                "websocket": self.config["enable_websocket"],
                "cors": self.config["enable_cors"],
                "auth": self.config["auth_required"],
                "metrics": self.config["enable_metrics"],
            },
        }
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Semantic Advination API",
            description="API for semantic command advination system",
            version=self.config["api_version"],
            docs_url=self.config["docs_url"] if self.config["enable_docs"] else None,
            redoc_url=None,
        )
        
        # Add CORS middleware
        if self.config["enable_cors"]:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config["cors_origins"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add request logging middleware
        @app.middleware("http")
        async def log_requests(request, call_next):
            request_id = str(uuid.uuid4())[:8]
            start_time = datetime.now()
            
            # Log request
            logging.info(f"Request {request_id}: {request.method} {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Log response
            process_time = (datetime.now() - start_time).total_seconds() * 1000
            logging.info(f"Response {request_id}: {response.status_code} ({process_time:.1f}ms)")
            
            # Add request ID to headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.1f}ms"
            
            return response
        
        # Authentication middleware
        if self.config["auth_required"]:
            @app.middleware("http")
            async def authenticate_request(request, call_next):
                # Skip auth for docs and metrics
                if request.url.path in [self.config["docs_url"], self.config["metrics_url"]]:
                    return await call_next(request)
                
                # Check for API key
                api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
                
                if not api_key or api_key not in self.config["auth_tokens"]:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid or missing API key"}
                    )
                
                return await call_next(request)
        
        return app
    
    def _setup_request_handlers(self):
        """Setup API request handlers."""
        if not self.app:
            return
        
        # Define Pydantic models for request/response
        class AdvinationRequest(BaseModel):
            """Request model for advination."""
            prefix: str = Field(..., description="Command prefix to advinate")
            context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
            session_id: Optional[str] = Field(None, description="Session identifier")
            user_id: Optional[str] = Field(None, description="User identifier")
            metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")
            
            @validator('prefix')
            def prefix_not_empty(cls, v):
                if not v or not v.strip():
                    raise ValueError('prefix cannot be empty')
                return v.strip()
        
        class DialogRequest(BaseModel):
            """Request model for dialog continuation."""
            dialog_id: str = Field(..., description="Dialog identifier")
            answer: str = Field(..., description="User's answer")
            session_id: Optional[str] = Field(None, description="Session identifier")
            metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")
        
        class CommandExecutionRequest(BaseModel):
            """Request model for command execution."""
            command: str = Field(..., description="Command to execute")
            context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
            session_id: Optional[str] = Field(None, description="Session identifier")
            metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata")
        
        # Root endpoint
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "service": "Semantic Advination API",
                "version": self.config["api_version"],
                "status": "operational",
                "adapter_id": self.adapter_id,
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "advinate": f"/api/{self.config['api_version']}/advinate",
                    "continue_dialog": f"/api/{self.config['api_version']}/dialog/continue",
                    "execute": f"/api/{self.config['api_version']}/execute",
                    "health": "/health",
                    "metrics": self.config["metrics_url"] if self.config["enable_metrics"] else None,
                    "docs": self.config["docs_url"] if self.config["enable_docs"] else None,
                },
                "features": self._platform_context["features"],
            }
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "adapter": {
                    "id": self.adapter_id,
                    "initialized": self._state["initialized"],
                    "requests_processed": self._state["processed_requests"],
                    "errors": self._state["errors"],
                },
                "orchestrator": self.orchestrator is not None,
                "api": {
                    "active_connections": self._api_state["active_connections"],
                    "websocket_connections": len(self._api_state["websocket_connections"]),
                },
                "uptime": self._get_uptime(),
            }
            
            # Check critical components
            if not self.orchestrator:
                health_status["status"] = "degraded"
                health_status["issues"] = ["orchestrator_not_available"]
            
            return health_status
        
        # Metrics endpoint
        if self.config["enable_metrics"]:
            @self.app.get(self.config["metrics_url"])
            async def metrics():
                """Metrics endpoint."""
                return self.get_metrics()
        
        # API v1 endpoints
        api_prefix = f"/api/{self.config['api_version']}"
        
        @self.app.post(f"{api_prefix}/advinate")
        async def advinate(request: AdvinationRequest):
            """
            Advinate command completion.
            
            Args:
                request: Advination request
                
            Returns:
                Advination response
            """
            # Check rate limit
            client_id = request.session_id or request.user_id or "anonymous"
            if not self._check_rate_limit(client_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            try:
                # Process request
                response = self.process(
                    raw_input=request.prefix,
                    context=request.context,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    **request.metadata,
                )
                
                # Update API state
                self._api_state["total_requests"] += 1
                
                return response
                
            except Exception as e:
                self._api_state["failed_requests"] += 1
                raise HTTPException(
                    status_code=500,
                    detail=f"Advination failed: {str(e)}"
                )
        
        @self.app.post(f"{api_prefix}/dialog/continue")
        async def continue_dialog(request: DialogRequest):
            """
            Continue a dialog.
            
            Args:
                request: Dialog continuation request
                
            Returns:
                Dialog continuation response
            """
            try:
                response = self.continue_dialog(
                    dialog_id=request.dialog_id,
                    user_response=request.answer,
                    session_id=request.session_id,
                    **request.metadata,
                )
                
                return response
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Dialog continuation failed: {str(e)}"
                )
        
        @self.app.post(f"{api_prefix}/execute")
        async def execute_command(request: CommandExecutionRequest):
            """
            Execute a command (if supported by platform).
            
            Args:
                request: Command execution request
                
            Returns:
                Execution result
            """
            try:
                # This is a placeholder - actual execution depends on platform
                # In API context, we typically don't execute arbitrary commands
                # For safety, we only allow simulated execution
                result = {
                    "command": request.command,
                    "executed": False,
                    "message": "Command execution not supported in API mode",
                    "timestamp": datetime.now().isoformat(),
                }
                
                return result
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Command execution failed: {str(e)}"
                )
        
        @self.app.get(f"{api_prefix}/suggestions")
        async def get_suggestions(
            prefix: str,
            limit: int = 5,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
        ):
            """
            Get command suggestions.
            
            Args:
                prefix: Command prefix
                limit: Maximum number of suggestions
                session_id: Session identifier
                user_id: User identifier
                
            Returns:
                List of suggestions
            """
            try:
                if not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                result = self.orchestrator.process_request(
                    prefix=prefix,
                    context={"source": "api_suggestions"},
                    user_id=user_id,
                )
                
                suggestions = []
                if result.suggestions:
                    suggestions = [
                        {
                            "text": s.text,
                            "confidence": s.match_score,
                            "source": s.source,
                            "metadata": s.metadata,
                        }
                        for s in result.suggestions[:limit]
                    ]
                
                return {
                    "prefix": prefix,
                    "suggestions": suggestions,
                    "count": len(suggestions),
                    "outcome": result.outcome.value,
                }
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get suggestions: {str(e)}"
                )
        
        @self.app.get(f"{api_prefix}/history")
        async def get_history(
            limit: int = 50,
            session_id: Optional[str] = None,
        ):
            """
            Get processing history.
            
            Args:
                limit: Maximum number of entries
                session_id: Optional session filter
                
            Returns:
                History entries
            """
            history = self.get_history(limit, session_id)
            return {
                "entries": history,
                "count": len(history),
                "session_id": session_id,
            }
        
        @self.app.get(f"{api_prefix}/stats")
        async def get_stats():
            """Get adapter statistics."""
            return self.get_stats()
        
        @self.app.get(f"{api_prefix}/config")
        async def get_config(key: Optional[str] = None):
            """
            Get configuration.
            
            Args:
                key: Optional configuration key
                
            Returns:
                Configuration value(s)
            """
            value = self.get_config(key)
            return {"key": key, "value": value}
        
        @self.app.patch(f"{api_prefix}/config")
        async def update_config(updates: Dict[str, Any]):
            """
            Update configuration.
            
            Args:
                updates: Configuration updates
                
            Returns:
                Updated configuration
            """
            self.update_config(updates, merge=True)
            return {"updated": True, "config": self.get_config()}
        
        # WebSocket endpoint
        if self.config["enable_websocket"] and self._websocket_manager:
            @self.app.websocket(f"{api_prefix}/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time communication."""
                await self._websocket_manager.handle_connection(websocket)
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed
        """
        if not self.config["rate_limit_per_minute"]:
            return True
        
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        client_key = f"{client_id}:{minute_key}"
        
        if client_key not in self._rate_limits:
            self._rate_limits[client_key] = {
                "count": 1,
                "first_request": now,
            }
            return True
        
        limit_data = self._rate_limits[client_key]
        limit_data["count"] += 1
        
        # Check if limit exceeded
        if limit_data["count"] > self.config["rate_limit_per_minute"]:
            return False
        
        # Clean old entries
        self._clean_old_rate_limits()
        
        return True
    
    def _clean_old_rate_limits(self):
        """Clean old rate limit entries."""
        now = datetime.now()
        keys_to_remove = []
        
        for key, data in self._rate_limits.items():
            if (now - data["first_request"]).total_seconds() > 120:  # 2 minutes
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._rate_limits[key]
    
    def _get_uptime(self) -> str:
        """Get adapter uptime as string."""
        try:
            start_time = datetime.fromisoformat(self._state["started_at"])
            uptime = datetime.now() - start_time
            
            days = uptime.days
            hours, remainder = divmod(uptime.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            parts.append(f"{seconds}s")
            
            return " ".join(parts)
        except:
            return "unknown"
    
    # Core adapter interface implementation
    def normalize_input(self, raw_input: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize API input to standard format.
        
        Args:
            raw_input: Raw API input (string or dict)
            context: Additional context
            
        Returns:
            Normalized input dictionary
        """
        # API can receive either string prefix or structured input
        if isinstance(raw_input, dict):
            # Structured input
            prefix = raw_input.get("prefix", "")
            metadata = {k: v for k, v in raw_input.items() if k != "prefix"}
        else:
            # String input
            prefix = str(raw_input)
            metadata = {}
        
        # Merge with context metadata
        if context and "metadata" in context:
            metadata.update(context["metadata"])
        
        return {
            "prefix": prefix.strip(),
            "context": context or {},
            "metadata": {
                **metadata,
                "input_type": "api",
                "timestamp": datetime.now().isoformat(),
                "adapter_id": self.adapter_id,
            },
        }
    
    def create_request_context(self, 
                              raw_input: Any,
                              session_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create context for API advination request.
        
        Args:
            raw_input: Raw API input
            session_data: Session data if available
            
        Returns:
            Context dictionary
        """
        # Build context from API state and input
        context = {
            "source": "api",
            "api_version": self.config["api_version"],
            "request_timestamp": datetime.now().isoformat(),
            "adapter": {
                "id": self.adapter_id,
                "type": "api",
                "version": self.config["api_version"],
            },
            "session": session_data or {},
        }
        
        # Add client information if available
        if isinstance(raw_input, dict) and "client_info" in raw_input:
            context["client"] = raw_input["client_info"]
        
        # Add API-specific metadata
        context["api"] = {
            "active_connections": self._api_state["active_connections"],
            "total_requests": self._api_state["total_requests"],
        }
        
        return context
    
    def format_response(self, 
                       result: OrchestrationResult,
                       original_input: Any) -> Dict[str, Any]:
        """
        Format orchestration result for API response.
        
        Args:
            result: Orchestration result
            original_input: Original API input
            
        Returns:
            API-formatted response dictionary
        """
        # Base response structure
        response = {
            "success": result.outcome.value in ["SUGGEST_EXACT", "SUGGEST_ADAPTED"],
            "outcome": result.outcome.value,
            "input": str(original_input),
            "timestamp": datetime.now().isoformat(),
            "adapter_id": self.adapter_id,
            "api_version": self.config["api_version"],
        }
        
        # Add suggestions if available
        if result.suggestions:
            response["suggestions"] = [
                {
                    "text": suggestion.text,
                    "confidence": suggestion.match_score,
                    "source": suggestion.source,
                    "metadata": suggestion.metadata,
                }
                for suggestion in result.suggestions
            ]
            response["suggestion_count"] = len(result.suggestions)
        
        # Add dialog information
        if result.dialog_id:
            response["dialog"] = {
                "id": result.dialog_id,
                "question": result.first_question,
                "question_type": result.question_type,
            }
        
        # Add task information for deferred results
        if result.task_id:
            response["task"] = {
                "id": result.task_id,
                "reason": result.reason,
                "retry_after": result.retry_after.isoformat() if result.retry_after else None,
                "priority": result.priority,
            }
        
        # Add metadata
        if result.metadata:
            response["metadata"] = result.metadata
        
        # Add processing information
        response["processing"] = {
            "adapter": self.adapter_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        return response
    
    def get_platform_name(self) -> str:
        """Get platform name."""
        return "api"
    
    def get_platform_version(self) -> str:
        """Get platform version."""
        return self.config["api_version"]
    
    # API-specific methods
    async def start_server(self):
        """
        Start the API server.
        
        Note: Requires FastAPI and uvicorn.
        """
        if not FASTAPI_AVAILABLE:
            raise AdapterError("FastAPI is required to start the API server")
        
        if self._api_state["server_started"]:
            raise AdapterError("API server is already running")
        
        try:
            import uvicorn
            
            # Configure logging
            log_level = self.config["log_level"].lower()
            log_config = None
            
            if self.config["debug"]:
                log_level = "debug"
            
            # Start server
            uvicorn_config = {
                "app": self.app,
                "host": self.config["host"],
                "port": self.config["port"],
                "log_level": log_level,
                "access_log": self.config["debug"],
            }
            
            if self.config["enable_ssl"] and self.config["ssl_cert"] and self.config["ssl_key"]:
                uvicorn_config["ssl_certfile"] = self.config["ssl_cert"]
                uvicorn_config["ssl_keyfile"] = self.config["ssl_key"]
            
            print(f"Starting API server on {self.config['host']}:{self.config['port']}")
            print(f"API version: {self.config['api_version']}")
            print(f"Documentation: http://{self.config['host']}:{self.config['port']}{self.config['docs_url']}")
            
            self._api_state["server_started"] = True
            
            # Run server
            await uvicorn.run(**uvicorn_config)
            
        except ImportError as e:
            raise AdapterError(f"Required packages not installed: {e}")
        except Exception as e:
            self._api_state["server_started"] = False
            raise AdapterError(f"Failed to start API server: {e}")
    
    def stop_server(self):
        """Stop the API server."""
        # In a real implementation, this would stop the uvicorn server
        # For now, just update state
        self._api_state["server_started"] = False
        print("API server stopped")
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information."""
        return {
            "adapter": self.get_adapter_info(),
            "api": {
                "version": self.config["api_version"],
                "host": self.config["host"],
                "port": self.config["port"],
                "started": self._api_state["server_started"],
                "start_time": self._api_state["start_time"],
                "active_connections": self._api_state["active_connections"],
                "total_requests": self._api_state["total_requests"],
                "failed_requests": self._api_state["failed_requests"],
            },
            "endpoints": self._get_endpoints_list(),
            "features": self._platform_context["features"],
        }
    
    def _get_endpoints_list(self) -> List[Dict[str, Any]]:
        """Get list of available endpoints."""
        api_prefix = f"/api/{self.config['api_version']}"
        
        endpoints = [
            {
                "path": "/",
                "method": "GET",
                "description": "Root endpoint with API information",
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check",
            },
            {
                "path": f"{api_prefix}/advinate",
                "method": "POST",
                "description": "Advinate command completion",
            },
            {
                "path": f"{api_prefix}/dialog/continue",
                "method": "POST",
                "description": "Continue a dialog",
            },
            {
                "path": f"{api_prefix}/suggestions",
                "method": "GET",
                "description": "Get command suggestions",
            },
            {
                "path": f"{api_prefix}/history",
                "method": "GET",
                "description": "Get processing history",
            },
            {
                "path": f"{api_prefix}/stats",
                "method": "GET",
                "description": "Get adapter statistics",
            },
            {
                "path": f"{api_prefix}/config",
                "method": "GET",
                "description": "Get configuration",
            },
            {
                "path": f"{api_prefix}/config",
                "method": "PATCH",
                "description": "Update configuration",
            },
        ]
        
        if self.config["enable_websocket"]:
            endpoints.append({
                "path": f"{api_prefix}/ws",
                "method": "WEBSOCKET",
                "description": "WebSocket for real-time communication",
            })
        
        if self.config["enable_metrics"]:
            endpoints.append({
                "path": self.config["metrics_url"],
                "method": "GET",
                "description": "Metrics endpoint",
            })
        
        if self.config["enable_docs"]:
            endpoints.append({
                "path": self.config["docs_url"],
                "method": "GET",
                "description": "API documentation",
            })
        
        return endpoints
    
    def broadcast_message(self, 
                         message_type: str,
                         data: Dict[str, Any],
                         session_ids: List[str] = None):
        """
        Broadcast message to WebSocket clients.
        
        Args:
            message_type: Type of message
            data: Message data
            session_ids: Optional list of specific sessions
        """
        if not self._websocket_manager:
            return
        
        self._websocket_manager.broadcast(message_type, data, session_ids)
    
    # Override base methods for API-specific behavior
    def start(self):
        """Start API adapter."""
        super().start()
        
        # Initialize WebSocket manager
        if self.config["enable_websocket"] and not self._websocket_manager:
            self._websocket_manager = WebSocketManager(self)
        
        print(f"API adapter started (version: {self.config['api_version']})")
        print(f"Endpoints available at: http://{self.config['host']}:{self.config['port']}/")
    
    def stop(self):
        """Stop API adapter."""
        # Stop WebSocket connections
        if self._websocket_manager:
            self._websocket_manager.disconnect_all()
        
        # Stop server if running
        if self._api_state["server_started"]:
            self.stop_server()
        
        super().stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics with API-specific metrics."""
        base_stats = super().get_stats()
        
        api_stats = {
            "api": {
                "server_started": self._api_state["server_started"],
                "active_connections": self._api_state["active_connections"],
                "websocket_connections": len(self._api_state["websocket_connections"]),
                "total_requests": self._api_state["total_requests"],
                "failed_requests": self._api_state["failed_requests"],
                "success_rate": (
                    (self._api_state["total_requests"] - self._api_state["failed_requests"]) /
                    max(self._api_state["total_requests"], 1)
                ),
                "start_time": self._api_state["start_time"],
                "uptime": self._get_uptime(),
            },
            "rate_limits": {
                "active_entries": len(self._rate_limits),
                "limit_per_minute": self.config["rate_limit_per_minute"],
            },
        }
        
        return {**base_stats, **api_stats}


class WebSocketManager:
    """Manager for WebSocket connections."""
    
    def __init__(self, adapter: 'APIAdapter'):
        self.adapter = adapter
        self.connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict[str, Any]] = {}
    
    async def handle_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        connection_id = str(uuid.uuid4())[:8]
        
        try:
            # Accept connection
            await websocket.accept()
            
            # Register connection
            self.connections[connection_id] = websocket
            self.connection_info[connection_id] = {
                "connected_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "session_id": None,
                "user_id": None,
            }
            
            # Update adapter state
            self.adapter._api_state["active_connections"] += 1
            self.adapter._api_state["websocket_connections"][connection_id] = {
                "connected_at": datetime.now().isoformat(),
            }
            
            # Send welcome message
            await self.send_message(connection_id, "connected", {
                "connection_id": connection_id,
                "message": "WebSocket connected successfully",
                "timestamp": datetime.now().isoformat(),
                "adapter_id": self.adapter.adapter_id,
                "api_version": self.adapter.config["api_version"],
            })
            
            # Handle messages
            await self._handle_messages(connection_id, websocket)
            
        except WebSocketDisconnect:
            # Client disconnected normally
            await self._handle_disconnect(connection_id)
            
        except Exception as e:
            # Error handling
            logging.error(f"WebSocket error for {connection_id}: {e}")
            await self._handle_disconnect(connection_id)
    
    async def _handle_messages(self, connection_id: str, websocket: WebSocket):
        """Handle incoming WebSocket messages."""
        try:
            while True:
                # Receive message
                message = await websocket.receive_text()
                
                # Update activity
                self.connection_info[connection_id]["last_activity"] = datetime.now().isoformat()
                
                try:
                    # Parse message
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    # Handle different message types
                    if message_type == "advinate":
                        await self._handle_advinate(connection_id, data)
                    elif message_type == "continue_dialog":
                        await self._handle_continue_dialog(connection_id, data)
                    elif message_type == "subscribe":
                        await self._handle_subscribe(connection_id, data)
                    elif message_type == "unsubscribe":
                        await self._handle_unsubscribe(connection_id, data)
                    elif message_type == "ping":
                        await self._handle_ping(connection_id, data)
                    else:
                        await self.send_error(connection_id, f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    await self.send_error(connection_id, "Invalid JSON message")
                except Exception as e:
                    await self.send_error(connection_id, f"Message processing failed: {str(e)}")
                    
        except WebSocketDisconnect:
            # Client disconnected
            pass
    
    async def _handle_advinate(self, connection_id: str, data: Dict[str, Any]):
        """Handle advination request over WebSocket."""
        try:
            prefix = data.get("prefix", "")
            context = data.get("context", {})
            session_id = data.get("session_id")
            user_id = data.get("user_id")
            
            # Update connection info
            if session_id:
                self.connection_info[connection_id]["session_id"] = session_id
            if user_id:
                self.connection_info[connection_id]["user_id"] = user_id
            
            # Process request
            response = self.adapter.process(
                raw_input=prefix,
                context=context,
                session_id=session_id,
                user_id=user_id,
                source="websocket",
            )
            
            # Send response
            await self.send_message(connection_id, "advination_result", response)
            
        except Exception as e:
            await self.send_error(connection_id, f"Advination failed: {str(e)}")
    
    async def _handle_continue_dialog(self, connection_id: str, data: Dict[str, Any]):
        """Handle dialog continuation over WebSocket."""
        try:
            dialog_id = data.get("dialog_id")
            answer = data.get("answer", "")
            session_id = data.get("session_id")
            
            if not dialog_id:
                await self.send_error(connection_id, "Missing dialog_id")
                return
            
            # Continue dialog
            response = self.adapter.continue_dialog(
                dialog_id=dialog_id,
                user_response=answer,
                session_id=session_id,
            )
            
            # Send response
            await self.send_message(connection_id, "dialog_result", response)
            
        except ValueError as e:
            await self.send_error(connection_id, f"Dialog not found: {str(e)}")
        except Exception as e:
            await self.send_error(connection_id, f"Dialog continuation failed: {str(e)}")
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle subscription request."""
        subscription_type = data.get("subscription_type", "")
        
        if subscription_type == "notifications":
            self.connection_info[connection_id]["subscribed_notifications"] = True
            await self.send_message(connection_id, "subscription_confirmed", {
                "subscription_type": "notifications",
                "message": "Subscribed to notifications",
            })
        else:
            await self.send_error(connection_id, f"Unknown subscription type: {subscription_type}")
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """Handle unsubscription request."""
        subscription_type = data.get("subscription_type", "")
        
        if subscription_type == "notifications":
            self.connection_info[connection_id]["subscribed_notifications"] = False
            await self.send_message(connection_id, "unsubscription_confirmed", {
                "subscription_type": "notifications",
                "message": "Unsubscribed from notifications",
            })
    
    async def _handle_ping(self, connection_id: str, data: Dict[str, Any]):
        """Handle ping message."""
        await self.send_message(connection_id, "pong", {
            "timestamp": datetime.now().isoformat(),
            "original_data": data.get("data"),
        })
    
    async def _handle_disconnect(self, connection_id: str):
        """Handle client disconnection."""
        if connection_id in self.connections:
            # Update adapter state
            self.adapter._api_state["active_connections"] -= 1
            if connection_id in self.adapter._api_state["websocket_connections"]:
                del self.adapter._api_state["websocket_connections"][connection_id]
            
            # Remove connection
            del self.connections[connection_id]
            if connection_id in self.connection_info:
                del self.connection_info[connection_id]
            
            logging.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(self, 
                          connection_id: str,
                          message_type: str,
                          data: Dict[str, Any]):
        """Send message to specific connection."""
        if connection_id not in self.connections:
            return
        
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }
            
            await self.connections[connection_id].send_text(json.dumps(message))
            
        except Exception as e:
            logging.error(f"Failed to send message to {connection_id}: {e}")
            await self._handle_disconnect(connection_id)
    
    async def send_error(self, connection_id: str, error_message: str):
        """Send error message."""
        await self.send_message(connection_id, "error", {
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
        })
    
    async def broadcast(self,
                       message_type: str,
                       data: Dict[str, Any],
                       connection_ids: List[str] = None):
        """
        Broadcast message to connections.
        
        Args:
            message_type: Type of message
            data: Message data
            connection_ids: Optional list of specific connections
        """
        targets = connection_ids or list(self.connections.keys())
        
        for connection_id in targets:
            if connection_id in self.connections:
                await self.send_message(connection_id, message_type, data)
    
    def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        for connection_id in list(self.connections.keys()):
            try:
                # In a real implementation, we would properly close the connection
                del self.connections[connection_id]
            except:
                pass
        
        self.connections.clear()
        self.connection_info.clear()


# API adapter factory and utilities
def create_api_adapter(orchestrator: ProductionOrchestrator = None,
                      config: Dict[str, Any] = None) -> APIAdapter:
    """
    Create and configure an API adapter.
    
    Args:
        orchestrator: Orchestrator instance
        config: API configuration
        
    Returns:
        Configured APIAdapter instance
    """
    adapter = APIAdapter(orchestrator, config)
    adapter.start()
    return adapter


async def run_api_server(adapter: APIAdapter = None,
                        orchestrator: ProductionOrchestrator = None,
                        config: Dict[str, Any] = None):
    """
    Run the API server.
    
    Args:
        adapter: Existing API adapter (creates new if None)
        orchestrator: Orchestrator instance
        config: API configuration
    """
    if not adapter:
        adapter = create_api_adapter(orchestrator, config)
    
    try:
        await adapter.start_server()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        adapter.stop_server()
    except Exception as e:
        print(f"API server error: {e}")
        adapter.stop_server()


# Register with adapter registry
try:
    from .base_adapter import AdapterRegistry
    AdapterRegistry.register("api", APIAdapter)
    AdapterRegistry.register("rest", APIAdapter)  # Alias
    AdapterRegistry.register("websocket", APIAdapter)  # Alias
except ImportError:
    pass


# Example usage
if __name__ == "__main__":
    """
    Example: Run API server standalone.
    
    Usage:
        python api_adapter.py [--host HOST] [--port PORT] [--debug]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Semantic Advination API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    config = {
        "host": args.host,
        "port": args.port,
        "debug": args.debug,
    }
    
    # Create and run adapter
    adapter = create_api_adapter(None, config)
    
    try:
        # Run async server
        asyncio.run(run_api_server(adapter))
    except KeyboardInterrupt:
        print("\nServer stopped")