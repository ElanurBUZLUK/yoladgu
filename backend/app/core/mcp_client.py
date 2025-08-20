import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import websockets
from websockets.exceptions import WebSocketException

logger = logging.getLogger(__name__)


class MCPTransportType(Enum):
    """MCP transport types"""
    HTTP = "http"
    WEBSOCKET = "websocket"
    STDIO = "stdio"


@dataclass
class MCPRequest:
    """MCP JSON-RPC request"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC response"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPClient:
    """Real MCP Client with HTTP/WebSocket/STDIO support"""
    
    def __init__(
        self,
        transport_type: MCPTransportType = MCPTransportType.HTTP,
        server_url: Optional[str] = None,
        server_command: Optional[List[str]] = None,
        timeout: int = 30
    ):
        self.transport_type = transport_type
        self.server_url = server_url or "http://localhost:8000/mcp"
        self.server_command = server_command or ["python", "-m", "app.mcp.server"]
        self.timeout = timeout
        
        # Connection state
        self.is_connected = False
        self.request_id_counter = 0
        self.pending_requests: Dict[Union[str, int], asyncio.Future] = {}
        
        # Transport clients
        self.http_client: Optional[httpx.AsyncClient] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.process: Optional[asyncio.subprocess.Process] = None
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            if self.transport_type == MCPTransportType.HTTP:
                return await self._connect_http()
            elif self.transport_type == MCPTransportType.WEBSOCKET:
                return await self._connect_websocket()
            elif self.transport_type == MCPTransportType.STDIO:
                return await self._connect_stdio()
            else:
                raise ValueError(f"Unsupported transport type: {self.transport_type}")
                
        except Exception as e:
            logger.error(f"MCP connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        try:
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None
            
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            if self.process:
                self.process.terminate()
                await self.process.wait()
                self.process = None
            
            self.is_connected = False
            logger.info("MCP Client disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with observability"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            request = MCPRequest(
                id=self._get_next_id(),
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                }
            )
            
            response = await self._send_request(request)
            
            if response.error:
                error_type = "mcp_error"
                raise Exception(f"Tool call failed: {response.error.get('message', 'Unknown error')} (code: {response.error.get('code', -1)})")
            
            success = True
            result = response.result or {}
            
            # Record usage metrics
            from app.core.mcp_observability import mcp_observability
            mcp_observability.record_usage(
                tool_name=tool_name,
                user_id=arguments.get("user_id"),
                session_id=arguments.get("session_id"),
                arguments_size=len(str(arguments)),
                response_size=len(str(result))
            )
            
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            # Record latency metrics
            latency_ms = (time.time() - start_time) * 1000
            from app.core.mcp_observability import mcp_observability
            mcp_observability.record_latency(
                tool_name=tool_name,
                latency_ms=latency_ms,
                success=success,
                error_type=error_type
            )
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        request = MCPRequest(
            id=self._get_next_id(),
            method="tools/list"
        )
        
        response = await self._send_request(request)
        
        if response.error:
            raise Exception(f"List tools failed: {response.error.get('message', 'Unknown error')} (code: {response.error.get('code', -1)})")
        
        tools_data = response.result or {}
        return tools_data.get("tools", [])
    
    async def _connect_http(self) -> bool:
        """Connect via HTTP"""
        try:
            self.http_client = httpx.AsyncClient(
                base_url=self.server_url,
                timeout=self.timeout
            )
            
            # Test connection
            response = await self.http_client.get("/health")
            if response.status_code == 200:
                self.is_connected = True
                logger.info("✅ MCP Client connected via HTTP")
                return True
            else:
                logger.error(f"HTTP connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"HTTP connection error: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect via WebSocket"""
        try:
            ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
            self.websocket = await websockets.connect(ws_url)
            
            # Test connection
            await self.websocket.ping()
            self.is_connected = True
            logger.info("✅ MCP Client connected via WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False
    
    async def _connect_stdio(self) -> bool:
        """Connect via STDIO"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Initialize connection
            init_request = MCPRequest(
                id=self._get_next_id(),
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": "adaptive-learning-backend",
                        "version": "1.0.0"
                    }
                }
            )
            
            response = await self._send_request(init_request)
            
            if response.error:
                logger.error(f"STDIO initialization failed: {response.error}")
                return False
            
            self.is_connected = True
            logger.info("✅ MCP Client connected via STDIO")
            
            # Start message listener
            asyncio.create_task(self._stdio_listener())
            return True
            
        except Exception as e:
            logger.error(f"STDIO connection error: {e}")
            return False
    
    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request and wait for response"""
        if self.transport_type == MCPTransportType.HTTP:
            return await self._send_http_request(request)
        elif self.transport_type == MCPTransportType.WEBSOCKET:
            return await self._send_websocket_request(request)
        elif self.transport_type == MCPTransportType.STDIO:
            return await self._send_stdio_request(request)
        else:
            raise ValueError(f"Unsupported transport type: {self.transport_type}")
    
    async def _send_http_request(self, request: MCPRequest) -> MCPResponse:
        """Send HTTP request"""
        if not self.http_client:
            raise RuntimeError("HTTP client not connected")
        
        try:
            response = await self.http_client.post(
                "/jsonrpc",
                json=asdict(request),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP error: {response.status_code}")
            
            data = response.json()
            return MCPResponse(**data)
            
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": f"HTTP request failed: {str(e)}"
                }
            )
    
    async def _send_websocket_request(self, request: MCPRequest) -> MCPResponse:
        """Send WebSocket request"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            await self.websocket.send(json.dumps(asdict(request)))
            response_data = await self.websocket.recv()
            data = json.loads(response_data)
            return MCPResponse(**data)
            
        except Exception as e:
            logger.error(f"WebSocket request failed: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": f"WebSocket request failed: {str(e)}"
                }
            )
    
    async def _send_stdio_request(self, request: MCPRequest) -> MCPResponse:
        """Send STDIO request"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("STDIO process not connected")
        
        try:
            # Create future for response
            future = asyncio.Future()
            self.pending_requests[request.id] = future
            
            # Send request
            request_json = json.dumps(asdict(request), ensure_ascii=False)
            self.process.stdin.write(f"{request_json}\n".encode())
            await self.process.stdin.drain()
            
            # Wait for response
            response_data = await asyncio.wait_for(future, timeout=self.timeout)
            return MCPResponse(**response_data)
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(request.id, None)
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": "Request timeout"
                }
            )
        except Exception as e:
            logger.error(f"STDIO request failed: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": f"STDIO request failed: {str(e)}"
                }
            )
    
    async def _stdio_listener(self):
        """Listen for STDIO messages"""
        if not self.process or not self.process.stdout:
            return
        
        try:
            while self.is_connected and self.process:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    message_data = json.loads(line.decode().strip())
                    await self._handle_stdio_message(message_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from STDIO: {e}")
                except Exception as e:
                    logger.error(f"Error handling STDIO message: {e}")
                    
        except Exception as e:
            logger.error(f"STDIO listener error: {e}")
    
    async def _handle_stdio_message(self, message_data: Dict[str, Any]):
        """Handle STDIO message"""
        message_id = message_data.get("id")
        
        if message_id and message_id in self.pending_requests:
            future = self.pending_requests.pop(message_id)
            if not future.done():
                future.set_result(message_data)
        else:
            # Handle notifications
            method = message_data.get("method")
            if method == "notifications/log":
                params = message_data.get("params", {})
                level = params.get("level", "info")
                message = params.get("message", "")
                logger.log(getattr(logging, level.upper(), logging.INFO), f"MCP: {message}")
    
    def _get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id_counter += 1
        return self.request_id_counter


# Global MCP client instance
mcp_client = MCPClient(transport_type=MCPTransportType.HTTP)
