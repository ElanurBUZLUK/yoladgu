import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


@dataclass
class MCPMessage:
    """MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPClientV2:
    """Real JSON-RPC compliant MCP Client"""
    
    def __init__(self, server_command: Optional[List[str]] = None):
        self.server_command = server_command or ["python", "-m", "app.mcp.server"]
        self.process = None
        self.is_connected = False
        self.request_id_counter = 0
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            # Start MCP server process
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Initialize connection
            init_result = await self._send_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "adaptive-learning-backend",
                        "version": "1.0.0"
                    }
                }
            )
            
            if init_result and not init_result.get("error"):
                self.is_connected = True
                logger.info("✅ MCP Client connected successfully")
                
                # Start message listener
                asyncio.create_task(self._message_listener())
                return True
            else:
                logger.error(f"❌ MCP initialization failed: {init_result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MCP connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.process:
            await self._send_notification(method="exit")
            self.process.terminate()
            await self.process.wait()
            self.process = None
        
        self.is_connected = False
        logger.info("❌ MCP Client disconnected")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        result = await self._send_request(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        # JSON-RPC response validation
        if not isinstance(result, dict):
            raise Exception("Invalid JSON-RPC response format")
        
        if "error" in result:
            error = result["error"]
            raise Exception(f"Tool call failed: {error.get('message', 'Unknown error')} (code: {error.get('code', -1)})")
        
        if "result" not in result:
            raise Exception("JSON-RPC response missing 'result' field")
        
        return result["result"]
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        result = await self._send_request(
            method="tools/list"
        )
        
        # JSON-RPC response validation
        if not isinstance(result, dict):
            raise Exception("Invalid JSON-RPC response format")
        
        if "error" in result:
            error = result["error"]
            raise Exception(f"List tools failed: {error.get('message', 'Unknown error')} (code: {error.get('code', -1)})")
        
        if "result" not in result:
            raise Exception("JSON-RPC response missing 'result' field")
        
        tools_data = result["result"]
        if not isinstance(tools_data, dict) or "tools" not in tools_data:
            raise Exception("Invalid tools response format")
        
        return tools_data["tools"]
    
    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC request"""
        request_id = str(self.request_id_counter)
        self.request_id_counter += 1
        
        message = MCPMessage(
            id=request_id,
            method=method,
            params=params or {}
        )
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send message
        await self._send_message(message)
        
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise Exception(f"Request timeout: {method}")
    
    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send JSON-RPC notification"""
        message = MCPMessage(
            method=method,
            params=params or {}
        )
        await self._send_message(message)
    
    async def _send_message(self, message: MCPMessage):
        """Send message to MCP server"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not running")
        
        message_json = json.dumps(message.__dict__, ensure_ascii=False)
        self.process.stdin.write(f"{message_json}\n".encode())
        await self.process.stdin.drain()
    
    async def _message_listener(self):
        """Listen for messages from MCP server"""
        if not self.process or not self.process.stdout:
            return
        
        while self.is_connected and self.process:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                message_data = json.loads(line.decode().strip())
                await self._handle_message(message_data)
                
            except Exception as e:
                logger.error(f"Error reading MCP message: {e}")
                break
    
    async def _handle_message(self, message_data: Dict[str, Any]):
        """Handle incoming MCP message"""
        message_type = message_data.get("jsonrpc")
        message_id = message_data.get("id")
        
        if message_type == "2.0" and message_id:
            # Response to request
            future = self.pending_requests.pop(message_id, None)
            if future:
                if "error" in message_data:
                    future.set_exception(Exception(message_data["error"]))
                else:
                    future.set_result(message_data)
            else:
                logger.warning(f"Unknown request ID: {message_id}")
        
        elif "method" in message_data:
            # Notification or request from server
            method = message_data["method"]
            if method == "notifications/log":
                # Handle log notifications
                params = message_data.get("params", {})
                level = params.get("level", "info")
                message = params.get("message", "")
                logger.log(getattr(logging, level.upper(), logging.INFO), f"MCP: {message}")


# Global MCP client instance
mcp_client_v2 = MCPClientV2()
