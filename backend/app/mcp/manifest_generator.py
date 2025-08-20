import json
import inspect
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.mcp.schema_validator import mcp_schema_validator

logger = logging.getLogger(__name__)


class MCPManifestGenerator:
    """MCP Tool Manifest Generator"""
    
    def __init__(self, server_module_path: str = "app.mcp.server"):
        self.server_module_path = server_module_path
        self.manifest_path = Path(__file__).parent / "mcp.json"
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate MCP manifest from server tools"""
        try:
            # Import server module
            import importlib
            server_module = importlib.import_module(self.server_module_path)
            
            tools = []
            
            # Find all @server.tool decorated functions
            for name, obj in inspect.getmembers(server_module):
                if inspect.isfunction(obj) and hasattr(obj, '_mcp_tool'):
                    tool_info = self._extract_tool_info(obj)
                    if tool_info:
                        tools.append(tool_info)
            
            manifest = {
                "mcpServers": {
                    "education-mcp": {
                        "command": "python",
                        "args": ["-m", "app.mcp.server"],
                        "env": {},
                        "description": "Educational MCP Server for Adaptive Learning System"
                    }
                },
                "tools": tools
            }
            
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to generate manifest: {e}")
            return self._get_default_manifest()
    
    def _extract_tool_info(self, func) -> Optional[Dict[str, Any]]:
        """Extract tool information from function with Pydantic validation"""
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Extract parameters
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = self._get_param_type(param)
                param_desc = self._get_param_description(func, param_name)
                
                # Enhanced property definition with validation
                property_def = {
                    "type": param_type,
                    "description": param_desc
                }
                
                # Add validation constraints based on type
                if param_type == "integer":
                    property_def["minimum"] = 0
                elif param_type == "number":
                    property_def["minimum"] = 0.0
                    property_def["maximum"] = 1.0
                elif param_type == "string":
                    property_def["minLength"] = 1
                    property_def["maxLength"] = 10000
                
                # Add default value if exists
                if param.default != inspect.Parameter.empty:
                    property_def["default"] = param.default
                else:
                    required.append(param_name)
                
                properties[param_name] = property_def
            
            # Create dynamic Pydantic model for validation
            schema_name = f"{func.__name__}_schema"
            try:
                dynamic_model = mcp_schema_validator.create_dynamic_schema(
                    schema_name, properties
                )
                logger.info(f"Created dynamic schema for {func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to create dynamic schema for {func.__name__}: {e}")
            
            tool_info = {
                "name": func.__name__,
                "description": func.__doc__ or f"Tool: {func.__name__}",
                "args_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
            
            return tool_info
            
        except Exception as e:
            logger.error(f"Failed to extract tool info for {func.__name__}: {e}")
            return None
    
    def _get_param_type(self, param) -> str:
        """Get parameter type"""
        if param.annotation == inspect.Parameter.empty:
            return "string"  # Default type
        
        type_name = str(param.annotation)
        
        # Map Python types to JSON schema types
        type_mapping = {
            "str": "string",
            "int": "integer", 
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object"
        }
        
        for py_type, json_type in type_mapping.items():
            if py_type in type_name:
                return json_type
        
        return "string"  # Default fallback
    
    def _get_param_description(self, func, param_name: str) -> str:
        """Get parameter description from docstring"""
        if not func.__doc__:
            return f"Parameter: {param_name}"
        
        # Simple docstring parsing
        doc_lines = func.__doc__.split('\n')
        for line in doc_lines:
            if param_name in line and ':' in line:
                return line.split(':', 1)[1].strip()
        
        return f"Parameter: {param_name}"
    
    def _get_default_manifest(self) -> Dict[str, Any]:
        """Get default manifest when generation fails"""
        return {
            "mcpServers": {
                "education-mcp": {
                    "command": "python",
                    "args": ["-m", "app.mcp.server"],
                    "env": {},
                    "description": "Educational MCP Server for Adaptive Learning System"
                }
            },
            "tools": [
                {
                    "name": "recommend_math",
                    "description": "Suggest math questions based on ELO & neighbor wrongs.",
                    "args_schema": {
                        "type": "object",
                        "properties": {
                            "student_id": {
                                "type": "string",
                                "description": "The ID of the student."
                            }
                        },
                        "required": ["student_id"]
                    }
                },
                {
                    "name": "generate_english_cloze",
                    "description": "Generate English cloze questions from student error logs.",
                    "args_schema": {
                        "type": "object",
                        "properties": {
                            "student_id": {
                                "type": "string",
                                "description": "The ID of the student."
                            },
                            "num_recent_errors": {
                                "type": "integer",
                                "description": "Number of recent errors to consider for cloze generation.",
                                "default": 5
                            }
                        },
                        "required": ["student_id"]
                    }
                },
                {
                    "name": "assess_cefr",
                    "description": "Estimate CEFR level with rubric-based JSON.",
                    "args_schema": {
                        "type": "object",
                        "properties": {
                            "student_id": {
                                "type": "string",
                                "description": "The ID of the student."
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to assess for CEFR level."
                            }
                        },
                        "required": ["student_id", "text"]
                    }
                },
                {
                    "name": "llm_generate",
                    "description": "Generate text or JSON using LLM models.",
                    "args_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt to send to the LLM"
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "System prompt for the LLM",
                                "default": ""
                            },
                            "output_type": {
                                "type": "string",
                                "enum": ["text", "json"],
                                "description": "Type of output to generate",
                                "default": "text"
                            },
                            "schema": {
                                "type": "object",
                                "description": "JSON schema for structured output"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for the LLM"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for generation",
                                "default": 0.7
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                                "default": 1000
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            ]
        }
    
    def save_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Save manifest to file"""
        try:
            with open(self.manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Manifest saved to {self.manifest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            return False
    
    def update_manifest(self) -> bool:
        """Update manifest file"""
        manifest = self.generate_manifest()
        return self.save_manifest(manifest)


# Global manifest generator
manifest_generator = MCPManifestGenerator()
