from mcp.server import Server
from mcp.types import Resource, TextResourceContents, ReadResourceResult
from typing import List, Dict, Any
import json
import asyncio

from .tools.question_generator import QuestionGeneratorTool
from .tools.answer_evaluator import AnswerEvaluatorTool
from .tools.analytics import AnalyticsTool
from .tools.pdf_parser import PDFParserTool
from .tools.pdf_content_reader import PDFContentReaderTool
from .tools.question_delivery import QuestionDeliveryTool


class AdaptiveQuestionMCPServer:
    """Adaptive Question System MCP Server"""
    
    def __init__(self):
        self.server = Server("adaptive-question-system")
        self.tools = {}
        self.resources = {}
        self._setup_tools()
        self._setup_resources()
        self._setup_handlers()
    
    def _setup_tools(self):
        """MCP tools'ları kaydet"""
        # Tool instances oluştur
        self.tools = {
            "question_generator": QuestionGeneratorTool(),
            "answer_evaluator": AnswerEvaluatorTool(),
            "analytics": AnalyticsTool(),
            "pdf_parser": PDFParserTool(),
            "pdf_content_reader": PDFContentReaderTool(),
            "question_delivery": QuestionDeliveryTool()
        }
        
        # Tools MCP handlers ile kaydedilecek
    
    def _setup_resources(self):
        """MCP resources'ları tanımla"""
        self.resources = {
            "question-templates://english/grammar": {
                "name": "English Grammar Question Templates",
                "mimeType": "application/json",
                "data": {
                    "past_tense": {
                        "multiple_choice": [
                            "I ____ to school yesterday. A) go B) went C) going D) goes",
                            "She ____ her homework last night. A) finish B) finished C) finishing D) finishes"
                        ],
                        "fill_blank": [
                            "Yesterday, I _____ (go) to the market.",
                            "Last week, we _____ (visit) our grandparents."
                        ]
                    },
                    "present_perfect": {
                        "multiple_choice": [
                            "I ____ never been to Paris. A) have B) has C) had D) having",
                            "She ____ already finished her work. A) have B) has C) had D) having"
                        ]
                    }
                }
            },
            "error-patterns://math/common": {
                "name": "Common Math Error Patterns",
                "mimeType": "application/json",
                "data": {
                    "arithmetic_errors": [
                        "addition_carry_over",
                        "subtraction_borrowing",
                        "multiplication_tables",
                        "division_remainder"
                    ],
                    "algebraic_errors": [
                        "variable_confusion",
                        "equation_solving",
                        "sign_errors",
                        "distribution_errors"
                    ],
                    "geometric_errors": [
                        "area_calculation",
                        "perimeter_confusion",
                        "angle_measurement",
                        "shape_properties"
                    ]
                }
            },
            "pdf-templates://question-formats": {
                "name": "PDF Question Format Templates",
                "mimeType": "application/json",
                "data": {
                    "multiple_choice_patterns": [
                        r"\\d+\\.\\s+.+\\?\\s*A\\)\\s+.+\\s+B\\)\\s+.+\\s+C\\)\\s+.+\\s+D\\)\\s+.+",
                        r"Question\\s+\\d+:\\s+.+\\?\\s*\\(a\\)\\s+.+\\s+\\(b\\)\\s+.+\\s+\\(c\\)\\s+.+\\s+\\(d\\)\\s+.+"
                    ],
                    "fill_blank_patterns": [
                        r".+___+.+",
                        r"Fill\\s+in\\s+the\\s+blank:.+",
                        r"Complete\\s+the\\s+sentence:.+"
                    ],
                    "open_ended_patterns": [
                        r"Write\\s+.+",
                        r"Describe\\s+.+",
                        r"Explain\\s+.+"
                    ]
                }
            },
            "learning-styles://adaptations": {
                "name": "Learning Style Adaptation Rules",
                "mimeType": "application/json",
                "data": {
                    "visual": {
                        "preferences": ["diagrams", "charts", "colors", "highlighting"],
                        "question_adaptations": {
                            "include_diagrams": True,
                            "use_colors": True,
                            "highlight_keywords": True,
                            "visual_examples": True
                        }
                    },
                    "auditory": {
                        "preferences": ["audio", "verbal_explanations", "discussions"],
                        "question_adaptations": {
                            "include_audio": True,
                            "text_to_speech": True,
                            "verbal_instructions": True,
                            "sound_feedback": True
                        }
                    },
                    "kinesthetic": {
                        "preferences": ["hands_on", "interactive", "movement"],
                        "question_adaptations": {
                            "interactive_elements": True,
                            "step_by_step": True,
                            "drag_drop": True,
                            "physical_manipulation": True
                        }
                    },
                    "mixed": {
                        "preferences": ["balanced_approach", "multiple_formats"],
                        "question_adaptations": {
                            "multiple_formats": True,
                            "flexible_presentation": True,
                            "choice_of_interaction": True
                        }
                    }
                }
            }
        }
    
    def _setup_handlers(self):
        """MCP handlers'ları kur"""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """Mevcut resources'ları listele"""
            resources = []
            for uri, resource_info in self.resources.items():
                resources.append(Resource(
                    uri=uri,
                    name=resource_info["name"],
                    mimeType=resource_info["mimeType"]
                ))
            return resources
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Resource içeriğini oku"""
            if uri in self.resources:
                resource_info = self.resources[uri]
                return ReadResourceResult(
                    contents=[
                        TextResourceContents(
                            uri=uri,
                            mimeType=resource_info["mimeType"],
                            text=json.dumps(resource_info["data"], ensure_ascii=False, indent=2)
                        )
                    ]
                )
            else:
                raise ValueError(f"Resource not found: {uri}")
        
        @self.server.list_tools()
        async def list_tools():
            """Mevcut tools'ları listele"""
            return [tool_instance.to_tool_definition() for tool_instance in self.tools.values()]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            """Tool çağrısını işle"""
            # Tool name'i tool instance'a map et
            tool_mapping = {
                "generate_english_question": "question_generator",
                "evaluate_answer": "answer_evaluator", 
                "analyze_performance": "analytics",
                "parse_pdf_questions": "pdf_parser",
                "read_pdf_content": "pdf_content_reader",
                "deliver_question_to_student": "question_delivery"
            }
            
            if name in tool_mapping:
                tool_key = tool_mapping[name]
                tool_instance = self.tools[tool_key]
                return await tool_instance.call(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def start(self, transport_type: str = "stdio"):
        """MCP server'ı başlat"""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            await stdio_server(self.server)
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
    
    def get_server(self) -> Server:
        """Server instance'ını döndür"""
        return self.server


# Global server instance
mcp_server = AdaptiveQuestionMCPServer()


async def start_mcp_server():
    """MCP server'ı başlat"""
    await mcp_server.start()


if __name__ == "__main__":
    # Standalone MCP server olarak çalıştır
    asyncio.run(start_mcp_server())