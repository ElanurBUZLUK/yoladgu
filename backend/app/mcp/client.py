from typing import Dict, Any, List, Optional
import json
import asyncio


class MCPClientService:
    """MCP Client Service - Simplified version for direct tool usage"""
    
    def __init__(self):
        self.is_connected = False
        # Import tools directly
        from .tools.question_generator import QuestionGeneratorTool
        from .tools.answer_evaluator import AnswerEvaluatorTool
        from .tools.analytics import AnalyticsTool
        from .tools.pdf_parser import PDFParserTool
        from .tools.pdf_content_reader import PDFContentReaderTool
        from .tools.question_delivery import QuestionDeliveryTool
        from .tools.math_generator import MathGeneratorTool
        from .tools.math_evaluator import MathEvaluatorTool
        
        self.tools = {
            "question_generator": QuestionGeneratorTool(),
            "answer_evaluator": AnswerEvaluatorTool(),
            "analytics": AnalyticsTool(),
            "pdf_parser": PDFParserTool(),
            "pdf_content_reader": PDFContentReaderTool(),
            "question_delivery": QuestionDeliveryTool(),
            "math_generator": MathGeneratorTool(),
            "math_evaluator": MathEvaluatorTool()
        }
    
    async def connect(self, server_command: Optional[List[str]] = None):
        """MCP connection simulation"""
        self.is_connected = True
        print("✅ MCP Client connected successfully (direct mode)")
    
    async def disconnect(self):
        """MCP disconnection simulation"""
        self.is_connected = False
        print("❌ MCP Client disconnected")
    
    async def generate_english_question(
        self, 
        user_id: str, 
        error_patterns: List[str], 
        difficulty_level: int,
        question_type: str = "multiple_choice",
        focus_areas: Optional[List[str]] = None,
        topic: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """İngilizce soru üret"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "user_id": user_id,
            "error_patterns": error_patterns,
            "difficulty_level": difficulty_level,
            "question_type": question_type
        }
        
        if focus_areas:
            arguments["focus_areas"] = focus_areas
        if topic:
            arguments["topic"] = topic
        if context:
            arguments["context"] = context
        
        try:
            tool = self.tools["question_generator"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error generating English question: {e}")
            raise
    
    async def evaluate_answer(
        self,
        question_content: str,
        correct_answer: str,
        student_answer: str,
        subject: str,
        question_type: str = "multiple_choice",
        difficulty_level: int = 3
    ) -> Dict[str, Any]:
        """Cevabı değerlendir"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "question_content": question_content,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "subject": subject,
            "question_type": question_type,
            "difficulty_level": difficulty_level
        }
        
        try:
            tool = self.tools["answer_evaluator"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            raise
    
    async def generate_math_question(
        self, 
        user_id: str, 
        difficulty_level: int,
        topic_category: str,
        question_type: str = "multiple_choice",
        context: Optional[str] = None,
        error_patterns: Optional[List[str]] = None,
        learning_objectives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Matematik sorusu üret"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "user_id": user_id,
            "difficulty_level": difficulty_level,
            "topic_category": topic_category,
            "question_type": question_type
        }
        
        if context:
            arguments["context"] = context
        if error_patterns:
            arguments["error_patterns"] = error_patterns
        if learning_objectives:
            arguments["learning_objectives"] = learning_objectives
        
        try:
            tool = self.tools["math_generator"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error generating math question: {e}")
            raise
    
    async def evaluate_math_answer(
        self,
        question_content: str,
        correct_answer: str,
        student_answer: str,
        question_type: str = "multiple_choice",
        difficulty_level: int = 3,
        partial_credit: bool = True
    ) -> Dict[str, Any]:
        """Matematik cevabını değerlendir"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "question_content": question_content,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "question_type": question_type,
            "difficulty_level": difficulty_level,
            "partial_credit": partial_credit
        }
        
        try:
            tool = self.tools["math_evaluator"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error evaluating math answer: {e}")
            raise
    
    async def analyze_math_performance(
        self,
        user_id: str,
        recent_attempts: List[Dict[str, Any]],
        analysis_type: str = "skill_assessment"
    ) -> Dict[str, Any]:
        """Matematik performansını analiz et"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "user_id": user_id,
            "recent_attempts": recent_attempts,
            "analysis_type": analysis_type
        }
        
        try:
            tool = self.tools["analytics"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error analyzing math performance: {e}")
            raise
    
    async def get_math_recommendations(
        self,
        user_id: str,
        current_skill_level: float,
        weak_areas: List[str],
        learning_goals: List[str]
    ) -> Dict[str, Any]:
        """Matematik önerileri al"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "user_id": user_id,
            "current_skill_level": current_skill_level,
            "weak_areas": weak_areas,
            "learning_goals": learning_goals
        }
        
        try:
            tool = self.tools["analytics"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error getting math recommendations: {e}")
            raise
            print(f"Error evaluating answer: {e}")
            raise
    
    async def analyze_performance(
        self,
        user_id: str,
        subject: str,
        recent_attempts: List[Dict[str, Any]],
        analysis_type: str = "level_assessment",
        time_period: str = "last_month"
    ) -> Dict[str, Any]:
        """Performans analizi yap"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "user_id": user_id,
            "subject": subject,
            "recent_attempts": recent_attempts,
            "analysis_type": analysis_type,
            "time_period": time_period
        }
        
        try:
            tool = self.tools["analytics"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            raise
    
    async def parse_pdf_questions(
        self,
        pdf_path: str,
        subject: str,
        expected_difficulty: int = 3,
        extract_images: bool = True,
        preserve_formatting: bool = True,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """PDF'den soru çıkar"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "pdf_path": pdf_path,
            "subject": subject,
            "expected_difficulty": expected_difficulty,
            "extract_images": extract_images,
            "preserve_formatting": preserve_formatting
        }
        
        if question_types:
            arguments["question_types"] = question_types
        
        try:
            tool = self.tools["pdf_parser"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error parsing PDF questions: {e}")
            raise
    
    async def get_resource(self, uri: str) -> Dict[str, Any]:
        """Resource içeriğini al - Mock implementation"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        # Mock resources
        resources = {
            "question-templates://english/grammar": {
                "past_tense": ["I ____ to school yesterday."],
                "present_perfect": ["I ____ never been to Paris."]
            },
            "error-patterns://math/common": {
                "arithmetic_errors": ["addition", "subtraction"],
                "algebraic_errors": ["variables", "equations"]
            }
        }
        
        return resources.get(uri, {})
    
    async def list_available_resources(self) -> List[Dict[str, str]]:
        """Mevcut resources'ları listele - Mock implementation"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        return [
            {
                "uri": "question-templates://english/grammar",
                "name": "English Grammar Templates",
                "mimeType": "application/json"
            },
            {
                "uri": "error-patterns://math/common",
                "name": "Math Error Patterns",
                "mimeType": "application/json"
            }
        ]
    
    async def read_pdf_content(
        self,
        pdf_path: str,
        page_range: Optional[Dict[str, int]] = None,
        extract_mode: str = "both",
        image_quality: str = "medium"
    ) -> Dict[str, Any]:
        """PDF içeriğini oku"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "pdf_path": pdf_path,
            "extract_mode": extract_mode,
            "image_quality": image_quality
        }
        
        if page_range:
            arguments["page_range"] = page_range
        
        try:
            tool = self.tools["pdf_content_reader"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error reading PDF content: {e}")
            raise
    
    async def deliver_question_to_student(
        self,
        question_data: Dict[str, Any],
        user_id: str,
        learning_style: str = "mixed",
        delivery_format: str = "web",
        accessibility_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Soruyu öğrenciye gönder"""
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        arguments = {
            "question_data": question_data,
            "user_id": user_id,
            "learning_style": learning_style,
            "delivery_format": delivery_format
        }
        
        if accessibility_options:
            arguments["accessibility_options"] = accessibility_options
        
        try:
            tool = self.tools["question_delivery"]
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            print(f"Error delivering question: {e}")
            raise
    
    async def health_check(self) -> bool:
        """MCP bağlantısının sağlığını kontrol et"""
        return self.is_connected


# Global MCP client instance
mcp_client = MCPClientService()


async def get_mcp_client() -> MCPClientService:
    """MCP client instance'ını al"""
    if not mcp_client.is_connected:
        await mcp_client.connect()
    return mcp_client