from typing import Dict, Any, List, Optional
from app.mcp.client import get_mcp_client, MCPClientService


class MCPService:
    """MCP operations için service layer"""
    
    def __init__(self):
        self.client: Optional[MCPClientService] = None
    
    async def _get_client(self) -> MCPClientService:
        """MCP client'ı al"""
        if not self.client:
            self.client = await get_mcp_client()
        return self.client
    
    async def generate_english_question_for_user(
        self,
        user_id: str,
        error_patterns: List[str],
        difficulty_level: int,
        question_type: str = "multiple_choice",
        context: Optional[str] = None,
        topic: Optional[str] = None,
        error_focus: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Kullanıcı için İngilizce soru üret"""
        client = await self._get_client()
        
        result = await client.generate_english_question(
            user_id=user_id,
            error_patterns=error_patterns,
            difficulty_level=difficulty_level,
            question_type=question_type,
            context=context,
            topic=topic,
            focus_areas=error_focus,
        )
        
        if result.get("success"):
            return result["question"]
        else:
            raise Exception(f"Question generation failed: {result.get('error', 'Unknown error')}")
    
    async def evaluate_student_answer(
        self,
        question_content: str,
        correct_answer: str,
        student_answer: str,
        subject: str,
        question_type: str = "multiple_choice",
        difficulty_level: int = 3
    ) -> Dict[str, Any]:
        """Öğrenci cevabını değerlendir"""
        client = await self._get_client()
        
        result = await client.evaluate_answer(
            question_content=question_content,
            correct_answer=correct_answer,
            student_answer=student_answer,
            subject=subject,
            question_type=question_type,
            difficulty_level=difficulty_level
        )
        
        if result.get("success"):
            return result["evaluation"]
        else:
            raise Exception(f"Answer evaluation failed: {result.get('error', 'Unknown error')}")
    
    async def analyze_student_performance(
        self,
        user_id: str,
        subject: str,
        recent_attempts: List[Dict[str, Any]],
        analysis_type: str = "level_assessment"
    ) -> Dict[str, Any]:
        """Öğrenci performansını analiz et"""
        client = await self._get_client()
        
        result = await client.analyze_performance(
            user_id=user_id,
            subject=subject,
            recent_attempts=recent_attempts,
            analysis_type=analysis_type
        )
        
        if result.get("success"):
            return result["analysis"]
        else:
            raise Exception(f"Performance analysis failed: {result.get('error', 'Unknown error')}")
    
    async def extract_questions_from_pdf(
        self,
        pdf_path: str,
        subject: str,
        expected_difficulty: int = 3,
        extract_images: bool = True
    ) -> List[Dict[str, Any]]:
        """PDF'den soruları çıkar"""
        
        client = await self._get_client()
        
        result = await client.extract_questions_from_pdf(
            pdf_path=pdf_path,
            subject=subject,
            expected_difficulty=expected_difficulty,
            extract_images=extract_images
        )
        
        if result.get("success"):
            return result["questions"]
        else:
            raise Exception(f"PDF extraction failed: {result.get('error', 'Unknown error')}")
    
    async def generate_math_question_for_user(
        self,
        user_id: str,
        difficulty_level: int,
        topic_category: str,
        question_type: str = "multiple_choice",
        context: Optional[str] = None,
        error_patterns: Optional[List[str]] = None,
        learning_objectives: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Kullanıcı için matematik soru üret"""
        client = await self._get_client()
        
        result = await client.generate_math_question(
            user_id=user_id,
            difficulty_level=difficulty_level,
            topic_category=topic_category,
            question_type=question_type,
            context=context,
            error_patterns=error_patterns or [],
            learning_objectives=learning_objectives or [],
        )
        
        if result.get("success"):
            return result["question"]
        else:
            raise Exception(f"Math question generation failed: {result.get('error', 'Unknown error')}")
    
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
        client = await self._get_client()
        
        result = await client.evaluate_math_answer(
            question_content=question_content,
            correct_answer=correct_answer,
            student_answer=student_answer,
            question_type=question_type,
            difficulty_level=difficulty_level,
            partial_credit=partial_credit
        )
        
        if result.get("success"):
            return result["evaluation"]
        else:
            raise Exception(f"Math answer evaluation failed: {result.get('error', 'Unknown error')}")
    
    async def analyze_math_performance(
        self,
        user_id: str,
        recent_attempts: List[Dict[str, Any]],
        analysis_type: str = "skill_assessment"
    ) -> Dict[str, Any]:
        """Matematik performansını analiz et"""
        client = await self._get_client()
        
        result = await client.analyze_math_performance(
            user_id=user_id,
            recent_attempts=recent_attempts,
            analysis_type=analysis_type
        )
        
        if result.get("success"):
            return result["analysis"]
        else:
            raise Exception(f"Math performance analysis failed: {result.get('error', 'Unknown error')}")
    
    async def get_math_learning_recommendations(
        self,
        user_id: str,
        current_skill_level: float,
        weak_areas: List[str],
        learning_goals: List[str]
    ) -> Dict[str, Any]:
        """Matematik öğrenme önerileri al"""
        client = await self._get_client()
        
        result = await client.get_math_recommendations(
            user_id=user_id,
            current_skill_level=current_skill_level,
            weak_areas=weak_areas,
            learning_goals=learning_goals
        )
        
        if result.get("success"):
            return result["recommendations"]
        else:
            raise Exception(f"Math recommendations failed: {result.get('error', 'Unknown error')}")
        client = await self._get_client()
        
        result = await client.parse_pdf_questions(
            pdf_path=pdf_path,
            subject=subject,
            expected_difficulty=expected_difficulty,
            extract_images=extract_images
        )
        
        if result.get("success"):
            return result["questions"]
        else:
            raise Exception(f"PDF parsing failed: {result.get('error', 'Unknown error')}")
    
    async def get_question_templates(self, subject: str) -> Dict[str, Any]:
        """Soru şablonlarını al"""
        client = await self._get_client()
        
        if subject == "english":
            uri = "question-templates://english/grammar"
        else:
            uri = "error-patterns://math/common"
        
        return await client.get_resource(uri)
    
    async def get_learning_style_adaptations(self, learning_style: str) -> Dict[str, Any]:
        """Öğrenme stili adaptasyonlarını al"""
        client = await self._get_client()
        
        adaptations = await client.get_resource("learning-styles://adaptations")
        return adaptations.get(learning_style, adaptations.get("mixed", {}))
    
    async def read_pdf_content(
        self,
        pdf_path: str,
        page_range: Optional[Dict[str, int]] = None,
        extract_mode: str = "both",
        image_quality: str = "medium"
    ) -> Dict[str, Any]:
        """PDF içeriğini oku"""
        client = await self._get_client()
        
        result = await client.read_pdf_content(
            pdf_path=pdf_path,
            page_range=page_range,
            extract_mode=extract_mode,
            image_quality=image_quality
        )
        
        if result.get("success"):
            return result["content"]
        else:
            raise Exception(f"PDF content reading failed: {result.get('error', 'Unknown error')}")
    
    async def deliver_question_to_student(
        self,
        question_data: Dict[str, Any],
        user_id: str,
        learning_style: str = "mixed",
        delivery_format: str = "web",
        accessibility_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Soruyu öğrenciye gönder"""
        client = await self._get_client()
        
        result = await client.deliver_question_to_student(
            question_data=question_data,
            user_id=user_id,
            learning_style=learning_style,
            delivery_format=delivery_format,
            accessibility_options=accessibility_options
        )
        
        if result.get("success"):
            return {
                "delivered_question": result["delivered_question"],
                "interactive_elements": result["interactive_elements"],
                "delivery_metadata": result["delivery_metadata"]
            }
        else:
            raise Exception(f"Question delivery failed: {result.get('error', 'Unknown error')}")
    
    async def health_check(self) -> bool:
        """MCP servisinin sağlığını kontrol et"""
        try:
            client = await self._get_client()
            return await client.health_check()
        except Exception:
            return False


# Global MCP service instance
mcp_service = MCPService()