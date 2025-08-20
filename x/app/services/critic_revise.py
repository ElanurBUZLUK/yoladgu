from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import re
from datetime import datetime

from app.services.llm_gateway import llm_gateway
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class CriticReviseService:
    """Critic & Revise service for 2-stage question generation"""
    
    def __init__(self):
        self.cache_ttl = 1800  # 30 minutes
        self.max_revision_attempts = 2
        
        # Advanced critique settings
        self.quality_threshold = 0.7
        self.cefr_compliance_weight = 0.3
        self.error_focus_weight = 0.25
        self.content_quality_weight = 0.25
        self.educational_value_weight = 0.2
        
    async def critic_and_revise(
        self,
        draft_question: Dict[str, Any],
        target_cefr: str,
        error_focus: List[str],
        context: str,
        question_format: str = "mcq"
    ) -> Dict[str, Any]:
        """Perform critic and revise cycle on draft question"""
        
        try:
            # Step 1: Critic the draft
            critique = await self._critic_question(
                draft_question, target_cefr, error_focus, context, question_format
            )
            
            # Step 2: Check if revision is needed
            if critique["needs_revision"]:
                logger.info(f"Question needs revision: {critique['issues']}")
                
                # Step 3: Revise the question
                revised_question = await self._revise_question(
                    draft_question, critique, target_cefr, error_focus, context, question_format
                )
                
                return {
                    "final_question": revised_question,
                    "critique": critique,
                    "was_revised": True,
                    "revision_count": 1
                }
            else:
                logger.info("Question passed critique without revision")
                return {
                    "final_question": draft_question,
                    "critique": critique,
                    "was_revised": False,
                    "revision_count": 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error in critic and revise: {e}")
            return {
                "final_question": draft_question,
                "critique": {"needs_revision": False, "issues": [], "score": 0.0},
                "was_revised": False,
                "revision_count": 0,
                "error": str(e)
            }
    
    async def _critic_question(
        self,
        question: Dict[str, Any],
        target_cefr: str,
        error_focus: List[str],
        context: str,
        question_format: str
    ) -> Dict[str, Any]:
        """Critic the generated question using multi-stage analysis"""
        
        cache_key = f"critic:{hash(json.dumps(question, sort_keys=True))}:{target_cefr}:{question_format}"
        
        # Try to get from cache first
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Stage 1: Schema validation
            schema_critique = self._validate_question_schema(question, question_format)
            
            # Stage 2: CEFR compliance check
            cefr_critique = self._validate_cefr_compliance(question, target_cefr)
            
            # Stage 3: Error focus alignment
            error_focus_critique = self._validate_error_focus_alignment(question, error_focus)
            
            # Stage 4: Content quality assessment
            content_quality_critique = self._validate_content_quality(question, context)
            
            # Stage 5: Educational value assessment
            educational_value_critique = self._validate_educational_value(question, target_cefr)
            
            # Stage 6: LLM-based critique (if available)
            llm_critique = await self._get_llm_critique(
                question, target_cefr, error_focus, context, question_format
            )
            
            # Combine all critiques
            combined_critique = self._combine_critiques([
                schema_critique,
                cefr_critique,
                error_focus_critique,
                content_quality_critique,
                educational_value_critique,
                llm_critique
            ])
            
            # Cache the result
            await cache_service.set(cache_key, combined_critique, self.cache_ttl)
            
            return combined_critique
                
        except Exception as e:
            logger.error(f"❌ Error in question critique: {e}")
            return self._fallback_critique(question, target_cefr)
    
    async def _revise_question(
        self,
        original_question: Dict[str, Any],
        critique: Dict[str, Any],
        target_cefr: str,
        error_focus: List[str],
        context: str,
        question_format: str
    ) -> Dict[str, Any]:
        """Revise the question based on critique"""
        
        try:
            # Build revision prompt
            revision_prompt = self._build_revision_prompt(
                original_question, critique, target_cefr, error_focus, context, question_format
            )
            
            # Get revised question from MCP
            from app.core.mcp_utils import mcp_utils
            
            try:
                if mcp_utils.is_initialized:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="llm_generate",
                        arguments={
                            "prompt": revision_prompt,
                            "system_prompt": self._get_revision_system_prompt(),
                            "output_type": "json",
                            "schema": self._get_question_schema(question_format),
                            "temperature": 0.3,
                            "max_tokens": 1000,
                            "context": context,
                            "user_id": "critic_revise",
                            "session_id": f"revision_{target_cefr}"
                        }
                    )
                    
                    if mcp_response["success"]:
                        revision_result = mcp_response["data"]
                        if isinstance(revision_result, str):
                            import json
                            revision_result = json.loads(revision_result)
                    else:
                        logger.warning(f"MCP revision failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                        revision_result = await llm_gateway.generate_structured_with_fallback(
                            task_type="question_revision",
                            prompt=revision_prompt,
                            schema=self._get_question_schema(question_format),
                            system_prompt=self._get_revision_system_prompt(),
                            complexity="medium"
                        )
                else:
                    logger.warning("MCP not initialized, using direct LLM")
                    # Fallback to direct LLM
                    revision_result = await llm_gateway.generate_structured_with_fallback(
                        task_type="question_revision",
                        prompt=revision_prompt,
                        schema=self._get_question_schema(question_format),
                        system_prompt=self._get_revision_system_prompt(),
                        complexity="medium"
                    )
            except Exception as e:
                logger.warning(f"MCP revision failed, using fallback: {e}")
                # Fallback to direct LLM
                revision_result = await llm_gateway.generate_structured_with_fallback(
                    task_type="question_revision",
                    prompt=revision_prompt,
                    schema=self._get_question_schema(question_format),
                    system_prompt=self._get_revision_system_prompt(),
                    complexity="medium"
                )
            
            if revision_result.get("success"):
                revised_question = revision_result.get("data", {})
                
                # Validate revised question
                validated_question = self._validate_question(revised_question, question_format)
                
                return validated_question
            else:
                logger.warning("LLM revision failed, returning original")
                return original_question
                
        except Exception as e:
            logger.error(f"❌ Error in question revision: {e}")
            return original_question
    
    def _build_critic_prompt(
        self,
        question: Dict[str, Any],
        target_cefr: str,
        error_focus: List[str],
        context: str,
        question_format: str
    ) -> str:
        """Build prompt for question critique"""
        
        prompt = f"""
Please critique the following English question for educational quality and appropriateness.

**Target CEFR Level:** {target_cefr}
**Error Focus Areas:** {', '.join(error_focus) if error_focus else 'General'}
**Question Format:** {question_format.upper()}
**Context:** {context[:200]}...

**Question to Critique:**
{json.dumps(question, indent=2)}

**Evaluation Criteria:**
1. Schema validity and completeness
2. CEFR level appropriateness
3. Error focus alignment
4. Content quality and clarity
5. Distractor plausibility (for MCQ)
6. Grammar and language accuracy
7. Educational value

Please provide a detailed critique following the specified JSON schema.
"""
        
        return prompt
    
    def _build_revision_prompt(
        self,
        original_question: Dict[str, Any],
        critique: Dict[str, Any],
        target_cefr: str,
        error_focus: List[str],
        context: str,
        question_format: str
    ) -> str:
        """Build prompt for question revision"""
        
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])
        
        prompt = f"""
Please revise the following English question based on the provided critique.

**Target CEFR Level:** {target_cefr}
**Error Focus Areas:** {', '.join(error_focus) if error_focus else 'General'}
**Question Format:** {question_format.upper()}
**Context:** {context[:200]}...

**Original Question:**
{json.dumps(original_question, indent=2)}

**Critique Issues:**
{chr(10).join(f"- {issue}" for issue in issues)}

**Revision Suggestions:**
{chr(10).join(f"- {suggestion}" for suggestion in suggestions)}

**Instructions:**
- Fix all identified issues
- Maintain the same question format
- Ensure CEFR level appropriateness
- Focus on the specified error areas
- Improve educational value
- Keep the question engaging and clear

Please provide the revised question following the specified JSON schema.
"""
        
        return prompt
    
    def _get_critic_schema(self) -> Dict[str, Any]:
        """Get JSON schema for critique response"""
        
        return {
            "type": "object",
            "properties": {
                "needs_revision": {"type": "boolean"},
                "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "schema_valid": {"type": "boolean"},
                "cefr_appropriate": {"type": "boolean"},
                "error_focus_aligned": {"type": "boolean"},
                "content_quality": {"type": "number", "minimum": 0, "maximum": 1},
                "distractor_quality": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["needs_revision", "overall_score", "issues", "suggestions"]
        }
    
    def _get_question_schema(self, question_format: str) -> Dict[str, Any]:
        """Get JSON schema for question response"""
        
        if question_format == "mcq":
            return {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "correct_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "difficulty_level": {"type": "integer", "minimum": 1, "maximum": 5},
                    "topic": {"type": "string"},
                    "cefr_level": {"type": "string"}
                },
                "required": ["content", "options", "correct_answer", "explanation"]
            }
        elif question_format == "cloze":
            return {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "blanks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer"},
                                "correct_answer": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "explanation": {"type": "string"},
                    "difficulty_level": {"type": "integer", "minimum": 1, "maximum": 5},
                    "topic": {"type": "string"},
                    "cefr_level": {"type": "string"}
                },
                "required": ["content", "blanks", "explanation"]
            }
        else:
            # Default schema
            return {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "difficulty_level": {"type": "integer", "minimum": 1, "maximum": 5},
                    "topic": {"type": "string"},
                    "cefr_level": {"type": "string"}
                },
                "required": ["content", "correct_answer", "explanation"]
            }
    
    def _get_critic_system_prompt(self) -> str:
        """Get system prompt for critic"""
        
        return """You are an expert English language educator and question validator. 
Your role is to critically evaluate English questions for educational quality, appropriateness, and effectiveness.

Focus on:
- Schema validity and completeness
- CEFR level appropriateness
- Error focus alignment
- Content quality and clarity
- Educational value
- Language accuracy

Provide constructive feedback and specific suggestions for improvement.
Always respond in the specified JSON format."""
    
    def _get_revision_system_prompt(self) -> str:
        """Get system prompt for revision"""
        
        return """You are an expert English language educator and question designer.
Your role is to revise English questions based on provided critique to improve their educational quality and effectiveness.

Guidelines:
- Fix all identified issues
- Maintain question format and structure
- Ensure CEFR level appropriateness
- Focus on specified error areas
- Improve clarity and educational value
- Keep questions engaging and relevant

Always respond in the specified JSON format."""
    
    def _validate_question_schema(self, question: Dict[str, Any], question_format: str) -> Dict[str, Any]:
        """Validate question schema and structure"""
        
        issues = []
        suggestions = []
        schema_score = 1.0
        
        # Check required fields
        required_fields = ["content", "correct_answer", "explanation"]
        for field in required_fields:
            if field not in question or not question[field]:
                issues.append(f"Missing required field: {field}")
                suggestions.append(f"Add {field} to the question")
                schema_score -= 0.3
        
        # Format-specific validation
        if question_format == "mcq":
            if "options" not in question or not isinstance(question["options"], list):
                issues.append("MCQ missing options array")
                suggestions.append("Add options array with 4 choices")
                schema_score -= 0.4
            elif len(question["options"]) != 4:
                issues.append("MCQ should have exactly 4 options")
                suggestions.append("Ensure exactly 4 options are provided")
                schema_score -= 0.2
        
        elif question_format == "cloze":
            if "blanks" not in question or not isinstance(question["blanks"], list):
                issues.append("Cloze missing blanks array")
                suggestions.append("Add blanks array with position and options")
                schema_score -= 0.4
        
        # Check content length
        content = question.get("content", "")
        if len(content) < 10:
            issues.append("Question content too short")
            suggestions.append("Expand question content")
            schema_score -= 0.2
        elif len(content) > 500:
            issues.append("Question content too long")
            suggestions.append("Make question more concise")
            schema_score -= 0.1
        
        return {
            "schema_valid": schema_score > 0.5,
            "schema_score": max(0.0, schema_score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _validate_cefr_compliance(self, question: Dict[str, Any], target_cefr: str) -> Dict[str, Any]:
        """Validate CEFR level compliance"""
        
        issues = []
        suggestions = []
        cefr_score = 1.0
        
        content = question.get("content", "")
        explanation = question.get("explanation", "")
        
        # Simple CEFR validation based on text complexity
        cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        target_index = cefr_levels.index(target_cefr) if target_cefr in cefr_levels else 2
        
        # Estimate complexity
        estimated_complexity = self._estimate_text_complexity(content + " " + explanation)
        
        # Check if complexity matches target
        complexity_diff = abs(estimated_complexity - target_index)
        if complexity_diff > 1:
            issues.append(f"CEFR level mismatch: estimated {cefr_levels[estimated_complexity]} vs target {target_cefr}")
            if estimated_complexity > target_index:
                suggestions.append("Simplify language for target CEFR level")
            else:
                suggestions.append("Increase complexity for target CEFR level")
            cefr_score -= 0.4
        
        # Check vocabulary appropriateness
        if self._has_inappropriate_vocabulary(content, target_cefr):
            issues.append("Vocabulary not appropriate for CEFR level")
            suggestions.append("Use vocabulary appropriate for target level")
            cefr_score -= 0.3
        
        return {
            "cefr_appropriate": cefr_score > 0.6,
            "cefr_score": max(0.0, cefr_score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _validate_error_focus_alignment(self, question: Dict[str, Any], error_focus: List[str]) -> Dict[str, Any]:
        """Validate alignment with error focus areas"""
        
        issues = []
        suggestions = []
        alignment_score = 1.0
        
        if not error_focus:
            return {
                "error_focus_aligned": True,
                "alignment_score": 1.0,
                "issues": [],
                "suggestions": []
            }
        
        content = question.get("content", "").lower()
        explanation = question.get("explanation", "").lower()
        full_text = content + " " + explanation
        
        # Check if error patterns are addressed
        addressed_patterns = 0
        for error in error_focus:
            error_lower = error.lower()
            if error_lower in full_text or any(word in full_text for word in error_lower.split()):
                addressed_patterns += 1
        
        coverage_ratio = addressed_patterns / len(error_focus)
        
        if coverage_ratio < 0.5:
            issues.append(f"Low error pattern coverage: {coverage_ratio:.1%}")
            suggestions.append("Include more error patterns from focus areas")
            alignment_score -= 0.5
        
        if coverage_ratio < 0.3:
            issues.append("Very low error pattern coverage")
            suggestions.append("Significantly improve error pattern alignment")
            alignment_score -= 0.3
        
        return {
            "error_focus_aligned": alignment_score > 0.6,
            "alignment_score": max(0.0, alignment_score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _validate_content_quality(self, question: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Validate content quality and relevance"""
        
        issues = []
        suggestions = []
        quality_score = 1.0
        
        content = question.get("content", "")
        explanation = question.get("explanation", "")
        
        # Check grammar and spelling (basic)
        if self._has_grammar_issues(content):
            issues.append("Grammar issues detected")
            suggestions.append("Fix grammar errors")
            quality_score -= 0.3
        
        # Check clarity
        if len(content.split()) < 5:
            issues.append("Question too short")
            suggestions.append("Make question more detailed")
            quality_score -= 0.2
        
        # Check relevance to context
        if context and not self._is_relevant_to_context(content, context):
            issues.append("Question not relevant to context")
            suggestions.append("Make question more relevant to provided context")
            quality_score -= 0.3
        
        # Check explanation quality
        if len(explanation) < 10:
            issues.append("Explanation too short")
            suggestions.append("Provide more detailed explanation")
            quality_score -= 0.2
        
        return {
            "content_quality_acceptable": quality_score > 0.6,
            "content_quality_score": max(0.0, quality_score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _validate_educational_value(self, question: Dict[str, Any], target_cefr: str) -> Dict[str, Any]:
        """Validate educational value and learning objectives"""
        
        issues = []
        suggestions = []
        educational_score = 1.0
        
        content = question.get("content", "")
        explanation = question.get("explanation", "")
        
        # Check if question tests understanding
        if self._is_too_simple(content):
            issues.append("Question too simple for educational value")
            suggestions.append("Increase question complexity")
            educational_score -= 0.3
        
        # Check if explanation is educational
        if not self._is_educational_explanation(explanation):
            issues.append("Explanation lacks educational value")
            suggestions.append("Provide more educational explanation")
            educational_score -= 0.3
        
        # Check for learning objectives
        if not self._has_learning_objectives(content, explanation):
            issues.append("Missing clear learning objectives")
            suggestions.append("Include clear learning objectives")
            educational_score -= 0.2
        
        return {
            "educational_value_acceptable": educational_score > 0.6,
            "educational_score": max(0.0, educational_score),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _combine_critiques(self, critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple critiques into a single assessment"""
        
        all_issues = []
        all_suggestions = []
        weighted_score = 0.0
        total_weight = 0.0
        
        # Combine scores with weights
        for critique in critiques:
            if "schema_score" in critique:
                weighted_score += critique["schema_score"] * 0.2
                total_weight += 0.2
            if "cefr_score" in critique:
                weighted_score += critique["cefr_score"] * self.cefr_compliance_weight
                total_weight += self.cefr_compliance_weight
            if "alignment_score" in critique:
                weighted_score += critique["alignment_score"] * self.error_focus_weight
                total_weight += self.error_focus_weight
            if "content_quality_score" in critique:
                weighted_score += critique["content_quality_score"] * self.content_quality_weight
                total_weight += self.content_quality_weight
            if "educational_score" in critique:
                weighted_score += critique["educational_score"] * self.educational_value_weight
                total_weight += self.educational_value_weight
            
            # Collect issues and suggestions
            all_issues.extend(critique.get("issues", []))
            all_suggestions.extend(critique.get("suggestions", []))
        
        # Calculate final score
        final_score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Determine if revision is needed
        needs_revision = final_score < self.quality_threshold or len(all_issues) > 0
        
        return {
            "needs_revision": needs_revision,
            "overall_score": final_score,
            "issues": list(set(all_issues)),  # Remove duplicates
            "suggestions": list(set(all_suggestions)),  # Remove duplicates
            "schema_valid": all(c.get("schema_valid", True) for c in critiques),
            "cefr_appropriate": all(c.get("cefr_appropriate", True) for c in critiques),
            "error_focus_aligned": all(c.get("error_focus_aligned", True) for c in critiques),
            "content_quality": final_score,
            "distractor_quality": 0.7  # Default value
        }
    
    def _validate_critique(self, critique: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize critique structure (legacy method)"""
        
        # Ensure required fields exist
        validated = {
            "needs_revision": critique.get("needs_revision", False),
            "overall_score": critique.get("overall_score", 0.0),
            "issues": critique.get("issues", []),
            "suggestions": critique.get("suggestions", []),
            "schema_valid": critique.get("schema_valid", True),
            "cefr_appropriate": critique.get("cefr_appropriate", True),
            "error_focus_aligned": critique.get("error_focus_aligned", True),
            "content_quality": critique.get("content_quality", 0.5),
            "distractor_quality": critique.get("distractor_quality", 0.5)
        }
        
        # Ensure lists are actually lists
        if not isinstance(validated["issues"], list):
            validated["issues"] = []
        if not isinstance(validated["suggestions"], list):
            validated["suggestions"] = []
        
        # Normalize score to 0-1 range
        validated["overall_score"] = max(0.0, min(1.0, float(validated["overall_score"])))
        validated["content_quality"] = max(0.0, min(1.0, float(validated["content_quality"])))
        validated["distractor_quality"] = max(0.0, min(1.0, float(validated["distractor_quality"])))
        
        return validated
    
    def _validate_question(self, question: Dict[str, Any], question_format: str) -> Dict[str, Any]:
        """Validate and normalize question structure"""
        
        # Ensure basic fields exist
        validated = {
            "content": question.get("content", ""),
            "correct_answer": question.get("correct_answer", ""),
            "explanation": question.get("explanation", ""),
            "difficulty_level": question.get("difficulty_level", 3),
            "topic": question.get("topic", ""),
            "cefr_level": question.get("cefr_level", "B1")
        }
        
        # Format-specific validation
        if question_format == "mcq":
            options = question.get("options", [])
            if isinstance(options, list) and len(options) >= 4:
                validated["options"] = options[:4]  # Ensure exactly 4 options
            else:
                validated["options"] = ["A", "B", "C", "D"]  # Fallback
        
        elif question_format == "cloze":
            blanks = question.get("blanks", [])
            if isinstance(blanks, list):
                validated["blanks"] = blanks
            else:
                validated["blanks"] = []
        
        # Normalize difficulty level
        validated["difficulty_level"] = max(1, min(5, int(validated["difficulty_level"])))
        
        return validated
    
    async def _get_llm_critique(
        self,
        question: Dict[str, Any],
        target_cefr: str,
        error_focus: List[str],
        context: str,
        question_format: str
    ) -> Dict[str, Any]:
        """Get LLM-based critique (if available)"""
        
        try:
            # Build critic prompt
            critic_prompt = self._build_critic_prompt(
                question, target_cefr, error_focus, context, question_format
            )
            
            # Get critique from MCP
            from app.core.mcp_utils import mcp_utils
            
            try:
                if mcp_utils.is_initialized:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="llm_generate",
                        arguments={
                            "prompt": critic_prompt,
                            "system_prompt": self._get_critic_system_prompt(),
                            "output_type": "json",
                            "schema": self._get_critic_schema(),
                            "temperature": 0.2,
                            "max_tokens": 800,
                            "context": context,
                            "user_id": "critic_revise",
                            "session_id": f"critique_{target_cefr}"
                        }
                    )
                    
                    if mcp_response["success"]:
                        critique_result = mcp_response["data"]
                        if isinstance(critique_result, str):
                            import json
                            critique_result = json.loads(critique_result)
                    else:
                        logger.warning(f"MCP critique failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                        critique_result = await llm_gateway.generate_structured_with_fallback(
                            task_type="question_critique",
                            prompt=critic_prompt,
                            schema=self._get_critic_schema(),
                            system_prompt=self._get_critic_system_prompt(),
                            complexity="medium"
                        )
                else:
                    logger.warning("MCP not initialized, using direct LLM")
                    # Fallback to direct LLM
                    critique_result = await llm_gateway.generate_structured_with_fallback(
                        task_type="question_critique",
                        prompt=critic_prompt,
                        schema=self._get_critic_schema(),
                        system_prompt=self._get_critic_system_prompt(),
                        complexity="medium"
                    )
            except Exception as e:
                logger.warning(f"MCP critique failed, using fallback: {e}")
                # Fallback to direct LLM
                critique_result = await llm_gateway.generate_structured_with_fallback(
                    task_type="question_critique",
                    prompt=critic_prompt,
                    schema=self._get_critic_schema(),
                    system_prompt=self._get_critic_system_prompt(),
                    complexity="medium"
                )
            
            if critique_result.get("success"):
                critique = critique_result.get("data", {})
                return self._validate_critique(critique)
            else:
                logger.warning("LLM critique failed, using basic validation")
                return self._get_basic_critique(question, target_cefr)
                
        except Exception as e:
            logger.error(f"Error in LLM critique: {e}")
            return self._get_basic_critique(question, target_cefr)
    
    def _get_basic_critique(self, question: Dict[str, Any], target_cefr: str) -> Dict[str, Any]:
        """Get basic critique when LLM is not available"""
        
        content = question.get("content", "")
        issues = []
        suggestions = []
        
        # Check content length
        if len(content) < 20:
            issues.append("Question content is too short")
            suggestions.append("Expand the question content")
        
        if len(content) > 200:
            issues.append("Question content is too long")
            suggestions.append("Make the question more concise")
        
        # Check for basic structure
        if not content.strip():
            issues.append("Question content is empty")
            suggestions.append("Add meaningful question content")
        
        # Determine if revision is needed
        needs_revision = len(issues) > 0
        
        return {
            "needs_revision": needs_revision,
            "overall_score": 0.7 if not needs_revision else 0.4,
            "issues": issues,
            "suggestions": suggestions,
            "schema_valid": True,
            "cefr_appropriate": True,
            "error_focus_aligned": True,
            "content_quality": 0.7 if not needs_revision else 0.4,
            "distractor_quality": 0.6
        }
    
    def _fallback_critique(self, question: Dict[str, Any], target_cefr: str) -> Dict[str, Any]:
        """Fallback critique when all methods fail"""
        
        return self._get_basic_critique(question, target_cefr)
    
    # Helper methods for validation
    def _estimate_text_complexity(self, text: str) -> int:
        """Estimate text complexity (0-5 scale)"""
        
        words = text.lower().split()
        if not words:
            return 2  # Default to B1
        
        # Count complex words (words with more than 6 letters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = complex_words / len(words)
        
        # Count average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple complexity estimation
        if complexity_ratio < 0.1 and avg_word_length < 4.5:
            return 0  # A1
        elif complexity_ratio < 0.2 and avg_word_length < 5.0:
            return 1  # A2
        elif complexity_ratio < 0.3 and avg_word_length < 5.5:
            return 2  # B1
        elif complexity_ratio < 0.4 and avg_word_length < 6.0:
            return 3  # B2
        elif complexity_ratio < 0.5 and avg_word_length < 6.5:
            return 4  # C1
        else:
            return 5  # C2
    
    def _has_inappropriate_vocabulary(self, text: str, target_cefr: str) -> bool:
        """Check if text has inappropriate vocabulary for CEFR level"""
        
        # Simple heuristic - can be enhanced with vocabulary lists
        cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        target_index = cefr_levels.index(target_cefr) if target_cefr in cefr_levels else 2
        
        # Estimate complexity
        estimated_complexity = self._estimate_text_complexity(text)
        
        # Check if complexity is too high for target level
        return estimated_complexity > target_index + 1
    
    def _has_grammar_issues(self, text: str) -> bool:
        """Basic grammar check (simplified)"""
        
        # Simple checks - can be enhanced with proper grammar checking
        issues = []
        
        # Check for basic patterns
        if "i " in text and not text.startswith("I "):
            issues.append("Lowercase 'i' should be 'I'")
        
        if text.count(".") > 0 and not text.endswith("."):
            issues.append("Missing period at end")
        
        return len(issues) > 0
    
    def _is_relevant_to_context(self, content: str, context: str) -> bool:
        """Check if content is relevant to context"""
        
        if not context:
            return True
        
        # Simple keyword matching
        content_words = set(content.lower().split())
        context_words = set(context.lower().split())
        
        # Check for common words
        common_words = content_words & context_words
        relevance_ratio = len(common_words) / len(content_words) if content_words else 0
        
        return relevance_ratio > 0.1  # At least 10% word overlap
    
    def _is_too_simple(self, content: str) -> bool:
        """Check if content is too simple"""
        
        words = content.split()
        if len(words) < 5:
            return True
        
        # Check for complex structures
        has_complex_structure = any(len(word) > 8 for word in words)
        
        return not has_complex_structure and len(words) < 10
    
    def _is_educational_explanation(self, explanation: str) -> bool:
        """Check if explanation has educational value"""
        
        if len(explanation) < 20:
            return False
        
        # Check for educational keywords
        educational_keywords = ["because", "therefore", "since", "as", "due to", "reason", "explain", "shows", "demonstrates"]
        explanation_lower = explanation.lower()
        
        return any(keyword in explanation_lower for keyword in educational_keywords)
    
    def _has_learning_objectives(self, content: str, explanation: str) -> bool:
        """Check if content has clear learning objectives"""
        
        # Check for learning objective indicators
        objective_indicators = ["learn", "understand", "identify", "recognize", "apply", "practice", "improve"]
        full_text = (content + " " + explanation).lower()
        
        return any(indicator in full_text for indicator in objective_indicators)


# Global instance
critic_revise_service = CriticReviseService()
