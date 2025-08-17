from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from difflib import SequenceMatcher
from collections import Counter
import json

logger = logging.getLogger(__name__)


class ContextCompressionService:
    """Context compression service for RAG-based question generation"""
    
    def __init__(self):
        self.max_tokens = 1200
        self.target_tokens = 1000
        self.similarity_threshold = 0.9
        self.cache_ttl = 1800  # 30 minutes
        
        # Advanced compression settings
        self.min_nugget_length = 10
        self.max_nugget_length = 200
        self.cefr_validation_enabled = True
        self.error_focus_weight = 0.8
        
    async def compress_context(
        self,
        passages: List[Dict[str, Any]],
        target_cefr: str,
        error_focus: List[str],
        budget_tokens: int = None
    ) -> Dict[str, Any]:
        """Compress context using Map→Reduce approach"""
        
        if budget_tokens:
            self.target_tokens = min(budget_tokens, self.max_tokens)
        
        try:
            # Map: Extract nuggets from each passage
            nuggets = await self._extract_nuggets(passages, target_cefr, error_focus)
            
            # Reduce: Deduplicate and merge similar nuggets
            unique_nuggets = await self._deduplicate_nuggets(nuggets)
            
            # Compress to target token budget
            compressed = await self._compress_to_budget(unique_nuggets)
            
            # Validate CEFR compliance
            cefr_validated = await self._validate_cefr_compliance(compressed, target_cefr)
            
            return {
                "compressed_text": cefr_validated["text"],
                "source_ids": cefr_validated["source_ids"],
                "token_count": cefr_validated["token_count"],
                "compression_ratio": len(passages) / len(unique_nuggets) if unique_nuggets else 1.0,
                "cefr_compliance": cefr_validated["cefr_compliant"],
                "error_coverage": cefr_validated["error_coverage"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in context compression: {e}")
            return await self._fallback_compression(passages, target_cefr)
    
    async def _extract_nuggets(
        self,
        passages: List[Dict[str, Any]],
        target_cefr: str,
        error_focus: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract key nuggets from passages using advanced semantic analysis"""
        
        nuggets = []
        
        for passage in passages:
            content = passage.get("content", "")
            source_id = passage.get("id", "")
            
            # Extract sentences that match error focus
            sentences = self._split_into_sentences(content)
            
            for sentence in sentences:
                # Filter by length
                if len(sentence) < self.min_nugget_length or len(sentence) > self.max_nugget_length:
                    continue
                
                # Check if sentence is relevant to error focus
                if self._is_relevant_to_errors(sentence, error_focus):
                    # Calculate advanced relevance score
                    relevance_score = self._calculate_advanced_relevance_score(sentence, error_focus, target_cefr)
                    
                    # Estimate CEFR level
                    cefr_level = self._estimate_cefr_level(sentence)
                    
                    # Check CEFR compliance if enabled
                    if self.cefr_validation_enabled:
                        cefr_compliant = self._is_cefr_compliant(cefr_level, target_cefr)
                        if not cefr_compliant:
                            continue  # Skip non-compliant nuggets
                    
                    nugget = {
                        "text": sentence.strip(),
                        "source_id": source_id,
                        "relevance_score": relevance_score,
                        "cefr_level": cefr_level,
                        "length": len(sentence.split()),
                        "complexity_score": self._calculate_complexity_score(sentence),
                        "error_coverage": self._calculate_error_coverage(sentence, error_focus)
                    }
                    nuggets.append(nugget)
        
        # Sort by advanced relevance score
        nuggets.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return nuggets
    
    async def _deduplicate_nuggets(self, nuggets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and very similar nuggets using intelligent merging"""
        
        if not nuggets:
            return []
        
        unique_nuggets = [nuggets[0]]
        
        for nugget in nuggets[1:]:
            is_duplicate = False
            best_match = None
            best_similarity = 0.0
            
            # Find the most similar existing nugget
            for existing in unique_nuggets:
                similarity = self._calculate_similarity(nugget["text"], existing["text"])
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_match = existing
                    best_similarity = similarity
            
            if best_match:
                # Merge similar nuggets instead of just keeping one
                merged_nugget = self._merge_similar_nuggets(best_match, nugget)
                
                # Replace the existing nugget with the merged one
                unique_nuggets.remove(best_match)
                unique_nuggets.append(merged_nugget)
                is_duplicate = True
            
            if not is_duplicate:
                unique_nuggets.append(nugget)
        
        # Sort by relevance score after deduplication
        unique_nuggets.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return unique_nuggets
    
    def _merge_similar_nuggets(self, nugget1: Dict[str, Any], nugget2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two similar nuggets intelligently"""
        
        # Choose the better text (longer and more informative)
        if len(nugget1["text"]) > len(nugget2["text"]):
            better_text = nugget1["text"]
            better_nugget = nugget1
            other_nugget = nugget2
        else:
            better_text = nugget2["text"]
            better_nugget = nugget2
            other_nugget = nugget1
        
        # Combine source IDs
        combined_sources = [better_nugget["source_id"]]
        if other_nugget["source_id"] not in combined_sources:
            combined_sources.append(other_nugget["source_id"])
        
        # Calculate combined scores
        combined_relevance = max(better_nugget["relevance_score"], other_nugget["relevance_score"])
        combined_complexity = (better_nugget.get("complexity_score", 0.5) + other_nugget.get("complexity_score", 0.5)) / 2
        combined_error_coverage = max(better_nugget.get("error_coverage", 0.5), other_nugget.get("error_coverage", 0.5))
        
        # Determine best CEFR level (prefer the one closer to target)
        cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        try:
            level1_index = cefr_levels.index(better_nugget["cefr_level"])
            level2_index = cefr_levels.index(other_nugget["cefr_level"])
            
            # Prefer the level that's more moderate (closer to B1/B2)
            target_index = 2  # B1
            distance1 = abs(level1_index - target_index)
            distance2 = abs(level2_index - target_index)
            
            better_cefr = better_nugget["cefr_level"] if distance1 <= distance2 else other_nugget["cefr_level"]
        except ValueError:
            better_cefr = better_nugget["cefr_level"]
        
        return {
            "text": better_text,
            "source_id": combined_sources[0],  # Use primary source
            "relevance_score": combined_relevance,
            "cefr_level": better_cefr,
            "length": len(better_text.split()),
            "complexity_score": combined_complexity,
            "error_coverage": combined_error_coverage,
            "merged_sources": combined_sources
        }
    
    async def _compress_to_budget(self, nuggets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compress nuggets to fit token budget using adaptive compression"""
        
        if not nuggets:
            return {"text": "", "source_ids": [], "token_count": 0}
        
        compressed_text = ""
        source_ids = []
        current_tokens = 0
        quality_score = 0.0
        nugget_count = 0
        
        # Calculate adaptive budget based on nugget quality
        total_quality = sum(nugget["relevance_score"] for nugget in nuggets)
        avg_quality = total_quality / len(nuggets) if nuggets else 0.5
        
        # Adjust target tokens based on quality
        adjusted_target = int(self.target_tokens * (0.8 + avg_quality * 0.4))  # 80%-120% of target
        
        for nugget in nuggets:
            nugget_tokens = nugget["length"]
            nugget_quality = nugget["relevance_score"]
            
            # Check if adding this nugget would exceed budget
            if current_tokens + nugget_tokens <= adjusted_target:
                if compressed_text:
                    compressed_text += " "
                compressed_text += nugget["text"]
                source_ids.append(nugget["source_id"])
                current_tokens += nugget_tokens
                quality_score += nugget_quality
                nugget_count += 1
            else:
                # Try to truncate the nugget if it's high quality
                if nugget_quality > 0.7:  # High quality nugget
                    remaining_tokens = adjusted_target - current_tokens
                    if remaining_tokens >= 5:  # Minimum meaningful length
                        truncated_text = self._truncate_text(nugget["text"], remaining_tokens)
                        if compressed_text:
                            compressed_text += " "
                        compressed_text += truncated_text
                        source_ids.append(nugget["source_id"])
                        current_tokens += len(truncated_text.split())
                        quality_score += nugget_quality * 0.5  # Reduced quality for truncation
                        nugget_count += 1
                break
        
        # Calculate average quality
        avg_quality = quality_score / nugget_count if nugget_count > 0 else 0.0
        
        return {
            "text": compressed_text,
            "source_ids": list(set(source_ids)),  # Remove duplicates
            "token_count": current_tokens,
            "quality_score": avg_quality,
            "nugget_count": nugget_count,
            "compression_ratio": len(nuggets) / nugget_count if nugget_count > 0 else 1.0
        }
    
    async def _validate_cefr_compliance(
        self,
        compressed: Dict[str, Any],
        target_cefr: str
    ) -> Dict[str, Any]:
        """Validate CEFR compliance of compressed text"""
        
        text = compressed["text"]
        
        # Simple CEFR validation (can be enhanced with proper CEFR validator)
        estimated_cefr = self._estimate_cefr_level(text)
        
        # Check if estimated CEFR is within acceptable range
        cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        target_index = cefr_levels.index(target_cefr) if target_cefr in cefr_levels else 2
        estimated_index = cefr_levels.index(estimated_cefr) if estimated_cefr in cefr_levels else 2
        
        # Allow ±1 level difference
        cefr_compliant = abs(target_index - estimated_index) <= 1
        
        return {
            **compressed,
            "cefr_compliant": cefr_compliant,
            "estimated_cefr": estimated_cefr,
            "target_cefr": target_cefr
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _is_relevant_to_errors(self, sentence: str, error_focus: List[str]) -> bool:
        """Check if sentence is relevant to error patterns"""
        
        if not error_focus:
            return True
        
        sentence_lower = sentence.lower()
        
        for error in error_focus:
            error_lower = error.lower()
            
            # Check for exact match or partial match
            if error_lower in sentence_lower or sentence_lower in error_lower:
                return True
            
            # Check for related words
            error_words = error_lower.split()
            sentence_words = sentence_lower.split()
            
            common_words = set(error_words) & set(sentence_words)
            if len(common_words) >= min(len(error_words), 2):
                return True
        
        return False
    
    def _calculate_advanced_relevance_score(self, sentence: str, error_focus: List[str], target_cefr: str) -> float:
        """Calculate advanced relevance score considering multiple factors"""
        
        if not error_focus:
            return 0.5
        
        # Base relevance score
        base_score = self._calculate_relevance_score(sentence, error_focus)
        
        # CEFR compliance bonus
        cefr_level = self._estimate_cefr_level(sentence)
        cefr_bonus = 0.2 if self._is_cefr_compliant(cefr_level, target_cefr) else -0.1
        
        # Complexity bonus (prefer moderate complexity)
        complexity_score = self._calculate_complexity_score(sentence)
        complexity_bonus = 0.1 if 0.3 <= complexity_score <= 0.7 else -0.05
        
        # Error coverage bonus
        error_coverage = self._calculate_error_coverage(sentence, error_focus)
        coverage_bonus = error_coverage * 0.3
        
        # Length bonus (prefer medium length)
        length = len(sentence.split())
        length_bonus = 0.1 if 10 <= length <= 25 else -0.05
        
        # Calculate final score
        final_score = base_score + cefr_bonus + complexity_bonus + coverage_bonus + length_bonus
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_relevance_score(self, sentence: str, error_focus: List[str]) -> float:
        """Calculate basic relevance score for a sentence"""
        
        if not error_focus:
            return 0.5
        
        sentence_lower = sentence.lower()
        total_score = 0.0
        
        for error in error_focus:
            error_lower = error.lower()
            
            # Exact match gets highest score
            if error_lower in sentence_lower:
                total_score += 1.0
            elif sentence_lower in error_lower:
                total_score += 0.8
            else:
                # Partial match based on word overlap
                error_words = set(error_lower.split())
                sentence_words = set(sentence_lower.split())
                
                if error_words and sentence_words:
                    overlap = len(error_words & sentence_words)
                    total_score += overlap / len(error_words) * 0.6
        
        return min(total_score / len(error_focus), 1.0)
    
    def _estimate_cefr_level(self, text: str) -> str:
        """Estimate CEFR level of text (simplified)"""
        
        # Simple heuristics for CEFR estimation
        words = text.lower().split()
        
        # Count complex words (words with more than 6 letters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = complex_words / len(words) if words else 0
        
        # Count sentence length
        sentences = self._split_into_sentences(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Simple CEFR estimation
        if complexity_ratio < 0.1 and avg_sentence_length < 8:
            return "A1"
        elif complexity_ratio < 0.2 and avg_sentence_length < 12:
            return "A2"
        elif complexity_ratio < 0.3 and avg_sentence_length < 15:
            return "B1"
        elif complexity_ratio < 0.4 and avg_sentence_length < 18:
            return "B2"
        elif complexity_ratio < 0.5 and avg_sentence_length < 20:
            return "C1"
        else:
            return "C2"
    
    def _is_cefr_compliant(self, estimated_cefr: str, target_cefr: str) -> bool:
        """Check if estimated CEFR is compliant with target CEFR"""
        
        cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        
        try:
            target_index = cefr_levels.index(target_cefr)
            estimated_index = cefr_levels.index(estimated_cefr)
            
            # Allow ±1 level difference
            return abs(target_index - estimated_index) <= 1
        except ValueError:
            # If CEFR level not found, assume compliant
            return True
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score for text"""
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Count complex words (words with more than 6 letters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = complex_words / len(words)
        
        # Count average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Count sentence length
        sentences = self._split_into_sentences(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Normalize scores
        complexity_score = min(complexity_ratio * 2, 1.0)  # Scale to 0-1
        word_length_score = min(avg_word_length / 8, 1.0)  # Normalize to 0-1
        sentence_length_score = min(avg_sentence_length / 20, 1.0)  # Normalize to 0-1
        
        # Combine scores
        final_score = (complexity_score + word_length_score + sentence_length_score) / 3
        
        return final_score
    
    def _calculate_error_coverage(self, sentence: str, error_focus: List[str]) -> float:
        """Calculate how well a sentence covers the error focus areas"""
        
        if not error_focus:
            return 0.5
        
        sentence_lower = sentence.lower()
        covered_errors = 0
        
        for error in error_focus:
            error_lower = error.lower()
            
            # Check for exact match or partial match
            if error_lower in sentence_lower or sentence_lower in error_lower:
                covered_errors += 1
            else:
                # Check for word overlap
                error_words = set(error_lower.split())
                sentence_words = set(sentence_lower.split())
                
                if error_words and sentence_words:
                    overlap = len(error_words & sentence_words)
                    if overlap >= min(len(error_words), 2):  # At least 2 words overlap
                        covered_errors += 0.5
        
        return covered_errors / len(error_focus)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit"""
        
        words = text.split()
        if len(words) <= max_tokens:
            return text
        
        # Truncate to max_tokens, keeping complete words
        truncated_words = words[:max_tokens]
        return " ".join(truncated_words)
    
    async def _fallback_compression(
        self,
        passages: List[Dict[str, Any]],
        target_cefr: str
    ) -> Dict[str, Any]:
        """Fallback compression when main method fails"""
        
        try:
            # Simple concatenation of first few passages
            compressed_text = ""
            source_ids = []
            token_count = 0
            
            for passage in passages[:3]:  # Take first 3 passages
                content = passage.get("content", "")
                source_id = passage.get("id", "")
                
                if token_count + len(content.split()) <= self.target_tokens:
                    if compressed_text:
                        compressed_text += " "
                    compressed_text += content
                    source_ids.append(source_id)
                    token_count += len(content.split())
                else:
                    break
            
            return {
                "compressed_text": compressed_text,
                "source_ids": source_ids,
                "token_count": token_count,
                "compression_ratio": 1.0,
                "cefr_compliance": True,
                "error_coverage": 0.5
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback compression also failed: {e}")
            return {
                "compressed_text": "Error in context compression",
                "source_ids": [],
                "token_count": 0,
                "compression_ratio": 1.0,
                "cefr_compliance": False,
                "error_coverage": 0.0
            }


# Global instance
context_compression_service = ContextCompressionService()
