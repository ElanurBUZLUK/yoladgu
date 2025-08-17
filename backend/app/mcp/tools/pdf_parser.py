from typing import Dict, Any, List, Optional
from .base import BaseMCPTool
import os
import re
import pdfplumber
import PyPDF2
from .pdf_content_reader import PDFContentReaderTool


class PDFParserTool(BaseMCPTool):
    """PDF'den soru çıkarma için MCP tool"""
    
    def get_name(self) -> str:
        return "parse_pdf_questions"
    
    def get_description(self) -> str:
        return "PDF'den soruları çıkarır, kategorize eder ve öğrenciye gönderilecek formata dönüştürür"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "PDF dosya yolu"
                },
                "subject": {
                    "type": "string",
                    "enum": ["math", "english"],
                    "description": "Ders konusu"
                },
                "expected_difficulty": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 3,
                    "description": "Beklenen zorluk seviyesi"
                },
                "extract_images": {
                    "type": "boolean",
                    "default": True,
                    "description": "Görselleri çıkar"
                },
                "preserve_formatting": {
                    "type": "boolean",
                    "default": True,
                    "description": "Formatı koru"
                },
                "question_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["multiple_choice", "fill_blank", "open_ended", "true_false"]
                    },
                    "description": "Çıkarılacak soru tipleri"
                }
            },
            "required": ["pdf_path", "subject"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """PDF parsing mantığı"""
        
        pdf_path = arguments["pdf_path"]
        subject = arguments["subject"]
        expected_difficulty = arguments.get("expected_difficulty", 3)
        extract_images = arguments.get("extract_images", True)
        preserve_formatting = arguments.get("preserve_formatting", True)
        question_types = arguments.get("question_types", ["multiple_choice", "fill_blank", "open_ended"])
        
        # PDF dosyası kontrolü
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": "PDF dosyası bulunamadı",
                "pdf_path": pdf_path
            }
        
        # PDF içeriğini çıkar
        pdf_content = await self._extract_pdf_content(pdf_path, extract_images)
        
        # Soruları parse et
        parsed_questions = await self._parse_questions(
            pdf_content, subject, expected_difficulty, question_types, preserve_formatting
        )
        
        # Kalite kontrol
        validated_questions = await self._validate_questions(parsed_questions, subject)
        
        return {
            "success": True,
            "extraction_info": {
                "pdf_path": pdf_path,
                "subject": subject,
                "total_pages": pdf_content.get("total_pages", 0),
                "extraction_method": "text_and_image" if extract_images else "text_only"
            },
            "questions": validated_questions,
            "statistics": {
                "total_extracted": len(parsed_questions),
                "validated_count": len(validated_questions),
                "by_type": self._count_by_type(validated_questions),
                "by_difficulty": self._count_by_difficulty(validated_questions)
            },
            "metadata": {
                "extraction_timestamp": "2024-01-01T00:00:00Z",
                "processing_options": {
                    "extract_images": extract_images,
                    "preserve_formatting": preserve_formatting,
                    "expected_difficulty": expected_difficulty
                }
            }
        }
    
    async def _extract_pdf_content(self, pdf_path: str, extract_images: bool) -> Dict[str, Any]:
        """PDF içeriğini çıkar"""
        # PDF Content Reader Tool'u kullan
        content_reader = PDFContentReaderTool()
        
        content_result = await content_reader.execute({
            "pdf_path": pdf_path,
            "extract_mode": "both" if extract_images else "text_only",
            "preserve_layout": True
        })
        
        if not content_result.get("success"):
            raise Exception(f"PDF content extraction failed: {content_result.get('error')}")
        
        content_data = content_result["content"]
        
        # Format'ı eski API ile uyumlu hale getir
        formatted_content = {
            "total_pages": content_data["pdf_info"]["total_pages"],
            "text_content": [],
            "images": content_data.get("images", []),
            "formatting_info": {
                "fonts": [],
                "font_sizes": [],
                "has_tables": False,
                "has_diagrams": len(content_data.get("images", [])) > 0
            },
            "pages": content_data.get("pages", [])
        }
        
        # Metin içeriğini düzenle
        if "text_content" in content_data:
            for page_data in content_data["text_content"]:
                if page_data.get("text"):
                    formatted_content["text_content"].append(page_data["text"])
                
                # Font bilgilerini topla
                if page_data.get("fonts"):
                    for font_info in page_data["fonts"].keys():
                        font_name = font_info.split("_")[0]
                        if font_name not in formatted_content["formatting_info"]["fonts"]:
                            formatted_content["formatting_info"]["fonts"].append(font_name)
                
                # Tablo kontrolü
                if page_data.get("table_count", 0) > 0:
                    formatted_content["formatting_info"]["has_tables"] = True
        
        return formatted_content
    
    async def _parse_questions(self, pdf_content: Dict, subject: str, expected_difficulty: int, 
                             question_types: List[str], preserve_formatting: bool) -> List[Dict[str, Any]]:
        """PDF içeriğinden soruları parse et"""
        
        questions = []
        
        # Hem text_content hem de pages'i kontrol et
        text_sources = []
        
        # Eski format text_content
        if "text_content" in pdf_content:
            for i, text in enumerate(pdf_content["text_content"]):
                text_sources.append({"text": text, "page": i + 1, "source": "text_content"})
        
        # Yeni format pages
        if "pages" in pdf_content:
            for page_data in pdf_content["pages"]:
                if page_data.get("text"):
                    text_sources.append({
                        "text": page_data["text"], 
                        "page": page_data["page_number"], 
                        "source": "pages"
                    })
        
        # Her metin kaynağını parse et
        for source in text_sources:
            text = source["text"]
            page_num = source["page"]
            
            # Regex pattern'ları ile soru tespiti
            detected_questions = await self._detect_questions_with_patterns(
                text, subject, expected_difficulty, page_num, preserve_formatting
            )
            
            questions.extend(detected_questions)
        
        return questions
    
    async def _detect_questions_with_patterns(self, text: str, subject: str, difficulty: int, 
                                            page_num: int, preserve_formatting: bool) -> List[Dict[str, Any]]:
        """Pattern'lar ile soru tespiti"""
        questions = []
        
        # Çoktan seçmeli soru pattern'ları
        multiple_choice_patterns = [
            r'(\d+\.?\s*[^?]*\?\s*(?:\n|^)\s*[A-D]\)\s*[^\n]+(?:\s*[A-D]\)\s*[^\n]+)*)',
            r'(Question\s+\d+[^?]*\?\s*(?:\n|^)\s*\([a-d]\)\s*[^\n]+(?:\s*\([a-d]\)\s*[^\n]+)*)',
            r'([^.]*\?\s*(?:\n|^)\s*[A-D][\)\.]\s*[^\n]+(?:\s*[A-D][\)\.]\s*[^\n]+){1,3})'
        ]
        
        # Fill in the blank pattern'ları
        fill_blank_patterns = [
            r'([^.]*(?:___+|_____+|______+)[^.]*\.?)',
            r'(Fill\s+in\s+the\s+blank[^.]*\.)',
            r'(Complete\s+the\s+sentence[^.]*\.)',
            r'([^.]*\([^)]*\)[^.]*\.)'  # Parantez içinde cevap
        ]
        
        # Open-ended pattern'ları
        open_ended_patterns = [
            r'(Write\s+[^.]*\.)',
            r'(Describe\s+[^.]*\.)',
            r'(Explain\s+[^.]*\.)',
            r'(Discuss\s+[^.]*\.)'
        ]
        
        # Pattern'ları uygula
        for pattern in multiple_choice_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                question = await self._parse_multiple_choice_question(
                    match.group(1), subject, difficulty, page_num, preserve_formatting
                )
                if question:
                    questions.append(question)
        
        for pattern in fill_blank_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                question = await self._parse_fill_blank_question(
                    match.group(1), subject, difficulty, page_num, preserve_formatting
                )
                if question:
                    questions.append(question)
        
        for pattern in open_ended_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                question = await self._parse_open_ended_question(
                    match.group(1), subject, difficulty, page_num, preserve_formatting
                )
                if question:
                    questions.append(question)
        
        return questions
    
    async def _parse_multiple_choice_question(self, text: str, subject: str, difficulty: int, 
                                            page_num: int, preserve_formatting: bool) -> Optional[Dict[str, Any]]:
        """Çoktan seçmeli soruyu parse et"""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Soru metnini bul
        question_text = lines[0].strip()
        
        # Seçenekleri bul
        options = []
        correct_answer = None
        
        for line in lines[1:]:
            line = line.strip()
            # A), B), C), D) formatı
            option_match = re.match(r'^([A-D])\)\s*(.+)$', line)
            if option_match:
                option_letter = option_match.group(1)
                option_text = option_match.group(2)
                options.append(f"{option_letter}) {option_text}")
                
                # Doğru cevabı tahmin et (basit heuristic)
                if not correct_answer and self._is_likely_correct_answer(option_text, subject):
                    correct_answer = option_letter
        
        if len(options) < 2:
            return None
        
        # Varsayılan doğru cevap
        if not correct_answer:
            correct_answer = "A"
        
        question_id = f"pdf_{hash(text)}_{page_num}"
        
        return {
            "id": question_id,
            "content": question_text,
            "question_type": "multiple_choice",
            "difficulty_level": difficulty,
            "subject": subject,
            "options": options,
            "correct_answer": correct_answer,
            "source_page": page_num,
            "original_text": text if preserve_formatting else None,
            "topic_category": self._determine_topic(question_text, subject),
            "estimated_time": self._estimate_time("multiple_choice", difficulty)
        }
    
    async def _parse_fill_blank_question(self, text: str, subject: str, difficulty: int, 
                                       page_num: int, preserve_formatting: bool) -> Optional[Dict[str, Any]]:
        """Boşluk doldurma sorusunu parse et"""
        # Cevabı bul
        correct_answer = None
        question_text = text
        
        # Parantez içinde cevap arama
        answer_match = re.search(r'\(([^)]+)\)', text)
        if answer_match:
            correct_answer = answer_match.group(1).strip()
            # Cevabı soru metninden çıkar
            question_text = re.sub(r'\([^)]+\)', '', text).strip()
        
        # "Answer:" formatı arama
        if "Answer:" in text:
            parts = text.split("Answer:")
            if len(parts) == 2:
                question_text = parts[0].strip()
                correct_answer = parts[1].strip()
        
        # Boşluk kontrolü
        if not ("___" in question_text or "_____" in question_text):
            return None
        
        # Varsayılan cevap
        if not correct_answer:
            if subject == "english":
                correct_answer = "went"  # Common past tense example
            else:
                correct_answer = "answer"
        
        question_id = f"pdf_{hash(text)}_{page_num}"
        
        return {
            "id": question_id,
            "content": question_text,
            "question_type": "fill_blank",
            "difficulty_level": difficulty,
            "subject": subject,
            "correct_answer": correct_answer,
            "source_page": page_num,
            "original_text": text if preserve_formatting else None,
            "topic_category": self._determine_topic(question_text, subject),
            "estimated_time": self._estimate_time("fill_blank", difficulty)
        }
    
    async def _parse_open_ended_question(self, text: str, subject: str, difficulty: int, 
                                       page_num: int, preserve_formatting: bool) -> Optional[Dict[str, Any]]:
        """Açık uçlu soruyu parse et"""
        question_text = text.strip()
        
        if len(question_text) < 10:
            return None
        
        question_id = f"pdf_{hash(text)}_{page_num}"
        
        return {
            "id": question_id,
            "content": question_text,
            "question_type": "open_ended",
            "difficulty_level": difficulty,
            "subject": subject,
            "evaluation_criteria": self._get_evaluation_criteria(subject),
            "source_page": page_num,
            "original_text": text if preserve_formatting else None,
            "topic_category": self._determine_topic(question_text, subject),
            "estimated_time": self._estimate_time("open_ended", difficulty)
        }
    
    def _is_likely_correct_answer(self, option_text: str, subject: str) -> bool:
        """Doğru cevap olma olasılığını değerlendir"""
        option_lower = option_text.lower()
        
        if subject == "english":
            # İngilizce için common correct patterns
            correct_indicators = ["went", "have", "has", "was", "were", "did"]
            return any(indicator in option_lower for indicator in correct_indicators)
        elif subject == "math":
            # Matematik için sayısal cevaplar
            return bool(re.search(r'\d+', option_text))
        
        return False
    
    def _get_evaluation_criteria(self, subject: str) -> List[str]:
        """Değerlendirme kriterlerini al"""
        if subject == "english":
            return ["Grammar", "Vocabulary", "Content", "Organization"]
        elif subject == "math":
            return ["Accuracy", "Method", "Explanation", "Final Answer"]
        else:
            return ["Content", "Clarity", "Completeness"]
    
    async def _parse_single_question(self, text: str, subject: str, difficulty: int, 
                                   page_num: int, preserve_formatting: bool) -> Dict[str, Any]:
        """Tek bir soruyu parse et"""
        
        question_id = f"pdf_{hash(text)}_{page_num}"
        
        # Soru tipini belirle
        if "A)" in text and "B)" in text:
            question_type = "multiple_choice"
            # Seçenekleri çıkar
            options = []
            for line in text.split('\n'):
                if line.strip().startswith(('A)', 'B)', 'C)', 'D)')):
                    options.append(line.strip())
            
            # Soru metnini temizle
            content = text.split('\n')[0] if '\n' in text else text
            content = content.split('A)')[0].strip() if 'A)' in content else content
            
            return {
                "id": question_id,
                "content": content,
                "question_type": question_type,
                "difficulty_level": difficulty,
                "subject": subject,
                "options": options,
                "correct_answer": "A",  # Mock - gerçekte answer key'den çıkarılacak
                "source_page": page_num,
                "original_text": text if preserve_formatting else None,
                "topic_category": self._determine_topic(content, subject),
                "estimated_time": self._estimate_time(question_type, difficulty)
            }
        
        elif "Fill in" in text or "______" in text or "___" in text:
            question_type = "fill_blank"
            
            # Cevabı bul (parantez içinde veya ayrı satırda)
            correct_answer = "went"  # Mock
            if "(Answer:" in text:
                answer_part = text.split("(Answer:")[1].split(")")[0].strip()
                correct_answer = answer_part
            
            return {
                "id": question_id,
                "content": text.split("(Answer:")[0].strip() if "(Answer:" in text else text,
                "question_type": question_type,
                "difficulty_level": difficulty,
                "subject": subject,
                "correct_answer": correct_answer,
                "source_page": page_num,
                "original_text": text if preserve_formatting else None,
                "topic_category": self._determine_topic(text, subject),
                "estimated_time": self._estimate_time(question_type, difficulty)
            }
        
        elif "Write" in text or "Describe" in text:
            question_type = "open_ended"
            
            return {
                "id": question_id,
                "content": text.strip(),
                "question_type": question_type,
                "difficulty_level": difficulty,
                "subject": subject,
                "evaluation_criteria": ["Grammar", "Vocabulary", "Content"],
                "source_page": page_num,
                "original_text": text if preserve_formatting else None,
                "topic_category": self._determine_topic(text, subject),
                "estimated_time": self._estimate_time(question_type, difficulty)
            }
        
        return None
    
    def _determine_topic(self, content: str, subject: str) -> str:
        """Soru konusunu belirle"""
        content_lower = content.lower()
        
        if subject == "english":
            if "past tense" in content_lower or "went" in content_lower or "yesterday" in content_lower:
                return "past_tense"
            elif "present perfect" in content_lower or "have" in content_lower:
                return "present_perfect"
            elif "vocabulary" in content_lower:
                return "vocabulary"
            else:
                return "general_grammar"
        
        elif subject == "math":
            if any(op in content_lower for op in ["+", "add", "sum"]):
                return "addition"
            elif any(op in content_lower for op in ["-", "subtract", "minus"]):
                return "subtraction"
            elif any(op in content_lower for op in ["×", "*", "multiply"]):
                return "multiplication"
            elif any(op in content_lower for op in ["÷", "/", "divide"]):
                return "division"
            else:
                return "general_math"
        
        return "general"
    
    def _estimate_time(self, question_type: str, difficulty: int) -> int:
        """Tahmini çözüm süresi (saniye)"""
        base_times = {
            "multiple_choice": 30,
            "fill_blank": 45,
            "open_ended": 180,
            "true_false": 20
        }
        
        base_time = base_times.get(question_type, 60)
        difficulty_multiplier = 1 + (difficulty - 1) * 0.3
        
        return int(base_time * difficulty_multiplier)
    
    async def _validate_questions(self, questions: List[Dict], subject: str) -> List[Dict[str, Any]]:
        """Soruları doğrula ve kalite kontrol yap"""
        validated = []
        
        for question in questions:
            # Temel validasyon
            if not question.get("content") or len(question["content"].strip()) < 10:
                continue
            
            # Soru tipine göre validasyon
            if question["question_type"] == "multiple_choice":
                if not question.get("options") or len(question["options"]) < 2:
                    continue
            
            elif question["question_type"] == "fill_blank":
                if not question.get("correct_answer"):
                    continue
            
            # Kalite puanı hesapla
            quality_score = self._calculate_quality_score(question, subject)
            question["quality_score"] = quality_score
            
            # Minimum kalite eşiğini geç
            if quality_score >= 0.6:
                validated.append(question)
        
        return validated
    
    def _calculate_quality_score(self, question: Dict, subject: str) -> float:
        """Soru kalite puanı hesapla"""
        score = 0.0
        
        # İçerik uzunluğu
        content_length = len(question["content"])
        if content_length >= 20:
            score += 0.3
        elif content_length >= 10:
            score += 0.2
        
        # Soru işareti varlığı
        if "?" in question["content"]:
            score += 0.2
        
        # Seçenekler (multiple choice için)
        if question["question_type"] == "multiple_choice":
            options_count = len(question.get("options", []))
            if options_count >= 4:
                score += 0.3
            elif options_count >= 2:
                score += 0.2
        
        # Doğru cevap varlığı
        if question.get("correct_answer"):
            score += 0.2
        
        return min(score, 1.0)
    
    def _count_by_type(self, questions: List[Dict]) -> Dict[str, int]:
        """Soru tipine göre sayım"""
        counts = {}
        for question in questions:
            q_type = question["question_type"]
            counts[q_type] = counts.get(q_type, 0) + 1
        return counts
    
    def _count_by_difficulty(self, questions: List[Dict]) -> Dict[str, int]:
        """Zorluk seviyesine göre sayım"""
        counts = {}
        for question in questions:
            difficulty = str(question["difficulty_level"])
            counts[difficulty] = counts.get(difficulty, 0) + 1
        return counts