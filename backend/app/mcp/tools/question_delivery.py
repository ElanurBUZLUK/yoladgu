from typing import Dict, Any, List, Optional
from .base import BaseMCPTool


class QuestionDeliveryTool(BaseMCPTool):
    """Soruları öğrenciye uygun formatta hazırlama ve gönderme için MCP tool"""
    
    def get_name(self) -> str:
        return "deliver_question_to_student"
    
    def get_description(self) -> str:
        return "Soruyu öğrenciye uygun formatta hazırlar ve API üzerinden gönderir"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question_data": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "question_type": {"type": "string"},
                        "difficulty_level": {"type": "integer"},
                        "subject": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "correct_answer": {"type": "string"},
                        "images": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "original_text": {"type": "string"},
                        "source_page": {"type": "integer"}
                    },
                    "required": ["id", "content", "question_type", "subject"]
                },
                "user_id": {
                    "type": "string",
                    "description": "Öğrenci ID'si"
                },
                "learning_style": {
                    "type": "string",
                    "enum": ["visual", "auditory", "kinesthetic", "mixed"],
                    "default": "mixed",
                    "description": "Öğrenme stili"
                },
                "delivery_format": {
                    "type": "string",
                    "enum": ["web", "mobile", "pdf_viewer", "interactive"],
                    "default": "web",
                    "description": "Sunum formatı"
                },
                "accessibility_options": {
                    "type": "object",
                    "properties": {
                        "high_contrast": {"type": "boolean", "default": False},
                        "large_text": {"type": "boolean", "default": False},
                        "screen_reader": {"type": "boolean", "default": False},
                        "audio_support": {"type": "boolean", "default": False}
                    }
                }
            },
            "required": ["question_data", "user_id"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Soru delivery mantığı"""
        
        question_data = arguments["question_data"]
        user_id = arguments["user_id"]
        learning_style = arguments.get("learning_style", "mixed")
        delivery_format = arguments.get("delivery_format", "web")
        accessibility_options = arguments.get("accessibility_options", {})
        
        try:
            # Öğrenme stiline göre adapte et
            adapted_question = await self._adapt_for_learning_style(
                question_data, learning_style
            )
            
            # Delivery formatına göre hazırla
            formatted_question = await self._format_for_delivery(
                adapted_question, delivery_format, accessibility_options
            )
            
            # Interaktif elementler ekle
            interactive_elements = await self._add_interactive_elements(
                formatted_question, learning_style, delivery_format
            )
            
            # Delivery metadata'sı oluştur
            delivery_metadata = {
                "user_id": user_id,
                "learning_style": learning_style,
                "delivery_format": delivery_format,
                "accessibility_enabled": any(accessibility_options.values()),
                "delivery_timestamp": "2024-01-01T00:00:00Z",
                "estimated_completion_time": formatted_question.get("estimated_time", 60)
            }
            
            return {
                "success": True,
                "delivered_question": formatted_question,
                "interactive_elements": interactive_elements,
                "delivery_metadata": delivery_metadata,
                "delivery_instructions": self._get_delivery_instructions(
                    learning_style, delivery_format
                )
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Question delivery failed: {str(e)}",
                "question_id": question_data.get("id"),
                "user_id": user_id
            }
    
    async def _adapt_for_learning_style(self, question_data: Dict[str, Any], learning_style: str) -> Dict[str, Any]:
        """Öğrenme stiline göre soru adaptasyonu"""
        
        adapted = question_data.copy()
        
        if learning_style == "visual":
            # Görsel öğrenciler için
            adapted["presentation_style"] = {
                "use_colors": True,
                "highlight_keywords": True,
                "include_diagrams": True,
                "visual_cues": True,
                "color_scheme": "high_contrast"
            }
            
            # Anahtar kelimeleri vurgula
            content = adapted["content"]
            adapted["content"] = self._add_visual_highlights(content)
            
            # Görsel ipuçları ekle
            if adapted["question_type"] == "multiple_choice":
                adapted["visual_options"] = self._create_visual_options(
                    adapted.get("options", [])
                )
        
        elif learning_style == "auditory":
            # İşitsel öğrenciler için
            adapted["presentation_style"] = {
                "text_to_speech": True,
                "audio_instructions": True,
                "verbal_feedback": True,
                "sound_effects": True
            }
            
            # Sesli okuma metni ekle
            adapted["audio_content"] = {
                "question_audio": self._generate_audio_text(adapted["content"]),
                "instruction_audio": "Please listen to the question and select your answer.",
                "option_audio": [self._generate_audio_text(opt) for opt in adapted.get("options", [])]
            }
        
        elif learning_style == "kinesthetic":
            # Kinestetik öğrenciler için
            adapted["presentation_style"] = {
                "interactive_elements": True,
                "drag_and_drop": True,
                "step_by_step": True,
                "hands_on_activities": True
            }
            
            # İnteraktif elementler
            if adapted["question_type"] == "fill_blank":
                adapted["interactive_input"] = {
                    "type": "drag_drop_words",
                    "word_bank": self._create_word_bank(adapted["content"]),
                    "drop_zones": self._identify_drop_zones(adapted["content"])
                }
        
        else:  # mixed
            # Karma yaklaşım
            adapted["presentation_style"] = {
                "multiple_formats": True,
                "choice_of_interaction": True,
                "flexible_presentation": True
            }
        
        return adapted
    
    async def _format_for_delivery(self, question_data: Dict[str, Any], delivery_format: str, 
                                 accessibility_options: Dict[str, Any]) -> Dict[str, Any]:
        """Delivery formatına göre formatlama"""
        
        formatted = question_data.copy()
        
        if delivery_format == "web":
            # Web formatı
            formatted["html_content"] = self._generate_html_content(question_data)
            formatted["css_classes"] = self._get_css_classes(question_data, accessibility_options)
            formatted["javascript_handlers"] = self._get_js_handlers(question_data)
        
        elif delivery_format == "mobile":
            # Mobil formatı
            formatted["mobile_layout"] = {
                "responsive": True,
                "touch_friendly": True,
                "swipe_gestures": True,
                "large_buttons": True
            }
            formatted["mobile_content"] = self._optimize_for_mobile(question_data)
        
        elif delivery_format == "pdf_viewer":
            # PDF viewer formatı
            formatted["pdf_annotations"] = self._create_pdf_annotations(question_data)
            formatted["overlay_elements"] = self._create_overlay_elements(question_data)
        
        elif delivery_format == "interactive":
            # İnteraktif format
            formatted["interactive_components"] = self._create_interactive_components(question_data)
            formatted["gamification"] = self._add_gamification_elements(question_data)
        
        # Accessibility adaptasyonları
        if accessibility_options.get("high_contrast"):
            formatted["accessibility"]["high_contrast"] = True
            formatted["color_scheme"] = "high_contrast"
        
        if accessibility_options.get("large_text"):
            formatted["accessibility"]["large_text"] = True
            formatted["font_size"] = "large"
        
        if accessibility_options.get("screen_reader"):
            formatted["accessibility"]["screen_reader"] = True
            formatted["aria_labels"] = self._generate_aria_labels(question_data)
        
        return formatted
    
    async def _add_interactive_elements(self, question_data: Dict[str, Any], 
                                      learning_style: str, delivery_format: str) -> Dict[str, Any]:
        """İnteraktif elementler ekle"""
        
        elements = {
            "progress_indicator": {
                "type": "progress_bar",
                "show_percentage": True,
                "animated": True
            },
            "feedback_system": {
                "immediate_feedback": True,
                "hint_system": True,
                "explanation_on_wrong": True
            },
            "navigation": {
                "previous_button": True,
                "next_button": True,
                "skip_option": True,
                "bookmark": True
            }
        }
        
        question_type = question_data.get("question_type")
        
        if question_type == "multiple_choice":
            elements["answer_selection"] = {
                "type": "radio_buttons",
                "hover_effects": True,
                "selection_animation": True,
                "keyboard_navigation": True
            }
        
        elif question_type == "fill_blank":
            elements["text_input"] = {
                "type": "text_field",
                "autocomplete": True,
                "spell_check": True,
                "word_suggestions": True
            }
        
        elif question_type == "open_ended":
            elements["text_editor"] = {
                "type": "rich_text_editor",
                "word_count": True,
                "grammar_check": True,
                "save_draft": True
            }
        
        # Learning style specific elements
        if learning_style == "kinesthetic":
            elements["kinesthetic_features"] = {
                "drag_drop": True,
                "touch_gestures": True,
                "interactive_animations": True
            }
        
        return elements
    
    def _add_visual_highlights(self, content: str) -> str:
        """Görsel vurgular ekle"""
        # Anahtar kelimeleri HTML span ile vurgula
        keywords = ["past tense", "present perfect", "grammar", "vocabulary"]
        
        highlighted_content = content
        for keyword in keywords:
            highlighted_content = highlighted_content.replace(
                keyword, 
                f'<span class="highlight-keyword">{keyword}</span>'
            )
        
        return highlighted_content
    
    def _create_visual_options(self, options: List[str]) -> List[Dict[str, Any]]:
        """Görsel seçenekler oluştur"""
        visual_options = []
        
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        
        for i, option in enumerate(options):
            visual_options.append({
                "text": option,
                "color": colors[i % len(colors)],
                "icon": f"option_{chr(65 + i)}",  # A, B, C, D
                "hover_effect": "scale_up"
            })
        
        return visual_options
    
    def _generate_audio_text(self, text: str) -> str:
        """Sesli okuma için metin hazırla"""
        # HTML etiketlerini temizle
        clean_text = text.replace("<span class=\"highlight-keyword\">", "").replace("</span>", "")
        
        # Özel karakterleri sesli okuma için uyarla
        clean_text = clean_text.replace("_____", "blank")
        clean_text = clean_text.replace("___", "blank")
        
        return clean_text
    
    def _create_word_bank(self, content: str) -> List[str]:
        """Kelime bankası oluştur"""
        # Basit kelime bankası - gerçekte daha sofistike olacak
        common_words = ["went", "go", "going", "goes", "have", "has", "had", "was", "were"]
        return common_words[:5]  # İlk 5 kelime
    
    def _identify_drop_zones(self, content: str) -> List[Dict[str, Any]]:
        """Drop zone'ları belirle"""
        drop_zones = []
        
        # Boşlukları bul
        import re
        blanks = re.finditer(r'_{3,}', content)
        
        for i, blank in enumerate(blanks):
            drop_zones.append({
                "id": f"drop_zone_{i}",
                "position": blank.start(),
                "width": blank.end() - blank.start(),
                "accepts": ["word"]
            })
        
        return drop_zones
    
    def _generate_html_content(self, question_data: Dict[str, Any]) -> str:
        """HTML içerik oluştur"""
        question_type = question_data.get("question_type")
        content = question_data.get("content", "")
        
        if question_type == "multiple_choice":
            options_html = ""
            for i, option in enumerate(question_data.get("options", [])):
                options_html += f'''
                <div class="option-container">
                    <input type="radio" id="option_{i}" name="answer" value="{chr(65+i)}">
                    <label for="option_{i}" class="option-label">{option}</label>
                </div>
                '''
            
            return f'''
            <div class="question-container">
                <div class="question-content">{content}</div>
                <div class="options-container">{options_html}</div>
            </div>
            '''
        
        elif question_type == "fill_blank":
            return f'''
            <div class="question-container">
                <div class="question-content">{content}</div>
                <div class="input-container">
                    <input type="text" class="fill-blank-input" placeholder="Your answer...">
                </div>
            </div>
            '''
        
        else:  # open_ended
            return f'''
            <div class="question-container">
                <div class="question-content">{content}</div>
                <div class="textarea-container">
                    <textarea class="open-ended-textarea" placeholder="Write your answer here..." rows="5"></textarea>
                </div>
            </div>
            '''
    
    def _get_css_classes(self, question_data: Dict[str, Any], accessibility_options: Dict[str, Any]) -> List[str]:
        """CSS sınıfları al"""
        classes = ["question-base"]
        
        classes.append(f"question-type-{question_data.get('question_type', 'unknown')}")
        classes.append(f"difficulty-{question_data.get('difficulty_level', 1)}")
        classes.append(f"subject-{question_data.get('subject', 'general')}")
        
        if accessibility_options.get("high_contrast"):
            classes.append("high-contrast")
        
        if accessibility_options.get("large_text"):
            classes.append("large-text")
        
        return classes
    
    def _get_js_handlers(self, question_data: Dict[str, Any]) -> Dict[str, str]:
        """JavaScript event handler'ları"""
        return {
            "onAnswerSelect": "handleAnswerSelection",
            "onSubmit": "handleQuestionSubmit",
            "onHintRequest": "showHint",
            "onTimeUpdate": "updateTimer"
        }
    
    def _optimize_for_mobile(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mobil için optimize et"""
        return {
            "touch_targets": "large",
            "font_size": "responsive",
            "layout": "single_column",
            "gestures_enabled": True
        }
    
    def _create_pdf_annotations(self, question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """PDF annotasyonları oluştur"""
        return [
            {
                "type": "highlight",
                "color": "yellow",
                "text": "Question area"
            }
        ]
    
    def _create_overlay_elements(self, question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Overlay elementleri oluştur"""
        return [
            {
                "type": "answer_box",
                "position": {"x": 100, "y": 200},
                "size": {"width": 300, "height": 50}
            }
        ]
    
    def _create_interactive_components(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """İnteraktif bileşenler oluştur"""
        return {
            "animations": True,
            "sound_effects": True,
            "progress_tracking": True,
            "adaptive_hints": True
        }
    
    def _add_gamification_elements(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gamification elementleri ekle"""
        return {
            "points_system": True,
            "badges": ["first_attempt", "perfect_score"],
            "leaderboard": False,  # Privacy için kapalı
            "achievements": True
        }
    
    def _generate_aria_labels(self, question_data: Dict[str, Any]) -> Dict[str, str]:
        """ARIA etiketleri oluştur"""
        return {
            "question": f"Question: {question_data.get('content', '')}",
            "options": "Answer options",
            "submit": "Submit your answer",
            "hint": "Get a hint for this question"
        }
    
    def _get_delivery_instructions(self, learning_style: str, delivery_format: str) -> Dict[str, str]:
        """Delivery talimatları"""
        instructions = {
            "visual": "Use colors and visual cues to highlight important information",
            "auditory": "Enable text-to-speech and audio feedback",
            "kinesthetic": "Provide interactive elements and hands-on activities",
            "mixed": "Offer multiple interaction methods"
        }
        
        return {
            "learning_style_instruction": instructions.get(learning_style, instructions["mixed"]),
            "format_instruction": f"Optimized for {delivery_format} delivery",
            "accessibility_note": "Accessibility options can be enabled in settings"
        }