import httpx
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import structlog
from sqlalchemy.orm import Session
from app.crud.question import create_question
from app.schemas.question import QuestionCreate
from app.services.llm_service import llm_service

logger = structlog.get_logger()

class QuestionIngestionService:
    """
    Harici sitelerden soru içe aktarma servisi
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            timeout=30.0
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def ingest_from_khan_academy(self, subject: str, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Khan Academy'den soru içe aktar
        """
        try:
            # Khan Academy API endpoint'i (örnek)
            url = f"https://www.khanacademy.org/api/v1/exercises/{subject}/{topic}"
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            questions = []
            
            for item in data.get('exercises', [])[:limit]:
                question_data = {
                    'content': item.get('question', ''),
                    'options': item.get('choices', []),
                    'correct_answer': item.get('correct_answer', ''),
                    'difficulty_level': self._map_difficulty(item.get('difficulty', 'medium')),
                    'subject_id': self._get_subject_id(subject),
                    'question_type': 'multiple_choice',
                    'tags': [subject, topic],
                    'created_by': 1,  # System user
                    'is_active': True
                }
                
                # Batch işlemi: AI ile ön zenginleştirme
                if question_data['content']:
                    # 1. İlk zorluk analizi
                    difficulty_analysis = await llm_service.analyze_question_difficulty(
                        question_data['content'], subject
                    )
                    question_data['difficulty_level'] = difficulty_analysis.get('difficulty_level', 2)
                    question_data['required_knowledge'] = difficulty_analysis.get('required_knowledge', [subject])
                    
                    # 2. Temel ipucu (batch'te hazırla)
                    question_data['hint'] = await llm_service.generate_question_hint(
                        question_data['content'], subject
                    )
                    
                    # 3. Temel açıklama (batch'te hazırla)
                    question_data['explanation'] = await llm_service.generate_question_explanation(
                        question_data['content'], question_data['correct_answer'], subject
                    )
                    
                    # 4. Metadata ekle
                    question_data['metadata'] = {
                        'batch_processed': True,
                        'initial_difficulty': question_data['difficulty_level'],
                        'grade_level': difficulty_analysis.get('grade_level', '9-10'),
                        'solution_steps': difficulty_analysis.get('solution_steps', 3),
                        'ai_generated': True
                    }
                
                questions.append(question_data)
            
            return questions
            
        except Exception as e:
            logger.error(f"Khan Academy'den soru içe aktarma hatası: {str(e)}")
            return []
    
    async def ingest_from_quizlet(self, subject: str, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Quizlet'ten soru içe aktar
        """
        try:
            # Quizlet API endpoint'i (örnek)
            url = f"https://api.quizlet.com/2.0/sets/search"
            params = {
                'q': f"{subject} {topic}",
                'limit': limit
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            questions = []
            
            for set_data in data.get('sets', []):
                for card in set_data.get('terms', [])[:limit]:
                    question_data = {
                        'content': card.get('term', ''),
                        'options': [card.get('definition', '')] + self._generate_fake_options(),
                        'correct_answer': card.get('definition', ''),
                        'difficulty_level': 1,
                        'subject_id': self._get_subject_id(subject),
                        'question_type': 'multiple_choice',
                        'tags': [subject, topic],
                        'created_by': 1,
                        'is_active': True
                    }
                    
                    questions.append(question_data)
            
            return questions
            
        except Exception as e:
            logger.error(f"Quizlet'ten soru içe aktarma hatası: {str(e)}")
            return []
    
    async def scrape_from_website(self, url: str, subject: str, topic: str) -> List[Dict[str, Any]]:
        """
        Web sitesinden HTML scraping ile soru çıkar
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            questions = []
            
            # Soru elementlerini bul (site yapısına göre ayarlanmalı)
            question_elements = soup.find_all(['div', 'p'], class_=re.compile(r'question|soru|problem'))
            
            for element in question_elements:
                question_text = element.get_text(strip=True)
                if len(question_text) > 20:  # Minimum uzunluk kontrolü
                    
                    # Seçenekleri bul
                    options = []
                    option_elements = element.find_next_siblings(['div', 'li'], class_=re.compile(r'option|choice|secenek'))
                    
                    for opt_elem in option_elements:
                        options.append(opt_elem.get_text(strip=True))
                    
                    if not options:
                        options = self._generate_fake_options()
                    
                    question_data = {
                        'content': question_text,
                        'options': options,
                        'correct_answer': options[0] if options else '',  # Varsayılan
                        'difficulty_level': 2,
                        'subject_id': self._get_subject_id(subject),
                        'question_type': 'multiple_choice',
                        'tags': [subject, topic],
                        'created_by': 1,
                        'is_active': True
                    }
                    
                    questions.append(question_data)
            
            return questions
            
        except Exception as e:
            logger.error(f"Web sitesinden soru çıkarma hatası: {str(e)}")
            return []
    
    async def save_questions_to_database(self, db: Session, questions: List[Dict[str, Any]]) -> int:
        """
        Soruları veritabanına kaydet
        """
        saved_count = 0
        
        for question_data in questions:
            try:
                question_create = QuestionCreate(**question_data)
                create_question(db, question_create)
                saved_count += 1
            except Exception as e:
                logger.error(f"Soru kaydetme hatası: {str(e)}")
                continue
        
        return saved_count
    
    def _map_difficulty(self, difficulty: str) -> int:
        """Zorluk seviyesini sayısal değere çevir"""
        difficulty_map = {
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'kolay': 1,
            'orta': 2,
            'zor': 3
        }
        return difficulty_map.get(difficulty.lower(), 2)
    
    def _get_subject_id(self, subject: str) -> int:
        """Konu adından ID döndür"""
        subject_map = {
            'matematik': 1,
            'mathematics': 1,
            'math': 1,
            'fizik': 2,
            'physics': 2,
            'kimya': 3,
            'chemistry': 3,
            'biyoloji': 4,
            'biology': 4,
            'türkçe': 5,
            'turkish': 5,
            'ingilizce': 6,
            'english': 6
        }
        return subject_map.get(subject.lower(), 1)
    
    def _generate_fake_options(self) -> List[str]:
        """Sahte seçenekler üret"""
        return [
            "Seçenek A",
            "Seçenek B", 
            "Seçenek C"
        ]
    
    async def ingest_from_csv(self, file_path: str, subject: str) -> List[Dict[str, Any]]:
        """
        CSV dosyasından soru içe aktar
        """
        import csv
        
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    question_data = {
                        'content': row.get('question', ''),
                        'options': [
                            row.get('option_a', ''),
                            row.get('option_b', ''),
                            row.get('option_c', ''),
                            row.get('option_d', '')
                        ],
                        'correct_answer': row.get('correct_answer', ''),
                        'difficulty_level': int(row.get('difficulty', 2)),
                        'subject_id': self._get_subject_id(subject),
                        'question_type': 'multiple_choice',
                        'tags': [subject] + (row.get('tags', '').split(',') if row.get('tags') else []),
                        'created_by': 1,
                        'is_active': True
                    }
                    
                    questions.append(question_data)
            
            return questions
            
        except Exception as e:
            logger.error(f"CSV dosyasından soru içe aktarma hatası: {str(e)}")
            return []

# Singleton instance
# Global async client - Will be initialized in FastAPI startup
question_ingestion_service = None

async def get_question_ingestion_service():
    """Get or create question ingestion service singleton"""
    global question_ingestion_service
    if question_ingestion_service is None:
        question_ingestion_service = QuestionIngestionService()
    return question_ingestion_service

async def close_question_ingestion_service():
    """Close question ingestion service and cleanup resources"""
    global question_ingestion_service
    if question_ingestion_service is not None:
        await question_ingestion_service.client.aclose()
        question_ingestion_service = None 