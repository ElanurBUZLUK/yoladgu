"""
Test cases for database models and repositories.
"""

import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, MathItem, EnglishItem, Attempt
from app.db.repositories.user import user_repository
from app.db.repositories.item import math_item_repository, english_item_repository
from app.db.repositories.attempt import attempt_repository


class TestUserModel:
    """Test User model and repository."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession):
        """Test user creation."""
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "hashed_password": "hashed_password",
            "tenant_id": "tenant_1",
            "lang": "tr",
            "role": "student"
        }
        
        user = await user_repository.create(db_session, obj_in=user_data)
        
        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.lang == "tr"
        assert user.role == "student"
        assert user.theta_math == 0.0
        assert user.theta_en == 0.0
    
    @pytest.mark.asyncio
    async def test_update_theta(self, db_session: AsyncSession):
        """Test theta value updates."""
        # Create user first
        user_data = {
            "username": "theta_user",
            "email": "theta@example.com",
            "hashed_password": "hashed_password",
            "tenant_id": "tenant_1"
        }
        user = await user_repository.create(db_session, obj_in=user_data)
        
        # Update theta values
        updated_user = await user_repository.update_theta(
            db_session, user.id, theta_math=0.5, theta_en=-0.2
        )
        
        assert updated_user.theta_math == 0.5
        assert updated_user.theta_en == -0.2
    
    @pytest.mark.asyncio
    async def test_update_error_profile(self, db_session: AsyncSession):
        """Test error profile updates."""
        # Create user first
        user_data = {
            "username": "error_user",
            "email": "error@example.com",
            "hashed_password": "hashed_password",
            "tenant_id": "tenant_1"
        }
        user = await user_repository.create(db_session, obj_in=user_data)
        
        # Update math error profile
        error_updates = {"sign_error": 0.3, "ratio_misuse": 0.2}
        updated_user = await user_repository.update_error_profile(
            db_session, user.id, "math", error_updates
        )
        
        assert updated_user.error_profile_math["sign_error"] == 0.3
        assert updated_user.error_profile_math["ratio_misuse"] == 0.2


class TestMathItemModel:
    """Test MathItem model and repository."""
    
    @pytest.mark.asyncio
    async def test_create_math_item(self, db_session: AsyncSession):
        """Test math item creation."""
        item_data = {
            "tenant_id": "tenant_1",
            "stem": "Solve: 2x + 3 = 7",
            "answer_key": "x=2",
            "skills": ["linear_equation", "algebra"],
            "difficulty_a": 1.2,
            "difficulty_b": 0.3,
            "lang": "en"
        }
        
        item = await math_item_repository.create(db_session, obj_in=item_data)
        
        assert item.stem == "Solve: 2x + 3 = 7"
        assert item.answer_key == "x=2"
        assert "linear_equation" in item.skills
        assert item.difficulty_b == 0.3
    
    @pytest.mark.asyncio
    async def test_search_by_skills(self, db_session: AsyncSession):
        """Test searching math items by skills."""
        # Create test items
        item1_data = {
            "tenant_id": "tenant_1",
            "stem": "Linear equation 1",
            "answer_key": "x=1",
            "skills": ["linear_equation"],
            "lang": "tr"
        }
        item2_data = {
            "tenant_id": "tenant_1",
            "stem": "Quadratic equation 1",
            "answer_key": "x=2",
            "skills": ["quadratic_equation"],
            "lang": "tr"
        }
        
        await math_item_repository.create(db_session, obj_in=item1_data)
        await math_item_repository.create(db_session, obj_in=item2_data)
        
        # Search by skills
        results = await math_item_repository.search_by_skills(
            db_session, ["linear_equation"], "tenant_1", "tr"
        )
        
        assert len(results) >= 1
        assert any("linear_equation" in item.skills for item in results)


class TestEnglishItemModel:
    """Test EnglishItem model and repository."""
    
    @pytest.mark.asyncio
    async def test_create_english_item(self, db_session: AsyncSession):
        """Test English item creation."""
        item_data = {
            "tenant_id": "tenant_1",
            "passage": "I went __ the store yesterday.",
            "blanks": [
                {
                    "span": "__",
                    "answer": "to",
                    "distractors": ["at", "in", "on"],
                    "skill_tag": "prepositions"
                }
            ],
            "level_cefr": "A2",
            "error_tags": ["prepositions"],
            "lang": "en"
        }
        
        item = await english_item_repository.create(db_session, obj_in=item_data)
        
        assert "went __ the store" in item.passage
        assert item.level_cefr == "A2"
        assert "prepositions" in item.error_tags
        assert len(item.blanks) == 1
        assert item.blanks[0]["answer"] == "to"


class TestAttemptModel:
    """Test Attempt model and repository."""
    
    @pytest.mark.asyncio
    async def test_create_attempt(self, db_session: AsyncSession):
        """Test attempt creation."""
        # Create user and item first
        user_data = {
            "username": "attempt_user",
            "email": "attempt@example.com",
            "hashed_password": "hashed_password",
            "tenant_id": "tenant_1"
        }
        user = await user_repository.create(db_session, obj_in=user_data)
        
        item_data = {
            "tenant_id": "tenant_1",
            "stem": "Test question",
            "answer_key": "42",
            "skills": ["test_skill"],
            "lang": "tr"
        }
        item = await math_item_repository.create(db_session, obj_in=item_data)
        
        # Create attempt
        attempt = await attempt_repository.create_attempt(
            db_session,
            user_id=user.id,
            item_id=item.id,
            item_type="math",
            answer="42",
            correct=True,
            time_ms=5000
        )
        
        assert attempt.user_id == user.id
        assert attempt.item_id == item.id
        assert attempt.correct == True
        assert attempt.time_ms == 5000
    
    @pytest.mark.asyncio
    async def test_get_user_performance_stats(self, db_session: AsyncSession):
        """Test user performance statistics."""
        # Create user
        user_data = {
            "username": "stats_user",
            "email": "stats@example.com",
            "hashed_password": "hashed_password",
            "tenant_id": "tenant_1"
        }
        user = await user_repository.create(db_session, obj_in=user_data)
        
        # Create some attempts
        for i in range(5):
            await attempt_repository.create_attempt(
                db_session,
                user_id=user.id,
                item_id=f"item_{i}",
                item_type="math",
                answer=str(i),
                correct=i % 2 == 0,  # 3 correct, 2 incorrect
                time_ms=1000 * (i + 1)
            )
        
        # Get performance stats
        stats = await attempt_repository.get_user_performance_stats(
            db_session, user.id, "math"
        )
        
        assert stats["total_attempts"] == 5
        assert stats["correct_attempts"] == 3
        assert stats["success_rate"] == 0.6