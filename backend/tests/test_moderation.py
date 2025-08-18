from app.domains.english.moderation import ContentModerator
import pytest

@pytest.mark.asyncio
async def test_moderation_approval():
    moderator = ContentModerator()
    question = {
        "stem": "This is a valid question about mathematics.",
        "options": ["A", "B", "C"],
        "target_cefr": "B1"
    }
    result = await moderator.moderate_question(question_content=question["stem"], options=question["options"], correct_answer="A", difficulty_level=1, topic="general")
    assert result["is_appropriate"] is True

@pytest.mark.asyncio
async def test_moderation_rejection_toxicity():
    moderator = ContentModerator()
    question = {
        "stem": "This is a bad question!",
        "options": ["A", "B", "C"],
        "target_cefr": "B1"
    }
    result = await moderator.moderate_question(question_content=question["stem"], options=question["options"], correct_answer="A", difficulty_level=1, topic="general")
    assert result["is_appropriate"] is False