#!/usr/bin/env python3
"""
Test script for English cloze generation pipeline.
"""

import logging
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cloze_generation_basic():
    """Test basic cloze question generation."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing basic cloze generation...")
        
        # Generate a simple A1 level question
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["prepositions", "articles"],
            topic="daily_life"
        )
        
        assert question is not None, "Question should not be None"
        assert question.passage, "Passage should not be empty"
        assert len(question.blanks) > 0, "Should have at least one blank"
        assert question.level_cefr == CEFRLevel.A1, "CEFR level should match"
        
        logger.info(f"✅ Generated question with {len(question.blanks)} blanks")
        logger.info(f"✅ Passage preview: {question.passage[:100]}...")
        
        # Check that blanks have required fields
        for i, blank in enumerate(question.blanks):
            assert blank.answer, f"Blank {i} should have an answer"
            assert blank.skill_tag, f"Blank {i} should have a skill tag"
            assert len(blank.distractors) > 0, f"Blank {i} should have distractors"
            logger.info(f"✅ Blank {i+1}: '{blank.answer}' ({blank.skill_tag})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic cloze generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_personalized_generation():
    """Test personalized cloze generation based on error profile."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing personalized cloze generation...")
        
        # Create personalization with high preposition errors
        personalization = {
            "error_profile": {
                "prepositions": 0.7,
                "articles": 0.2,
                "subject_verb_agreement": 0.1
            },
            "common_errors": {
                "prepositions": ["in", "on"]  # User commonly confuses these
            }
        }
        
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A2,
            target_error_tags=["prepositions", "articles", "subject_verb_agreement"],
            personalization=personalization
        )
        
        assert question is not None, "Personalized question should not be None"
        assert len(question.blanks) > 0, "Should have blanks"
        
        # Check that prepositions are prioritized (due to high error rate)
        preposition_blanks = [b for b in question.blanks if b.skill_tag == "prepositions"]
        assert len(preposition_blanks) > 0, "Should have preposition blanks due to high error rate"
        
        logger.info(f"✅ Generated personalized question with {len(question.blanks)} blanks")
        logger.info(f"✅ Preposition blanks: {len(preposition_blanks)}")
        
        # Check that common errors appear as distractors
        for blank in preposition_blanks:
            common_distractors = set(blank.distractors) & {"in", "on"}
            if common_distractors:
                logger.info(f"✅ Found personalized distractors: {common_distractors}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Personalized generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_cefr_levels():
    """Test generation for different CEFR levels."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing different CEFR levels...")
        
        levels = [CEFRLevel.A1, CEFRLevel.A2, CEFRLevel.B1]
        
        for level in levels:
            question = english_cloze_generator.generate_cloze_question(
                level_cefr=level,
                target_error_tags=["prepositions", "articles"]
            )
            
            assert question is not None, f"Question for {level.value} should not be None"
            assert question.level_cefr == level, f"CEFR level should match {level.value}"
            
            logger.info(f"✅ {level.value}: Generated question with {len(question.blanks)} blanks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CEFR level testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_passage():
    """Test generation with custom passage text."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing custom passage generation...")
        
        custom_passage = "I go to school at 8 o'clock. I have a book in my bag. The teacher is very nice. She goes to the library on Monday."
        
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["prepositions", "articles", "subject_verb_agreement"],
            passage_text=custom_passage
        )
        
        assert question is not None, "Custom passage question should not be None"
        assert len(question.blanks) > 0, "Should find blanks in custom passage"
        
        logger.info(f"✅ Custom passage: Generated {len(question.blanks)} blanks")
        logger.info(f"✅ Blanked passage: {question.passage}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom passage testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distractor_generation():
    """Test distractor generation quality."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing distractor generation...")
        
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["prepositions", "articles"]
        )
        
        assert len(question.blanks) > 0, "Should have blanks to test"
        
        for i, blank in enumerate(question.blanks):
            # Check distractor count
            assert len(blank.distractors) >= 2, f"Blank {i} should have at least 2 distractors"
            assert len(blank.distractors) <= 3, f"Blank {i} should have at most 3 distractors"
            
            # Check that answer is not in distractors
            assert blank.answer not in blank.distractors, f"Answer '{blank.answer}' should not be in distractors"
            
            # Check for duplicate distractors
            assert len(set(blank.distractors)) == len(blank.distractors), f"Blank {i} should not have duplicate distractors"
            
            logger.info(f"✅ Blank {i+1}: Answer='{blank.answer}', Distractors={blank.distractors}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Distractor generation testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rationale_generation():
    """Test rationale generation for answers."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing rationale generation...")
        
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["prepositions", "articles"]
        )
        
        assert len(question.blanks) > 0, "Should have blanks to test"
        
        for i, blank in enumerate(question.blanks):
            assert blank.rationale, f"Blank {i} should have a rationale"
            assert len(blank.rationale) > 10, f"Rationale {i} should be meaningful"
            
            logger.info(f"✅ Blank {i+1} rationale: {blank.rationale}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Rationale generation testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_question_validation():
    """Test cloze question validation."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing question validation...")
        
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["prepositions", "articles"]
        )
        
        validation = english_cloze_generator.validate_cloze_question(question)
        
        assert "valid" in validation, "Validation should include 'valid' field"
        assert "quality_score" in validation, "Validation should include quality score"
        assert "metrics" in validation, "Validation should include metrics"
        
        logger.info(f"✅ Question valid: {validation['valid']}")
        logger.info(f"✅ Quality score: {validation['quality_score']:.2f}")
        logger.info(f"✅ Metrics: {validation['metrics']}")
        
        if validation["issues"]:
            logger.info(f"⚠️  Issues found: {validation['issues']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Question validation testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting English cloze generation tests...")
    
    tests = [
        ("Basic Cloze Generation", test_cloze_generation_basic),
        ("Personalized Generation", test_personalized_generation),
        ("Different CEFR Levels", test_different_cefr_levels),
        ("Custom Passage", test_custom_passage),
        ("Distractor Generation", test_distractor_generation),
        ("Rationale Generation", test_rationale_generation),
        ("Question Validation", test_question_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! English cloze generation is working correctly.")
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()