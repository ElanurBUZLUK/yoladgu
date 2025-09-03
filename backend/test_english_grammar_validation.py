#!/usr/bin/env python3
"""
Test script for English grammar validation and ambiguity checking.
"""

import logging
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_grammar_validation():
    """Test grammar validation functionality."""
    try:
        from app.services.english_generation_service import GrammarValidator
        
        logger.info("Testing grammar validation...")
        
        validator = GrammarValidator()
        
        # Test correct grammar
        correct_text = "I go to school at 8 o'clock. She has a book. They are students."
        result = validator.validate_grammar(correct_text)
        
        assert result["valid"] == True, "Correct grammar should be valid"
        assert len(result["errors"]) == 0, "Correct grammar should have no errors"
        assert result["score"] >= 0.8, "Correct grammar should have high score"
        
        logger.info(f"✅ Correct grammar: Valid={result['valid']}, Score={result['score']:.2f}")
        
        # Test incorrect grammar
        incorrect_text = "He go to school. She have an book. They goes home."
        result = validator.validate_grammar(incorrect_text)
        
        logger.info(f"✅ Incorrect grammar: Valid={result['valid']}, Errors={len(result['errors'])}")
        
        for error in result["errors"]:
            logger.info(f"  - {error['description']} at '{error['text']}'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Grammar validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_answer_guarantee():
    """Test single answer guarantee checking."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel, GrammarValidator
        )
        
        logger.info("Testing single answer guarantee...")
        
        validator = GrammarValidator()
        
        # Generate a question
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["articles", "prepositions"]
        )
        
        # Check single answer guarantee
        result = validator.check_single_answer_guarantee(question)
        
        logger.info(f"✅ Single answer guaranteed: {result['guaranteed']}")
        
        if not result["guaranteed"]:
            logger.info(f"  Ambiguous blanks: {len(result['ambiguous_blanks'])}")
            for issue in result["issues"]:
                logger.info(f"  - {issue}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Single answer guarantee test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ambiguity_detection():
    """Test ambiguity detection functionality."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel, GrammarValidator, ClozeQuestion, ClozeBlank
        )
        
        logger.info("Testing ambiguity detection...")
        
        validator = GrammarValidator()
        
        # Create a potentially ambiguous question
        ambiguous_passage = "I drink water every day. She is in hospital. They make homework."
        
        # Create mock blanks for testing
        blanks = [
            ClozeBlank(
                span="water",
                answer="water",
                distractors=["a water", "the water"],
                skill_tag="articles",
                position=8
            ),
            ClozeBlank(
                span="in",
                answer="in",
                distractors=["at", "on"],
                skill_tag="prepositions",
                position=30
            ),
            ClozeBlank(
                span="make",
                answer="make",
                distractors=["do"],
                skill_tag="collocations",
                position=50
            )
        ]
        
        question = ClozeQuestion(
            passage=ambiguous_passage,
            blanks=blanks,
            level_cefr=CEFRLevel.A2,
            error_tags=["articles", "prepositions", "collocations"]
        )
        
        # Detect ambiguity
        result = validator.detect_ambiguity(question)
        
        logger.info(f"✅ Ambiguity detected: {result['ambiguous']}")
        logger.info(f"✅ Severity: {result['severity']}")
        logger.info(f"✅ Issues found: {len(result['ambiguity_issues'])}")
        
        for issue in result["ambiguity_issues"]:
            logger.info(f"  - {issue['type']}: {issue['description']}")
        
        if result["recommendations"]:
            logger.info("✅ Recommendations:")
            for rec in result["recommendations"]:
                logger.info(f"  - {rec}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ambiguity detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_validation():
    """Test comprehensive validation with all checks."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing comprehensive validation...")
        
        # Generate a question
        question = english_cloze_generator.generate_cloze_question(
            level_cefr=CEFRLevel.A1,
            target_error_tags=["articles", "prepositions", "subject_verb_agreement"]
        )
        
        # Comprehensive validation
        validation = english_cloze_generator.validate_cloze_question(question)
        
        logger.info(f"✅ Overall valid: {validation['valid']}")
        logger.info(f"✅ Quality score: {validation['quality_score']:.2f}")
        
        # Grammar validation results
        grammar = validation["grammar_validation"]
        logger.info(f"✅ Grammar valid: {grammar['valid']}, Score: {grammar['score']:.2f}")
        
        # Single answer check
        single_answer = validation["single_answer_check"]
        logger.info(f"✅ Single answer guaranteed: {single_answer['guaranteed']}")
        
        # Ambiguity analysis
        ambiguity = validation["ambiguity_analysis"]
        logger.info(f"✅ Ambiguity severity: {ambiguity['severity']}")
        
        # Issues
        if validation["issues"]:
            logger.info("⚠️  Issues found:")
            for issue in validation["issues"]:
                logger.info(f"  - {issue}")
        
        # Metrics
        metrics = validation["metrics"]
        logger.info(f"✅ Metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validated_generation():
    """Test validated question generation with ambiguity resolution."""
    try:
        from app.services.english_generation_service import (
            english_cloze_generator, CEFRLevel
        )
        
        logger.info("Testing validated question generation...")
        
        # Generate validated question
        question, validation = english_cloze_generator.generate_validated_cloze_question(
            level_cefr=CEFRLevel.A2,
            target_error_tags=["prepositions", "articles"],
            max_attempts=3
        )
        
        assert question is not None, "Should generate a question"
        assert validation is not None, "Should provide validation results"
        
        logger.info(f"✅ Generated validated question:")
        logger.info(f"  - Valid: {validation['valid']}")
        logger.info(f"  - Quality score: {validation['quality_score']:.2f}")
        logger.info(f"  - Blanks: {len(question.blanks)}")
        logger.info(f"  - Grammar score: {validation['metrics']['grammar_score']:.2f}")
        logger.info(f"  - Single answer guaranteed: {validation['metrics']['single_answer_guaranteed']}")
        logger.info(f"  - Ambiguity severity: {validation['metrics']['ambiguity_severity']}")
        
        # Show passage preview
        logger.info(f"  - Passage preview: {question.passage[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validated generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ambiguity_resolution():
    """Test ambiguity resolution functionality."""
    try:
        from app.services.english_generation_service import (
            GrammarValidator, ClozeQuestion, ClozeBlank, CEFRLevel
        )
        
        logger.info("Testing ambiguity resolution...")
        
        validator = GrammarValidator()
        
        # Create an ambiguous question
        ambiguous_passage = "I drink water. She is at school."
        
        blanks = [
            ClozeBlank(
                span="water",
                answer="water",
                distractors=["a water", "the water"],
                skill_tag="articles",
                position=8
            )
        ]
        
        question = ClozeQuestion(
            passage=ambiguous_passage,
            blanks=blanks,
            level_cefr=CEFRLevel.A1,
            error_tags=["articles"]
        )
        
        # Detect ambiguity
        ambiguity_results = validator.detect_ambiguity(question)
        
        logger.info(f"✅ Original ambiguity: {ambiguity_results['ambiguous']}")
        
        # Resolve ambiguity
        resolved_question = validator.resolve_ambiguity(question, ambiguity_results)
        
        logger.info(f"✅ Original passage: {question.passage}")
        logger.info(f"✅ Resolved passage: {resolved_question.passage}")
        
        # Check if ambiguity was reduced
        new_ambiguity = validator.detect_ambiguity(resolved_question)
        logger.info(f"✅ Resolved ambiguity: {new_ambiguity['ambiguous']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ambiguity resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting English grammar validation and ambiguity checking tests...")
    
    tests = [
        ("Grammar Validation", test_grammar_validation),
        ("Single Answer Guarantee", test_single_answer_guarantee),
        ("Ambiguity Detection", test_ambiguity_detection),
        ("Comprehensive Validation", test_comprehensive_validation),
        ("Validated Generation", test_validated_generation),
        ("Ambiguity Resolution", test_ambiguity_resolution)
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
        logger.info("🎉 All tests passed! Grammar validation and ambiguity checking are working correctly.")
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()