#!/usr/bin/env python3
"""
Test script for English error taxonomy implementation.
"""

import logging
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_error_taxonomy_initialization():
    """Test that error taxonomy initializes correctly."""
    try:
        from app.services.english_generation_service import (
            english_error_taxonomy, ErrorType, CEFRLevel
        )
        
        logger.info("Testing error taxonomy initialization...")
        
        # Test that confusion sets are loaded
        preposition_sets = english_error_taxonomy.get_confusion_sets_by_error_type(ErrorType.PREPOSITIONS)
        assert len(preposition_sets) > 0, "No preposition confusion sets found"
        
        article_sets = english_error_taxonomy.get_confusion_sets_by_error_type(ErrorType.ARTICLES)
        assert len(article_sets) > 0, "No article confusion sets found"
        
        sva_sets = english_error_taxonomy.get_confusion_sets_by_error_type(ErrorType.SUBJECT_VERB_AGREEMENT)
        assert len(sva_sets) > 0, "No subject-verb agreement confusion sets found"
        
        collocation_sets = english_error_taxonomy.get_confusion_sets_by_error_type(ErrorType.COLLOCATIONS)
        assert len(collocation_sets) > 0, "No collocation confusion sets found"
        
        logger.info(f"✅ Found {len(preposition_sets)} preposition sets")
        logger.info(f"✅ Found {len(article_sets)} article sets")
        logger.info(f"✅ Found {len(sva_sets)} subject-verb agreement sets")
        logger.info(f"✅ Found {len(collocation_sets)} collocation sets")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error taxonomy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cefr_level_filtering():
    """Test CEFR level filtering functionality."""
    try:
        from app.services.english_generation_service import (
            english_error_taxonomy, CEFRLevel
        )
        
        logger.info("Testing CEFR level filtering...")
        
        # Test A1 level filtering
        a1_sets = english_error_taxonomy.get_confusion_sets_by_cefr_level(CEFRLevel.A1)
        assert len(a1_sets) > 0, "No A1 level confusion sets found"
        
        # Test B1 level filtering (should include A1, A2, B1)
        b1_sets = english_error_taxonomy.get_confusion_sets_by_cefr_level(CEFRLevel.B1)
        assert len(b1_sets) >= len(a1_sets), "B1 sets should include A1 sets"
        
        logger.info(f"✅ A1 level: {len(a1_sets)} confusion sets")
        logger.info(f"✅ B1 level: {len(b1_sets)} confusion sets")
        
        # Verify all A1 sets are appropriate level
        for cs in a1_sets:
            assert cs.cefr_level.value <= CEFRLevel.A1.value, f"Invalid CEFR level: {cs.cefr_level}"
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CEFR level filtering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_targeted_confusion_sets():
    """Test targeted confusion set selection based on error profile."""
    try:
        from app.services.english_generation_service import (
            english_error_taxonomy, CEFRLevel
        )
        
        logger.info("Testing targeted confusion set selection...")
        
        # Test error profile with high preposition errors
        error_profile = {
            "prepositions": 0.6,
            "articles": 0.2,
            "subject_verb_agreement": 0.4,
            "collocations": 0.1
        }
        
        targeted_sets = english_error_taxonomy.get_targeted_confusion_sets(
            error_profile=error_profile,
            cefr_level=CEFRLevel.A2,
            min_error_rate=0.3
        )
        
        assert len(targeted_sets) > 0, "No targeted confusion sets found"
        
        # Check that prepositions are prioritized (highest error rate)
        first_set = targeted_sets[0]
        assert first_set.error_type.value == "prepositions", "Prepositions should be prioritized"
        
        logger.info(f"✅ Found {len(targeted_sets)} targeted confusion sets")
        logger.info(f"✅ Top priority: {first_set.error_type.value} (rate: {getattr(first_set, 'priority_score', 'N/A')})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Targeted confusion set selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cefr_classification():
    """Test CEFR level classification of texts."""
    try:
        from app.services.english_generation_service import (
            english_error_taxonomy, CEFRLevel
        )
        
        logger.info("Testing CEFR level classification...")
        
        # Test simple A1 text
        a1_text = "Hello. I am John. I have a car. It is red."
        a1_level = english_error_taxonomy.classify_cefr_level(a1_text)
        logger.info(f"✅ A1 text classified as: {a1_level.value}")
        
        # Test more complex text
        b1_text = "The government should invest more money in education and technology to create better opportunities for young people."
        b1_level = english_error_taxonomy.classify_cefr_level(b1_text)
        logger.info(f"✅ B1 text classified as: {b1_level.value}")
        
        # Test advanced text
        c1_text = "The unprecedented complexity of contemporary globalization phenomena requires sophisticated analytical frameworks to comprehend their multifaceted implications."
        c1_level = english_error_taxonomy.classify_cefr_level(c1_text)
        logger.info(f"✅ C1 text classified as: {c1_level.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CEFR classification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confusion_set_validation():
    """Test confusion set validation functionality."""
    try:
        from app.services.english_generation_service import (
            english_error_taxonomy, ConfusionSet, ErrorType, CEFRLevel
        )
        
        logger.info("Testing confusion set validation...")
        
        # Test valid confusion set
        valid_set = ConfusionSet(
            error_type=ErrorType.PREPOSITIONS,
            target_word="at",
            confusors=["in", "on"],
            context_patterns=[r"\b(at|in|on)\s+\d+:\d+"],
            cefr_level=CEFRLevel.A1,
            examples=["I wake up at 7 o'clock."]
        )
        
        is_valid = english_error_taxonomy.validate_confusion_set(valid_set)
        assert is_valid, "Valid confusion set should pass validation"
        logger.info("✅ Valid confusion set passed validation")
        
        # Test invalid confusion set (target in confusors)
        invalid_set = ConfusionSet(
            error_type=ErrorType.PREPOSITIONS,
            target_word="at",
            confusors=["at", "in", "on"],  # Target word in confusors
            context_patterns=[r"\b(at|in|on)\s+\d+:\d+"],
            cefr_level=CEFRLevel.A1,
            examples=["I wake up at 7 o'clock."]
        )
        
        is_invalid = english_error_taxonomy.validate_confusion_set(invalid_set)
        assert not is_invalid, "Invalid confusion set should fail validation"
        logger.info("✅ Invalid confusion set correctly failed validation")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Confusion set validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_statistics():
    """Test error taxonomy statistics."""
    try:
        from app.services.english_generation_service import english_error_taxonomy
        
        logger.info("Testing error taxonomy statistics...")
        
        stats = english_error_taxonomy.get_error_statistics()
        
        assert "total_error_types" in stats
        assert "total_confusion_sets" in stats
        assert "error_type_distribution" in stats
        assert "cefr_level_distribution" in stats
        
        logger.info(f"✅ Total error types: {stats['total_error_types']}")
        logger.info(f"✅ Total confusion sets: {stats['total_confusion_sets']}")
        logger.info(f"✅ Error type distribution: {stats['error_type_distribution']}")
        logger.info(f"✅ CEFR level distribution: {stats['cefr_level_distribution']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error statistics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting English error taxonomy tests...")
    
    tests = [
        ("Error Taxonomy Initialization", test_error_taxonomy_initialization),
        ("CEFR Level Filtering", test_cefr_level_filtering),
        ("Targeted Confusion Sets", test_targeted_confusion_sets),
        ("CEFR Classification", test_cefr_classification),
        ("Confusion Set Validation", test_confusion_set_validation),
        ("Error Statistics", test_error_statistics)
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
        logger.info("🎉 All tests passed! English error taxonomy is working correctly.")
    else:
        logger.error("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()