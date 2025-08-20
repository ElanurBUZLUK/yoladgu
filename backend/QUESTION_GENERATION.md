# 🚀 Enhanced Question Generation System

## Overview

The Enhanced Question Generation System is a sophisticated hybrid approach that combines template-based generation with AI-powered GPT generation, featuring intelligent orchestration, caching, and comprehensive logging.

## 🏗️ Architecture

### Core Components

1. **QuestionGenerator** - Main orchestration class
2. **Template System** - Pre-defined question templates
3. **GPT Integration** - AI-powered question generation
4. **Caching Layer** - Performance optimization
5. **Logging System** - Comprehensive monitoring

### System Flow

```
Request → Cache Check → Orchestration → Generation → Caching → Response
   ↓           ↓           ↓           ↓         ↓         ↓
Log Start   Cache Hit   Strategy    Template   Store    Log Result
           or Miss      Decision    or GPT    Result
```

## 🎯 Orchestration Logic

### Strategy Decision Tree

```
1. Check if template exists for error_type
   ├─ No template → Use GPT directly (if enabled)
   └─ Template exists → Hybrid strategy
       ├─ 70% chance: Use template
       └─ 30% chance: Use GPT
           ├─ GPT succeeds → Return result
           └─ GPT fails → Fallback to template
```

### Orchestration Methods

- **`_orchestrate_generation_strategy()`** - Central decision maker
- **`_has_template_for_error_type()`** - Template availability check
- **Fallback mechanisms** - Automatic recovery from failures

## 💾 Caching System

### Cache Features

- **TTL-based expiration** - 24-hour cache lifetime
- **Size management** - Automatic cleanup when limit reached
- **Usage tracking** - Hit count and last access time
- **Smart eviction** - Remove oldest 25% when full

### Cache Operations

```python
# Generate cache key
cache_key = question_generator._generate_cache_key(
    subject, error_type, difficulty_level, student_context
)

# Check cache
cached_question = question_generator._get_cached_question(cache_key)

# Cache management
question_generator.clear_expired_cache()
question_generator.clear_all_cache()
```

## 📊 Enhanced Logging

### Log Levels

- **INFO** - Normal operations, strategy decisions
- **DEBUG** - Detailed template selection, cache operations
- **WARNING** - Fallback usage, template failures
- **ERROR** - GPT failures, unexpected errors

### Logged Information

- Generation start/end times
- Orchestration strategy decisions
- Cache hit/miss statistics
- Generation method used
- Performance metrics (generation time)
- Error details and fallback reasons

### Example Log Output

```
2024-01-15 10:30:15 - app.services.question_generator - INFO - Starting question generation for english/present_perfect_error (difficulty: 3)
2024-01-15 10:30:15 - app.services.question_generator - INFO - Generation strategy: template
2024-01-15 10:30:15 - app.services.question_generator - DEBUG - Selected template: template_001 for english/present_perfect_error
2024-01-15 10:30:15 - app.services.question_generator - INFO - Template generation successful for english/present_perfect_error
2024-01-15 10:30:15 - app.services.question_generator - INFO - Question cached with key: a1b2c3d4e5f6
2024-01-15 10:30:15 - app.services.question_generator - INFO - Question generation completed for english/present_perfect_error in 45.23ms using template
```

## 🔧 Configuration

### Orchestration Settings

```python
# Template vs GPT ratio (70% template, 30% GPT)
template_gpt_ratio = 0.7

# Cache configuration
cache_ttl_hours = 24
max_cache_size = 1000
```

### Environment Variables

```bash
# Question Generation settings
USE_TEMPLATE_QUESTIONS=true
USE_GPT_QUESTIONS=true
TEMPLATE_FALLBACK=true
MAX_GPT_QUESTIONS_PER_DAY=100
GPT_CREATIVITY=0.7
QUESTION_DIVERSITY_THRESHOLD=0.8
```

## 📈 Performance Monitoring

### Statistics Available

```python
stats = question_generator.get_generation_stats()

# Cache performance
cache_stats = stats['cache_performance']
print(f"Total cached questions: {cache_stats['total_cached_questions']}")
print(f"Average cache usage: {cache_stats['average_cache_usage']:.2f}")

# Orchestration settings
orch_settings = stats['orchestration_settings']
print(f"Template/GPT ratio: {orch_settings['template_gpt_ratio']}")
print(f"Cache TTL: {orch_settings['cache_ttl_hours']} hours")
```

### Cache Performance Metrics

- **Hit Rate** - Percentage of cache hits
- **Usage Count** - How often each question is used
- **Age Distribution** - Spread of cache entry ages
- **Eviction Rate** - How often entries are removed

## 🧪 Testing

### Test Script

Run the comprehensive test suite:

```bash
cd backend
python test_question_generator.py
```

### Test Coverage

1. **Template Generation** - Basic template functionality
2. **Cache Hit/Miss** - Caching system validation
3. **Math Questions** - Subject-specific generation
4. **Fallback Logic** - Error handling and recovery
5. **Statistics** - Performance monitoring
6. **Cache Management** - Cleanup operations

## 🚨 Error Handling

### Fallback Mechanisms

1. **Template → GPT** - If template fails
2. **GPT → Template** - If GPT fails
3. **Error Fallback** - If all methods fail

### Error Logging

```python
try:
    question_data = await self._generate_gpt_question(...)
except Exception as e:
    logger.error(f"GPT generation failed for {subject}/{error_type}: {e}")
    # Automatic fallback to template
    question_data = self._generate_template_question(...)
```

## 🔒 Security Features

### API Key Management

- Environment variable-based configuration
- Automatic validation on startup
- Graceful degradation when keys are missing
- Secure logging (no key exposure)

### Access Control

- Daily GPT usage limits
- Rate limiting per user
- Cost monitoring and alerts

## 📚 API Usage

### Basic Question Generation

```python
from app.services.question_generator import question_generator

# Generate a question
question = await question_generator.generate_question(
    subject="english",
    error_type="present_perfect_error",
    difficulty_level=3,
    student_context="Student struggles with present perfect tense"
)

# Access metadata
print(f"Method: {question['metadata']['generation_method']}")
print(f"Strategy: {question['metadata']['orchestration_strategy']}")
print(f"Generation time: {question['metadata']['generation_time_ms']}ms")
```

### Advanced Features

```python
# Get comprehensive statistics
stats = question_generator.get_generation_stats()

# Cache management
expired_count = question_generator.clear_expired_cache()
all_cleared = question_generator.clear_all_cache()

# Check GPT availability
can_use_gpt = question_generator._can_use_gpt()
```

## 🚀 Future Enhancements

### Planned Features

1. **Redis Integration** - Distributed caching
2. **Machine Learning** - Dynamic strategy optimization
3. **A/B Testing** - Strategy performance comparison
4. **Real-time Monitoring** - Live performance dashboards
5. **Predictive Caching** - Pre-generate popular questions

### Extensibility

The system is designed for easy extension:

- New generation methods can be added
- Custom orchestration strategies
- Plugin-based template systems
- Custom caching backends

## 📖 Best Practices

### Development

1. **Always use async/await** for question generation
2. **Check cache first** before generating new questions
3. **Log all operations** for debugging and monitoring
4. **Handle errors gracefully** with fallback mechanisms

### Production

1. **Monitor cache performance** regularly
2. **Set appropriate TTL** based on question freshness requirements
3. **Use structured logging** for better analysis
4. **Implement rate limiting** to prevent abuse

### Maintenance

1. **Clear expired cache** periodically
2. **Monitor GPT usage** and costs
3. **Review orchestration ratios** based on performance
4. **Update templates** regularly for variety

---

**For more information, see the main project documentation and security guidelines.**
