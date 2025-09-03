# Implementation Plan

- [x] 1. Set up FastAPI project structure and core configuration
  - Create directory structure for app/core, app/api, app/models, app/services, app/db
  - Implement pydantic Settings for configuration management
  - Set up environment variable handling with .env support
  - _Requirements: 5.2, 5.3_

- [ ] 2. Implement authentication and authorization system
  - [ ] 2.1 Create JWT token generation and validation
    - Implement JWT service with access/refresh token support
    - Create token validation middleware for protected endpoints
    - Add role-based claims to JWT payload (student, teacher, admin, service)
    - _Requirements: 5.1, 5.3_

  - [ ] 2.2 Implement RBAC (Role-Based Access Control)
    - Create permission decorators for endpoint protection
    - Implement role hierarchy and permission matrix
    - Add tenant isolation for multi-tenant support
    - _Requirements: 5.3_

  - [ ] 2.3 Add PII redaction middleware
    - Create middleware to mask sensitive data in logs
    - Implement email, IP address, and personal data redaction
    - Add audit logging for data access
    - _Requirements: 5.2_

- [x] 3. Set up database layer with SQLModel and migrations
  - [x] 3.1 Create database models
    - Implement User model with theta values and error profiles
    - Create MathItem and EnglishItem models with metadata
    - Add Attempt model for tracking student responses
    - Create Decision and Event models for bandit logging
    - _Requirements: 1.1, 1.2, 2.1, 2.2_

  - [x] 3.2 Implement repository pattern
    - Create base repository with CRUD operations
    - Implement UserRepository with profile update methods
    - Add ItemRepository with search and filtering capabilities
    - Create AttemptRepository for performance tracking
    - _Requirements: 1.1, 1.2, 4.2_

  - [x] 3.3 Set up Alembic migrations
    - Configure Alembic for database schema management
    - Create initial migration with all table definitions
    - Add indexes for performance optimization (user_id, item_id, skills)
    - _Requirements: 1.1, 1.2_

- [x] 4. Implement IRT-based student profiling service
  - [x] 4.1 Create IRT 2PL model implementation
    - Implement theta estimation using maximum likelihood
    - Add online theta update after each attempt
    - Create difficulty parameter (a, b) estimation for items
    - _Requirements: 1.1, 1.2_

  - [x] 4.2 Implement error profile tracking
    - Create error categorization for math (sign_error, ratio_misuse, etc.)
    - Add English error tracking (prepositions, articles, SVA, collocations)
    - Implement error profile update logic based on incorrect answers
    - _Requirements: 1.4, 2.1_

  - [x] 4.3 Add profile service with caching
    - Create ProfileService with Redis caching (1h TTL)
    - Implement profile retrieval and update methods
    - Add profile validation and constraint checking
    - _Requirements: 1.1, 1.2, 3.1_

- [x] 5. Build hybrid retrieval system
  - [x] 5.1 Implement dense vector search
    - Set up vector database (Qdrant/Weaviate) connection
    - Create embedding generation for items using sentence transformers
    - Implement dense search with metadata filtering
    - _Requirements: 8.1, 8.2_

  - [x] 5.2 Add sparse (BM25) search capability
    - Implement BM25 search using OpenSearch or PostgreSQL full-text
    - Create keyword extraction and query preprocessing
    - Add language-specific text processing (Turkish/English)
    - _Requirements: 8.1, 8.2_

  - [x] 5.3 Create hybrid search fusion
    - Implement Reciprocal Rank Fusion (RRF) for combining dense/sparse results
    - Add metadata filtering for language, level, and skills
    - Create retrieval caching with 24h TTL
    - _Requirements: 8.1, 8.2, 8.4_

- [x] 6. Implement re-ranking service with cross-encoder
  - Create cross-encoder model integration for relevance scoring
  - Implement batch inference for performance optimization
  - Add heuristic scoring fusion (cosine + BM25 + freshness)
  - Create re-ranking cache for frequent queries
  - _Requirements: 8.3_

- [x] 7. Build bandit service for adaptive question selection
  - [x] 7.1 Implement LinUCB algorithm
    - Create context vector generation (theta, skills, device, recency)
    - Implement LinUCB with confidence bounds calculation
    - Add arm selection with upper confidence bound scoring
    - _Requirements: 7.1, 7.2_

  - [x] 7.2 Add constrained bandit implementation
    - Implement minimum success rate constraint (60%)
    - Add coverage constraint for topic diversity (80%)
    - Create constraint violation detection and handling
    - _Requirements: 7.3_

  - [x] 7.3 Create bandit logging and propensity tracking
    - Implement decision logging with propensity scores
    - Add exploration/exploitation ratio tracking
    - Create offline evaluation support (IPS/DR)
    - _Requirements: 7.2, 4.4_

- [-] 8. Implement math question generation service
  - [x] 8.1 Create parametric template system
    - Implement template engine for linear equations, ratios, geometry
    - Add parameter constraint validation and generation
    - Create template versioning and management
    - _Requirements: 6.1, 6.2_

  - [x] 8.2 Add programmatic solver integration
    - Integrate SymPy for equation solving and validation
    - Implement single-solution guarantee checking
    - Add solution step generation for explanations
    - _Requirements: 6.2, 6.3_

  - [x] 8.3 Create distractor generation
    - Implement misconception-based wrong answer generation
    - Add common error pattern detection and usage
    - Create distractor uniqueness and plausibility validation
    - _Requirements: 6.4_

- [x] 9. Build English cloze question generation service
  - [x] 9.1 Implement error taxonomy and confusion sets
    - Create confusion sets for prepositions, articles, collocations
    - Add error pattern matching for targeted question generation
    - Implement CEFR level classification for passages
    - _Requirements: 2.1, 2.2_

  - [x] 9.2 Create cloze generation pipeline
    - Implement rule-based blank selection for target error types
    - Add LLM integration for context-appropriate answer generation
    - Create personalized distractor generation based on error history
    - _Requirements: 2.1, 2.3_

  - [x] 9.3 Add grammar validation and ambiguity checking
    - Integrate grammar checker for answer validation
    - Implement single-answer guarantee checking
    - Add ambiguity detection and resolution
    - _Requirements: 2.4_

- [ ] 10. Create recommendation orchestration service
  - [x] 10.1 Implement main recommendation pipeline
    - Create orchestrator service coordinating all components
    - Implement retrieval → re-ranking → diversification → bandit flow
    - Add error handling and fallback mechanisms
    - _Requirements: 1.1, 1.3, 2.2, 7.1_

  - [ ] 10.2 Add diversification and curriculum coverage
    - Implement MMR (Maximal Marginal Relevance) for diversity
    - Add curriculum gap detection and filling
    - Create topic coverage tracking and balancing
    - _Requirements: 1.3, 2.2, 7.3_

- [ ] 11. Implement API endpoints
  - [ ] 11.1 Create authentication endpoints
    - Implement POST /auth/login with JWT token generation
    - Add POST /auth/register with user creation and validation
    - Create token refresh endpoint
    - _Requirements: 5.1_

  - [ ] 11.2 Add recommendation endpoints
    - Implement POST /v1/recommend/next for question recommendations
    - Add GET /v1/profile/{user_id} for profile retrieval
    - Create POST /v1/profile/update for theta and error profile updates
    - _Requirements: 1.1, 1.2, 3.1, 3.2_

  - [x] 11.3 Create generation endpoints
    - Implement POST /v1/generate/math for math question generation
    - Add POST /v1/generate/en_cloze for English cloze generation
    - Create validation and quality assurance integration
    - _Requirements: 6.1, 6.2, 2.1, 2.2_

  - [ ] 11.4 Add attempt tracking endpoints
    - Implement POST /v1/attempt for answer submission and theta update
    - Add POST /v1/feedback for user feedback collection
    - Create attempt history and analytics endpoints
    - _Requirements: 1.2, 1.4, 2.1_

- [ ] 12. Add quality assurance and validation
  - [ ] 12.1 Implement math question validation
    - Create solver-based answer key verification
    - Add parameter constraint validation
    - Implement template quality scoring
    - _Requirements: 6.2, 6.3_

  - [ ] 12.2 Add English question validation
    - Implement grammar checking integration
    - Create ambiguity detection and scoring
    - Add CEFR level validation for passages
    - _Requirements: 2.4_

  - [ ] 12.3 Create content moderation
    - Add toxicity detection for generated content
    - Implement bias checking for questions and answers
    - Create human-in-the-loop approval workflow
    - _Requirements: 5.2_

- [ ] 13. Implement observability and monitoring
  - [ ] 13.1 Add structured logging
    - Implement JSON structured logging with request IDs
    - Add performance metrics logging (retrieval_ms, rerank_ms, llm_ms)
    - Create audit logging for decisions and data access
    - _Requirements: 4.4, 5.4_

  - [ ] 13.2 Create metrics collection
    - Implement Prometheus metrics for latency, error rate, cache hit rate
    - Add business metrics (faithfulness, difficulty_match, coverage)
    - Create bandit-specific metrics (exploration_ratio, constraint_violations)
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 13.3 Add health checks and admin endpoints
    - Implement GET /health with service status
    - Create GET /v1/admin/metrics for system metrics
    - Add GET /v1/admin/decisions/{request_id} for decision audit
    - _Requirements: 4.1, 4.4_

- [ ] 14. Add caching and performance optimization
  - [ ] 14.1 Implement Redis caching
    - Set up Redis connection and configuration
    - Add retrieval result caching with TTL management
    - Implement semantic caching for similar queries
    - _Requirements: 8.4_

  - [ ] 14.2 Add database optimization
    - Create database connection pooling
    - Add query optimization and indexing
    - Implement read replicas for analytics queries
    - _Requirements: 1.1, 1.2_

- [ ] 15. Create comprehensive test suite
  - [ ] 15.1 Write unit tests
    - Create tests for IRT calculations and profile updates
    - Add tests for bandit algorithm implementations
    - Test math solver and English grammar validation
    - _Requirements: 1.1, 1.2, 6.2, 2.4_

  - [ ] 15.2 Add integration tests
    - Create API endpoint integration tests
    - Test database operations and migrations
    - Add external service integration tests (LLM, vector DB)
    - _Requirements: All requirements_

  - [ ] 15.3 Implement performance tests
    - Create load tests for recommendation endpoints
    - Add latency benchmarks for p95 < 700ms target
    - Test cache performance and hit rates
    - _Requirements: 4.1, 4.2_