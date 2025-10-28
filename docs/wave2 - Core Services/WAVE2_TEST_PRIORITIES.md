# Wave 2 Test Coverage Priorities

## Current State
- **Total Coverage:** 8.06%
- **Target Coverage:** 80%
- **Gap:** 71.94%

## Modules Requiring Tests (Prioritized)

### Priority 1: Core Infrastructure (Critical)
These modules are essential for the application to function and should be tested first.

#### 1. `app/config.py` (0% → Target: 90%)
**Lines:** 13 statements
**Missing:** 9-41
**Why Critical:** Configuration affects all application behavior
**Test Requirements:**
- Environment variable loading
- Configuration validation
- Default values
- Secret handling

#### 2. `app/db/base.py` (0% → Target: 95%)
**Lines:** 4 statements
**Missing:** 6-15
**Why Critical:** Foundation for all database operations
**Test Requirements:**
- Database connection
- Session management
- Transaction handling

#### 3. `app/db/models.py` (0% → Target: 85%)
**Lines:** 39 statements
**Missing:** 10-109
**Why Critical:** Data integrity depends on correct model definitions
**Test Requirements:**
- Model field validation
- Relationships
- Constraints
- Serialization

### Priority 2: Execution Infrastructure (High)
These modules handle workflow execution and must be reliable.

#### 4. `app/exec_loop.py` (0% → Target: 80%)
**Lines:** 10 statements
**Missing:** 15-74
**Why Important:** Orchestrates all flow executions
**Test Requirements:**
- Flow execution lifecycle
- Error handling
- State management
- Concurrency handling

#### 5. `app/logging.py` (0% → Target: 75%)
**Lines:** 8 statements
**Missing:** 9-51
**Why Important:** Critical for debugging and monitoring
**Test Requirements:**
- Log formatting
- Log levels
- Structured logging
- Context propagation

### Priority 3: Core Business Logic (High)
These modules implement the actual business functionality.

#### 6. `app/core/base.py` (0% → Target: 80%)
**Lines:** 64 statements
**Missing:** 8-529
**Why Important:** Foundation for all tools and flows
**Test Requirements:**
- Base class functionality
- Tool execution
- State management
- Error handling

#### 7. `app/core/types.py` (0% → Target: 75%)
**Lines:** 85 statements
**Missing:** 8-460
**Why Important:** Type safety and validation
**Test Requirements:**
- Type validation
- Pydantic models
- Serialization
- Edge cases

#### 8. `app/core/decorators.py` (0% → Target: 70%)
**Lines:** 96 statements
**Missing:** 8-397
**Why Important:** Cross-cutting concerns (caching, retries, etc.)
**Test Requirements:**
- Decorator functionality
- Error handling
- Performance impact
- Edge cases

#### 9. `app/core/exceptions.py` (0% → Target: 90%)
**Lines:** 34 statements
**Missing:** 10-342
**Why Important:** Error handling throughout the application
**Test Requirements:**
- Exception types
- Error messages
- Error context
- Serialization

### Priority 4: LLM Integration (Medium-High)
These modules handle AI provider integration.

#### 10. `app/adapters/llm_factory.py` (0% → Target: 85%)
**Lines:** 14 statements
**Missing:** 8-41
**Why Important:** Abstracts LLM provider selection
**Test Requirements:**
- Provider selection
- Configuration
- Fallback logic
- Error handling

#### 11. `app/adapters/llm_openai.py` (0% → Target: 80%)
**Lines:** 15 statements
**Missing:** 13-101
**Why Important:** Primary LLM provider
**Test Requirements:**
- API calls (mocked)
- Response parsing
- Error handling
- Rate limiting

#### 12. `app/adapters/llm_vertex.py` (0% → Target: 80%)
**Lines:** 15 statements
**Missing:** 13-102
**Why Important:** Alternative LLM provider
**Test Requirements:**
- API calls (mocked)
- Response parsing
- Error handling
- Authentication

### Priority 5: RAG System (Medium)
These modules handle document processing and retrieval.

#### 13. `app/rag/ingestion.py` (0% → Target: 75%)
**Lines:** 17 statements
**Missing:** 10-112
**Why Important:** Document processing pipeline
**Test Requirements:**
- Document parsing
- Chunking
- Embedding generation
- Storage

#### 14. `app/rag/retriever.py` (0% → Target: 75%)
**Lines:** 14 statements
**Missing:** 10-103
**Why Important:** Context retrieval for LLM
**Test Requirements:**
- Query processing
- Vector search
- Result ranking
- Caching

### Priority 6: Flow Implementations (Medium)
These modules implement specific business workflows.

#### 15. `app/flows/maturity_assessment/graph.py` (0% → Target: 70%)
**Lines:** 10 statements
**Missing:** 11-67
**Why Important:** Core business workflow
**Test Requirements:**
- Flow execution
- State transitions
- Error handling
- Output validation

#### 16. `app/flows/usecase_grooming/graph.py` (0% → Target: 70%)
**Lines:** 10 statements
**Missing:** 10-65
**Why Important:** Core business workflow
**Test Requirements:**
- Flow execution
- State transitions
- Error handling
- Output validation

### Priority 7: Tool Registry Enhancement (Low)
Already has some coverage, needs completion.

#### 17. `app/tools/registry.py` (63.64% → Target: 90%)
**Lines:** 22 statements (8 missing)
**Missing:** 65-74
**Why Low Priority:** Already has good coverage
**Test Requirements:**
- Complete coverage of missing lines
- Error scenarios
- Edge cases

## Estimated Test Writing Effort

### Phase 1: Critical Infrastructure (Week 1)
- Config, DB base, DB models, Logging
- **Estimated Lines of Test Code:** 400-500
- **Estimated Tests:** 25-30
- **Coverage Gain:** ~15%

### Phase 2: Core Business Logic (Week 2)
- Base classes, Types, Decorators, Exceptions
- **Estimated Lines of Test Code:** 800-1000
- **Estimated Tests:** 50-60
- **Coverage Gain:** ~35%

### Phase 3: LLM & RAG (Week 3)
- LLM adapters, RAG ingestion, RAG retriever
- **Estimated Lines of Test Code:** 600-700
- **Estimated Tests:** 35-40
- **Coverage Gain:** ~15%

### Phase 4: Flow Implementations (Week 4)
- Maturity assessment, Use case grooming
- **Estimated Lines of Test Code:** 400-500
- **Estimated Tests:** 20-25
- **Coverage Gain:** ~5%

### Phase 5: Integration & E2E (Week 5)
- Integration tests, E2E tests, Performance tests
- **Estimated Lines of Test Code:** 500-600
- **Estimated Tests:** 15-20
- **Coverage Gain:** ~10%

## Total Effort Estimate
- **Total Test Code:** 2,700-3,300 lines
- **Total Tests:** 145-175 tests
- **Total Time:** 5 weeks
- **Expected Final Coverage:** 80-85%

## Testing Strategy

### Unit Tests (70% of effort)
- Mock all external dependencies
- Test individual functions/classes in isolation
- Fast execution (< 1 second per test)
- High coverage of edge cases

### Integration Tests (20% of effort)
- Test component interactions
- Use real database (test instance)
- Mock only external APIs (LLM, GCS)
- Medium execution time (1-5 seconds per test)

### E2E Tests (10% of effort)
- Test complete workflows
- Use deployed service
- Real infrastructure
- Slow execution (5-30 seconds per test)

## Success Metrics
- Achieve 80%+ overall coverage
- All critical paths covered
- Fast test execution (< 5 minutes total)
- Clear, maintainable test code
- Automated CI/CD integration

## Test Infrastructure Needs
1. **pytest fixtures** for common test setup
2. **Mock factories** for LLM responses
3. **Test database** setup/teardown
4. **Coverage reporting** in CI/CD
5. **Test data generators** for realistic data
6. **Performance benchmarks** for critical paths

## Continuous Improvement
- Add tests for every new feature
- Maintain 80%+ coverage requirement
- Review coverage reports weekly
- Refactor tests as code evolves
- Update test documentation
