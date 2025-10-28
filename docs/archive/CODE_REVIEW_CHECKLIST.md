# Code Review Checklist

## Overview

This checklist ensures consistent, high-quality code reviews across the AI Agency platform. Use this as a guide when reviewing pull requests.

---

## 1. Code Quality

### Type Hints
- [ ] All public functions have complete type hints
- [ ] Return types are specified for all functions
- [ ] Type hints use Python 3.11+ syntax (`str | None` not `Optional[str]`)
- [ ] Complex types use appropriate aliases or protocols
- [ ] No use of `Any` without justification

### Docstrings
- [ ] All public modules have module-level docstrings
- [ ] All public classes have docstrings with Attributes section
- [ ] All public functions have docstrings following Google style
- [ ] Docstrings include Args, Returns, Raises, and Example sections where appropriate
- [ ] Complex logic has inline comments explaining "why" not "what"

### Naming Conventions
- [ ] Variable names are descriptive and reveal intent
- [ ] Function names use `snake_case`
- [ ] Class names use `PascalCase`
- [ ] Constants use `UPPERCASE_WITH_UNDERSCORES`
- [ ] Boolean variables use `is_`, `has_`, `can_` prefixes
- [ ] No single-letter variables except in list comprehensions/loops

### Code Structure
- [ ] Functions are focused on single responsibility
- [ ] Functions are < 50 lines (with reasonable exceptions)
- [ ] No deeply nested code (max 3-4 levels)
- [ ] Early returns used to reduce nesting
- [ ] Complex logic extracted into helper functions
- [ ] No code duplication (DRY principle)

### Import Organization
- [ ] Imports organized: stdlib, third-party, local
- [ ] Each group separated by blank line
- [ ] Imports sorted alphabetically within groups
- [ ] No wildcard imports (`from module import *`)
- [ ] No unused imports

---

## 2. Error Handling

### Exception Usage
- [ ] Custom exceptions from `app.core.exceptions` are used
- [ ] Exceptions include helpful context (IDs, operation names)
- [ ] No bare `except:` clauses
- [ ] Specific exception types are caught, not generic `Exception`
- [ ] Exception chaining used with `raise ... from e`
- [ ] Exceptions logged with structured context

### Validation
- [ ] Input validation uses Pydantic models
- [ ] Validation errors are descriptive and actionable
- [ ] Edge cases are handled (empty strings, None, negative numbers)
- [ ] Database query results checked for None
- [ ] External API responses validated

### Retry Logic
- [ ] Retry decorators used for transient failures
- [ ] Appropriate retry configuration (max attempts, backoff)
- [ ] Only retryable exceptions are caught
- [ ] Retry logic doesn't mask permanent failures

---

## 3. Testing

### Test Coverage
- [ ] Test coverage >= 80% for new code
- [ ] Critical paths have 100% coverage
- [ ] Both success and failure cases tested
- [ ] Edge cases tested (empty inputs, None, extreme values)
- [ ] Concurrent operations tested if applicable

### Test Quality
- [ ] Tests follow AAA pattern (Arrange, Act, Assert)
- [ ] Test names clearly describe what is being tested
- [ ] Tests are isolated and independent
- [ ] Mocks used for external dependencies
- [ ] Fixtures used for common test setup
- [ ] No test interdependencies (order shouldn't matter)

### Test Organization
- [ ] Tests in appropriate directory (unit/integration/e2e)
- [ ] Test files mirror source code structure
- [ ] Shared fixtures in conftest.py
- [ ] Test data in fixtures/ directory if needed

---

## 4. Async/Await Patterns

### Async Usage
- [ ] All I/O operations use `async def`
- [ ] Database queries use `await`
- [ ] HTTP calls use async client (httpx, not requests)
- [ ] File operations use `aiofiles`
- [ ] No blocking operations in async functions

### Concurrency
- [ ] `asyncio.gather()` used for concurrent operations
- [ ] Proper error handling in concurrent operations
- [ ] Timeouts configured for external calls
- [ ] Resource limits considered (connection pools)

### Resource Management
- [ ] Async context managers used (`async with`)
- [ ] Database sessions properly closed
- [ ] HTTP clients properly closed
- [ ] File handles properly closed
- [ ] No resource leaks

---

## 5. Security

### Secrets Management
- [ ] No hardcoded secrets or API keys
- [ ] Secrets loaded from environment variables
- [ ] `.env` file not committed (in `.gitignore`)
- [ ] Sensitive data not logged
- [ ] API keys masked in error messages

### Input Validation
- [ ] All user input validated
- [ ] SQL injection prevented (using parameterized queries)
- [ ] Path traversal prevented (validating file paths)
- [ ] XSS prevented (input sanitization)
- [ ] File upload size limits enforced

### Authentication & Authorization
- [ ] Authentication required for protected endpoints
- [ ] Authorization checks before sensitive operations
- [ ] User permissions validated
- [ ] API keys validated
- [ ] Rate limiting implemented

### Data Protection
- [ ] Sensitive data encrypted at rest if needed
- [ ] HTTPS enforced for external communications
- [ ] PII handled according to privacy policy
- [ ] Data retention policies followed

---

## 6. Performance

### Database Optimization
- [ ] No N+1 query problems
- [ ] Appropriate indexes exist for queries
- [ ] Eager loading used where appropriate
- [ ] Query result sets limited
- [ ] Connection pooling configured

### Caching
- [ ] Expensive operations cached
- [ ] Cache invalidation strategy defined
- [ ] Cache TTL configured appropriately
- [ ] Cache size limits set

### Resource Usage
- [ ] No unnecessary memory allocations
- [ ] Large files streamed, not loaded entirely
- [ ] Background tasks used for long operations
- [ ] Pagination implemented for large result sets
- [ ] Connection pools sized appropriately

### Monitoring
- [ ] Key operations logged
- [ ] Performance metrics collected
- [ ] Slow operations identified and logged
- [ ] Error rates tracked

---

## 7. Documentation

### Code Documentation
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] Configuration changes documented
- [ ] Architecture diagrams updated if needed
- [ ] Migration guide provided for breaking changes

### Comments
- [ ] Complex logic explained
- [ ] Non-obvious decisions documented
- [ ] TODOs tracked (with issue numbers)
- [ ] Deprecated code marked with alternatives

### Changelog
- [ ] CHANGELOG.md updated
- [ ] Breaking changes highlighted
- [ ] Migration steps documented

---

## 8. Dependencies

### Dependency Management
- [ ] New dependencies justified
- [ ] Dependencies added to `pyproject.toml`
- [ ] Dependency versions pinned appropriately
- [ ] No unnecessary dependencies
- [ ] License compatibility checked

### Version Compatibility
- [ ] Python version requirements met (3.11+)
- [ ] Dependency versions compatible
- [ ] Breaking changes from dependency updates handled

---

## 9. Database Changes

### Migrations
- [ ] Migration files created for schema changes
- [ ] Migration tested (up and down)
- [ ] Migration idempotent if possible
- [ ] Data migration strategy for production
- [ ] Indexes added for new queries

### Schema Design
- [ ] Appropriate data types used
- [ ] Constraints defined (NOT NULL, UNIQUE, etc.)
- [ ] Foreign keys defined with proper CASCADE rules
- [ ] Indexes on frequently queried columns
- [ ] Default values set where appropriate

---

## 10. API Design

### REST Principles
- [ ] Appropriate HTTP methods used (GET, POST, PUT, DELETE)
- [ ] Proper HTTP status codes returned
- [ ] Consistent URL structure
- [ ] Versioning strategy followed
- [ ] Pagination for list endpoints

### Request/Response
- [ ] Request models use Pydantic
- [ ] Response models documented
- [ ] Error responses consistent
- [ ] Appropriate response codes for errors
- [ ] JSON response format consistent

### Backward Compatibility
- [ ] Breaking changes avoided if possible
- [ ] Deprecation warnings for removed features
- [ ] API version incremented for breaking changes
- [ ] Migration path documented

---

## 11. Configuration

### Environment Variables
- [ ] New config added to `Settings` class
- [ ] `.env.example` updated
- [ ] Default values provided where sensible
- [ ] Required vs optional config documented
- [ ] Validation for config values

### Feature Flags
- [ ] Feature flags used for experimental features
- [ ] Feature flag defaults documented
- [ ] Cleanup plan for old feature flags

---

## 12. Deployment Considerations

### Docker
- [ ] Dockerfile updated if needed
- [ ] Docker image builds successfully
- [ ] Container healthcheck defined
- [ ] Resource limits appropriate

### Monitoring
- [ ] Logging appropriate for production debugging
- [ ] Metrics exposed for monitoring
- [ ] Alerts configured for errors
- [ ] Performance impact considered

### Rollback Plan
- [ ] Rollback procedure documented
- [ ] Database migrations reversible
- [ ] Feature flags allow quick disable

---

## Final Checks

### Code Quality Tools
- [ ] `ruff check` passes with no errors
- [ ] `ruff format` applied
- [ ] `mypy` passes with no errors
- [ ] All tests pass (`pytest`)
- [ ] Test coverage >= 80%

### Git
- [ ] Commits have clear, descriptive messages
- [ ] No merge conflicts
- [ ] Branch up to date with main/develop
- [ ] No debug code or commented-out code
- [ ] `.gitignore` updated if needed

### Pull Request
- [ ] PR title clearly describes changes
- [ ] PR description explains what and why
- [ ] Breaking changes highlighted
- [ ] Screenshots/demos for UI changes
- [ ] Related issues linked

---

## Approval Criteria

### Must Have (Blocking)
- All tests passing
- No security vulnerabilities
- Code quality tools passing (ruff, mypy)
- Test coverage >= 80%
- No hardcoded secrets
- Backward compatibility maintained or migration path documented

### Should Have (Non-blocking, but address before merge)
- All docstrings complete
- Performance optimizations applied
- Monitoring/logging adequate
- Documentation updated

### Nice to Have (Can be follow-up)
- Additional test coverage (>80%)
- Performance improvements
- Code refactoring for clarity

---

## Review Process

1. **Self-Review**: Author reviews their own code against this checklist before requesting review
2. **Automated Checks**: CI/CD runs automated tests and quality checks
3. **Peer Review**: At least one reviewer approves using this checklist
4. **Address Feedback**: Author addresses all blocking feedback
5. **Final Approval**: Reviewer approves after feedback addressed
6. **Merge**: Code merged to main branch

---

## Notes for Reviewers

- **Be Constructive**: Provide specific, actionable feedback
- **Explain Why**: Don't just say "change this," explain the reasoning
- **Recognize Good Work**: Call out well-written code
- **Ask Questions**: If something is unclear, ask for clarification
- **Consider Context**: Understand the full context and requirements
- **Balance Perfectionism**: Don't block on minor style preferences

## Notes for Authors

- **Respond to All Feedback**: Address or discuss every comment
- **Don't Take It Personally**: Reviews are about code quality, not personal criticism
- **Ask for Clarification**: If feedback is unclear, ask questions
- **Update PR Description**: Keep description current as changes are made
- **Test Changes**: Verify all changes work before requesting re-review
