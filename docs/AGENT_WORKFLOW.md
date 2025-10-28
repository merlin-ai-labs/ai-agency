# Agent Workflow Guide

## Overview

This document defines the systematic workflow for invoking specialized AI agents during development. Following this workflow ensures consistent code quality, security, documentation, and testing standards.

## The Agent Team

We have 10 specialized agents working together:

1. **tech-lead** - Architecture decisions, Wave planning, technical guidance
2. **devops-engineer** - Infrastructure, deployment, CI/CD, GCP resources
3. **backend-engineer** - API implementation, business logic, database operations
4. **security-engineer** - Security reviews, vulnerability scanning, best practices
5. **code-reviewer** - Code quality reviews, standards enforcement, refactoring suggestions
6. **qa-engineer** - Unit tests, integration tests, E2E tests, smoke tests
7. **docs-engineer** - Documentation maintenance, README updates, API docs
8. **code-cleaner** - Codebase cleanup, unused file removal, script consolidation
9. **agent-orchestrator** - Wave coordination, agent sequencing, progress tracking
10. **product-manager** - Requirements gathering, stakeholder communication, backlog management

## Core Principle

**DO NOT execute commands directly without involving specialized agents for reviews.**

Instead of:
```
❌ Make changes → Deploy → Done
```

Use:
```
✅ Make changes → QA testing → Security review → Code review → Documentation → Cleanup → Deploy
```

## Standard Workflows

### Workflow 1: Feature Implementation

**When to use**: Implementing a new feature or significant enhancement

**Process**:
1. **Planning** (`tech-lead`)
   - Design architecture
   - Define implementation approach
   - Identify dependencies

2. **Implementation** (`backend-engineer` or `devops-engineer`)
   - Write code following design
   - Implement business logic
   - Add database migrations if needed

3. **Testing** (`qa-engineer`) - **REQUIRED**
   - Write unit tests (target: 80%+ coverage)
   - Write integration tests
   - Write E2E tests for deployed endpoints
   - Verify all tests pass

4. **Security Review** (`security-engineer`) - **REQUIRED**
   - Review for vulnerabilities
   - Check secrets management
   - Validate authentication/authorization
   - Scan for security issues

5. **Code Review** (`code-reviewer`) - **REQUIRED**
   - Review code quality
   - Check adherence to standards
   - Suggest refactoring
   - Verify test coverage

6. **Documentation** (`docs-engineer`) - **REQUIRED**
   - Update API documentation
   - Update README.md if needed
   - Document new features
   - Update architecture docs

7. **Deployment** (`devops-engineer`)
   - Deploy to GCP
   - Run smoke tests
   - Verify deployment

**Example**:
```bash
# 1. Tech lead designs the feature
Task(agent="tech-lead", prompt="Design authentication system with API keys")

# 2. Backend engineer implements
Task(agent="backend-engineer", prompt="Implement API key authentication middleware")

# 3. QA engineer writes tests
Task(agent="qa-engineer", prompt="Write unit and integration tests for API key auth")

# 4. Security engineer reviews
Task(agent="security-engineer", prompt="Review API key authentication implementation for security issues")

# 5. Code reviewer reviews
Task(agent="code-reviewer", prompt="Review authentication code for quality and best practices")

# 6. Docs engineer updates docs
Task(agent="docs-engineer", prompt="Document API key authentication in API docs")

# 7. DevOps deploys
Task(agent="devops-engineer", prompt="Deploy authentication feature to Cloud Run")
```

### Workflow 2: Infrastructure Change

**When to use**: Changing GCP infrastructure, adding services, region migrations

**Process**:
1. **Planning** (`tech-lead` + `devops-engineer`)
   - Assess impact
   - Plan migration strategy
   - Identify risks

2. **Implementation** (`devops-engineer`)
   - Update infrastructure
   - Update configuration files
   - Test connectivity

3. **Security Review** (`security-engineer`) - **REQUIRED**
   - Review IAM permissions
   - Check network security
   - Validate secrets management

4. **Testing** (`qa-engineer`) - **REQUIRED**
   - Write infrastructure tests
   - Test deployed service
   - Run smoke tests

5. **Documentation** (`docs-engineer`) - **REQUIRED**
   - Update DEPLOYMENT.md
   - Update ARCHITECTURE.md
   - Update developer onboarding guide
   - Update environment configs

6. **Code Review** (`code-reviewer`)
   - Review configuration changes
   - Check for hardcoded values
   - Verify documentation

### Workflow 3: Wave Completion

**When to use**: Completing a development Wave

**Process**:
1. **Implementation** (Various engineers)
   - Complete all Wave features
   - Ensure all functionality works

2. **Testing** (`qa-engineer`) - **REQUIRED**
   - Run full test suite
   - Verify 80%+ coverage
   - Run E2E tests on deployed service

3. **Security Review** (`security-engineer`) - **REQUIRED**
   - Comprehensive security audit
   - Vulnerability scanning
   - Check all new endpoints

4. **Code Review** (`code-reviewer`) - **REQUIRED**
   - Review all Wave changes
   - Check code quality
   - Suggest refactoring

5. **Documentation** (`docs-engineer`) - **REQUIRED**
   - Update Wave progress
   - Document all features
   - Update CHANGELOG.md
   - Update README.md

6. **Cleanup** (`code-cleaner`) - **REQUIRED**
   - Remove unused scripts
   - Delete temporary files
   - Consolidate duplicates
   - Clean up old test artifacts

7. **Retrospective** (`agent-orchestrator`)
   - Review what worked well
   - Identify improvements
   - Plan next Wave

### Workflow 4: Bug Fix

**When to use**: Fixing bugs or issues

**Process**:
1. **Investigation** (Relevant engineer)
   - Reproduce bug
   - Identify root cause
   - Design fix

2. **Implementation** (Relevant engineer)
   - Fix the bug
   - Add regression test

3. **Testing** (`qa-engineer`) - **REQUIRED**
   - Write test that catches the bug
   - Verify fix works
   - Run regression tests

4. **Code Review** (`code-reviewer`) - **REQUIRED if significant**
   - Review fix approach
   - Check for side effects

5. **Documentation** (`docs-engineer`) - **Optional**
   - Update troubleshooting guide if needed
   - Document fix in CHANGELOG.md

### Workflow 5: Pre-Deployment Checklist

**REQUIRED before EVERY deployment**

**Process**:
1. **Run Tests** (`qa-engineer`)
   ```bash
   # Unit tests
   pytest -m unit

   # Integration tests
   pytest -m integration
   ```

2. **Security Scan** (`security-engineer`)
   - Run security audit
   - Check for exposed secrets
   - Verify IAM permissions

3. **Code Review** (`code-reviewer`)
   - Final code review
   - Check for debug code
   - Verify logging

4. **Documentation Check** (`docs-engineer`)
   - Verify docs are up-to-date
   - Check CHANGELOG.md updated

5. **Deploy** (`devops-engineer`)
   ```bash
   # Deploy to Cloud Run
   gcloud run services replace clouddeploy.yaml --region=europe-west1
   ```

6. **Post-Deployment** (`qa-engineer`)
   ```bash
   # Run smoke tests
   pytest tests/test_e2e/test_smoke.py -v -m smoke

   # Run full E2E tests
   pytest tests/test_e2e/ -v -m "e2e and deployed"
   ```

## Agent Invocation Requirements

### REQUIRED Invocations

These invocations are **mandatory** and should never be skipped:

- **qa-engineer**: After implementing any feature or fix
- **security-engineer**: Before deployment, after infrastructure changes
- **code-reviewer**: Before merging significant changes, before deployment
- **docs-engineer**: After Wave completion, infrastructure changes, API changes
- **code-cleaner**: End of each Wave, before major releases

### OPTIONAL Invocations

These invocations are recommended but can be skipped for minor changes:

- **tech-lead**: Minor bug fixes (not needed)
- **product-manager**: Internal refactoring (not needed)
- **agent-orchestrator**: Single-feature tasks (not needed)

## Agent Dependencies

Understanding agent dependencies helps determine invocation order:

```
tech-lead (planning)
    ↓
backend-engineer / devops-engineer (implementation)
    ↓
qa-engineer (testing)
    ↓
security-engineer (security review)
    ↓
code-reviewer (quality review)
    ↓
docs-engineer (documentation)
    ↓
code-cleaner (cleanup) [Wave end only]
    ↓
devops-engineer (deployment)
    ↓
qa-engineer (E2E testing)
```

## Common Mistakes to Avoid

### ❌ Deploying without reviews
```bash
# Wrong approach
Make changes → gcloud run services replace clouddeploy.yaml
```

```bash
# Correct approach
Make changes → qa-engineer → security-engineer → code-reviewer → Deploy
```

### ❌ Skipping documentation updates
```bash
# Wrong approach
Add feature → Deploy → Done
```

```bash
# Correct approach
Add feature → Test → Review → Document → Deploy
```

### ❌ Not cleaning up after Waves
```bash
# Wrong approach
Complete Wave 1 → Start Wave 2 immediately
```

```bash
# Correct approach
Complete Wave 1 → Test → Review → Document → Cleanup → Start Wave 2
```

## Quick Reference

### For Feature Implementation:
```
1. tech-lead (design)
2. backend-engineer (implement)
3. qa-engineer (test) ✅ REQUIRED
4. security-engineer (security) ✅ REQUIRED
5. code-reviewer (review) ✅ REQUIRED
6. docs-engineer (document) ✅ REQUIRED
7. devops-engineer (deploy)
8. qa-engineer (E2E test) ✅ REQUIRED
```

### For Infrastructure Changes:
```
1. tech-lead + devops-engineer (plan)
2. devops-engineer (implement)
3. security-engineer (security) ✅ REQUIRED
4. qa-engineer (test) ✅ REQUIRED
5. docs-engineer (document) ✅ REQUIRED
6. code-reviewer (review)
```

### For Wave Completion:
```
1. qa-engineer (full test suite) ✅ REQUIRED
2. security-engineer (audit) ✅ REQUIRED
3. code-reviewer (review) ✅ REQUIRED
4. docs-engineer (update all docs) ✅ REQUIRED
5. code-cleaner (cleanup) ✅ REQUIRED
```

## Using Agents in Practice

### Invoking an Agent

Use the Task tool to invoke agents:

```python
Task(
    subagent_type="qa-engineer",
    description="Write tests for auth",
    prompt="""
    Write comprehensive tests for the API key authentication system:
    - Unit tests for middleware
    - Integration tests for protected endpoints
    - E2E tests for deployed service
    Target: 80%+ coverage
    """
)
```

### Parallel Agent Invocation

When agents don't depend on each other, invoke them in parallel:

```python
# Both can run simultaneously
Task(subagent_type="security-engineer", prompt="Review auth implementation")
Task(subagent_type="docs-engineer", prompt="Document auth API")
```

### Sequential Agent Invocation

When one agent depends on another's output:

```python
# First: implement
result = Task(subagent_type="backend-engineer", prompt="Implement auth")

# Then: test (needs implementation to be done)
Task(subagent_type="qa-engineer", prompt="Test auth implementation")
```

## Best Practices

1. **Always invoke required agents** - Don't skip security, testing, or code review
2. **Document changes** - Keep docs synchronized with code
3. **Clean up regularly** - Remove obsolete code after each Wave
4. **Test deployed services** - E2E tests catch integration issues
5. **Review before deploying** - Never deploy unreviewed code
6. **Update this guide** - Keep workflow documentation current

## Enforcement

To ensure this workflow is followed:

1. **Pre-deployment checklist** - Required before any deployment
2. **Wave completion checklist** - Required before starting next Wave
3. **Code review verification** - Reviewer checks that agents were invoked
4. **Documentation reviews** - Docs engineer verifies workflow was followed

## Questions?

If you're unsure which agents to invoke:

1. Check the "When to Invoke This Agent" section in the agent's definition file (`.claude/agents/*.md`)
2. Refer to the workflow diagrams above
3. Default to invoking more agents rather than fewer
4. Ask the tech-lead or agent-orchestrator for guidance

---

**Remember**: The agent workflow exists to ensure quality, security, and maintainability. Following it systematically prevents bugs, security issues, and technical debt.
