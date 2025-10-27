# Agent Coordination Plan

This document outlines the execution plan for the development team agents.

## Team Structure

We have **10 specialized agents** working together:

1. **tech-lead** - Standards, base classes, patterns
2. **devops-engineer** - CI/CD, Docker, Alembic
3. **database-engineer** - DB sessions, models, CRUD
4. **llm-engineer** - LLM adapters (OpenAI, Vertex), storage
5. **rag-engineer** - RAG system with pgvector
6. **tools-engineer** - All 5 tools + registry
7. **flows-engineer** - Both flows + orchestration
8. **qa-engineer** - Comprehensive testing
9. **security-engineer** - Security audit, auth, rate limiting
10. **code-reviewer** - Final review and approval

## Execution Waves

### **Wave 1: Foundation** (Launch Together)
**Agents:** `tech-lead`, `devops-engineer`

**Why Together:** They establish the foundation without dependencies on each other.

**Expected Duration:** 30-45 minutes

**Deliverables:**
- Coding standards and base classes (tech-lead)
- CI/CD, Docker, Alembic setup (devops-engineer)

**Launch Command:**
```
Use the tech-lead and devops-engineer agents to establish the foundation.
```

---

### **Wave 2: Core Services** (Launch Together After Wave 1)
**Agents:** `database-engineer`, `llm-engineer`

**Dependencies:**
- Requires tech-lead's base classes
- Requires devops-engineer's Alembic setup

**Expected Duration:** 45-60 minutes

**Deliverables:**
- Database session management, CRUD operations
- LLM adapters (OpenAI, Vertex), GCS storage

**Launch Command:**
```
Use the database-engineer and llm-engineer agents to implement core services.
```

---

### **Wave 3: RAG System** (Launch After Wave 2)
**Agents:** `rag-engineer`

**Dependencies:**
- Requires database-engineer's session management
- Requires llm-engineer's embeddings adapter

**Expected Duration:** 45-60 minutes

**Deliverables:**
- RAG ingestion and retrieval
- pgvector integration
- Document chunking

**Launch Command:**
```
Use the rag-engineer agent to implement the RAG system.
```

---

### **Wave 4: Business Logic** (Launch Together After Wave 3)
**Agents:** `tools-engineer`, then `flows-engineer`

**Dependencies:**
- tools-engineer requires: llm-engineer, rag-engineer
- flows-engineer requires: tools-engineer completion

**Expected Duration:** 90-120 minutes

**Deliverables:**
- All 5 tools implemented
- Both flows implemented
- Execution loop working

**Launch Commands:**
```
# First launch tools-engineer
Use the tools-engineer agent to implement all tools.

# Then after completion, launch flows-engineer
Use the flows-engineer agent to implement both flows.
```

---

### **Wave 5: Quality Assurance** (Launch Together After Wave 4)
**Agents:** `qa-engineer`, `security-engineer`, `docs-engineer`

**Dependencies:** All implementation complete

**Expected Duration:** 60-90 minutes

**Deliverables:**
- Comprehensive test suite (80%+ coverage)
- Security features and audit report
- Complete documentation

**Launch Command:**
```
Use the qa-engineer, security-engineer, and docs-engineer agents to complete quality assurance.
```

---

### **Wave 6: Final Review** (Launch After Wave 5)
**Agents:** `code-reviewer`

**Dependencies:** Everything complete

**Expected Duration:** 30-45 minutes

**Deliverables:**
- Code review report
- Production readiness assessment
- Final approval or list of blockers

**Launch Command:**
```
Use the code-reviewer agent to perform final code review.
```

---

## Total Estimated Time

**Sequential:** 5-7 hours
**With Parallel Execution:** 3-4 hours

## How to Execute

### Option A: Automated (Recommended)
Run each wave sequentially, launching agents in parallel where possible.

### Option B: Manual
Launch each agent individually and wait for completion.

### Option C: Hybrid
Launch foundation and core services, then manually coordinate the rest.

## Agent Invocation

Agents can be invoked by:
1. **Explicit call**: "Use the tech-lead agent to..."
2. **Automatic**: Agents with "MUST BE USED" in description are auto-invoked

## Progress Tracking

After each wave, review:
1. What was completed
2. What issues were encountered
3. Whether next wave can proceed

## Integration Points

Each agent creates files that others depend on:

```
tech-lead → app/core/ (base classes)
  ↓
devops-engineer → CI/CD, Alembic
  ↓
database-engineer → app/db/session.py
llm-engineer → app/adapters/*.py
  ↓
rag-engineer → app/rag/*.py
  ↓
tools-engineer → app/tools/*/v1/
  ↓
flows-engineer → app/flows/*/graph.py
  ↓
qa-engineer → tests/
security-engineer → app/core/security.py
docs-engineer → docs/
  ↓
code-reviewer → Final report
```

## Conflict Resolution

If agents create conflicting code:
1. Later agents should override earlier agents
2. Review conflicts manually
3. Prioritize functionality over style
4. Ensure tests pass after resolution

## Success Criteria

✅ All tests passing
✅ 80%+ test coverage
✅ No critical security issues
✅ All documentation complete
✅ Code reviewer approval

## Ready to Launch

Start with:
```
Use the tech-lead and devops-engineer agents to establish the foundation.
```
