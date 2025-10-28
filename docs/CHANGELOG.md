# Changelog

All notable changes to the AI Consulting Agency Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Agent Improvements
- Created **docs-engineer** agent for systematic documentation maintenance
- Created **code-cleaner** agent for codebase cleanup and script consolidation
- Enhanced **qa-engineer** agent with E2E testing for deployed Cloud Run services
- Added AGENT_WORKFLOW.md comprehensive workflow guide

## [0.2.1] - 2025-10-28

### Wave 1 Deployment Complete

#### Deployed
- **Production URL**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Region**: europe-west1 (Belgium)
- **Status**: Live and operational

#### Features Available
- FastAPI application with Swagger UI at `/docs`
- OpenAPI schema at `/openapi.json`
- Stub endpoints for `/runs` (POST and GET)
- Auto-deployment via GitHub Actions

#### Known Issues
- `/healthz` endpoint returns 404 (Cloud Run routing issue)
  - Investigation: Endpoint defined in code and shows in OpenAPI spec
  - Symptom: Cloud Run returns Google 404 page instead of reaching app
  - Workaround: Use `/docs` endpoint to verify service availability
  - Impact: Low (monitoring can use `/docs` or `/openapi.json` instead)
  - Status: To be investigated in Wave 2

#### Infrastructure
- Cloud Run service deployed successfully
- Cloud SQL PostgreSQL 15 instance running (ai-agency-db)
- GCS bucket configured (merlin-ai-agency-artifacts-eu)
- Secrets Manager storing credentials
- Service account: ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com

#### Documentation Updates
- Updated all URLs to production deployment
- Documented /healthz routing issue
- Added Known Issues section to README.md
- Updated deployment information across all docs

## [0.2.0] - 2025-10-28

### Infrastructure Migration

#### Changed
- **BREAKING**: Migrated all GCP services from us-central1 to europe-west1 (Belgium)
- Updated Artifact Registry to europe-west1-docker.pkg.dev
- Updated GCS bucket to merlin-ai-agency-artifacts-eu
- Migrated all configuration files to europe-west1

#### Added
- **Cloud SQL**: PostgreSQL 15 instance (ai-agency-db) in europe-west1
- **Automatic Deployment**: GitHub Actions auto-deploy on push to main
- **E2E Testing**: Post-deployment smoke tests and full E2E test suite
- **Secrets Management**: DATABASE_URL and OpenAI API key in Secret Manager
- **Service Account**: ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com

#### Deployment
- **Cloud Run Service**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Swagger UI**: /docs endpoint
- **Health Check**: /healthz endpoint
- **OpenAPI Schema**: /openapi.json endpoint

### CI/CD

#### Added
- Automatic deployment workflow (.github/workflows/deploy.yml)
- E2E testing workflow (.github/workflows/e2e-tests.yml)
- Post-deployment smoke tests
- GitHub secrets configuration (GCP_SA_KEY, GCP_PROJECT_ID)

#### Changed
- Deploy workflow now triggers on push to main (not manual only)
- Docker images built for linux/amd64 platform
- Deployment target: Cloud Run in europe-west1

### Documentation

#### Added
- AGENT_WORKFLOW.md - Comprehensive agent invocation workflow guide
- E2E test documentation in qa-engineer agent
- Deployed service testing procedures
- Smoke test templates

#### Updated
- README.md - Added production deployment URLs and infrastructure details
- DEVELOPER_ONBOARDING.md - Updated all gcloud commands to europe-west1
- .env.example - Updated region and bucket references
- All documentation references from us-central1 to europe-west1

### Testing

#### Added
- E2E test markers (@pytest.mark.e2e, @pytest.mark.deployed, @pytest.mark.smoke)
- Deployed service test fixtures (tests/test_e2e/conftest.py)
- Smoke tests for post-deployment validation (tests/test_e2e/test_smoke.py)
- Full E2E test suite for Cloud Run service (tests/test_e2e/test_deployed_service.py)
- Manual E2E test runner script (scripts/run_e2e_tests.sh)

#### Changed
- Updated pytest.ini with new E2E test markers
- Enhanced qa-engineer agent with deployed service testing responsibilities

### Configuration

#### Changed
- Updated .env.example with europe-west1 region
- Updated .env.gcp for production deployment
- Updated clouddeploy.yaml with Cloud SQL connection and secrets
- Updated scripts/deploy.sh to europe-west1
- Updated scripts/setup_gcp.sh to europe-west1

### Security

#### Added
- OpenAI API key stored in Secret Manager (openai-api-key)
- Database URL stored in Secret Manager (cloudsql-database-url)
- Database password stored in Secret Manager
- Service account with minimal IAM permissions

#### Changed
- Removed hardcoded credentials from configuration
- All secrets now managed via GCP Secret Manager
- Cloud Run service uses service account for authentication

## [0.1.0] - 2025-10-25

### Wave 1 - Foundation

#### Infrastructure
- ✅ PostgreSQL 15 + pgvector database setup
- ✅ Docker containerization with multi-stage builds
- ✅ FastAPI application structure
- ✅ Basic health check endpoint
- ✅ Alembic migration system
- ✅ Testing framework (pytest)

#### Development
- ✅ Wave-based development methodology (6 waves)
- ✅ 10 specialized AI agents for development
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Code quality tools (ruff, mypy, pytest)
- ✅ Development environment setup

#### Documentation
- ✅ ARCHITECTURE.md - Technical architecture
- ✅ CODING_STANDARDS.md - Python standards
- ✅ DEPLOYMENT.md - Multi-cloud deployment guide
- ✅ CODE_REVIEW_CHECKLIST.md - Review guidelines
- ✅ WAVE1_REVIEW.md - Wave 1 completion review
- ✅ DEVELOPER_ONBOARDING.md - Team onboarding guide

---

## Version History

### Version Numbering
- **Major version (X.0.0)**: Breaking changes, major feature releases
- **Minor version (0.X.0)**: New features, infrastructure changes
- **Patch version (0.0.X)**: Bug fixes, documentation updates

### Status Legend
- ✅ Complete
- 🚧 In Progress
- ⏳ Pending
- ❌ Deprecated

---

## Migration Notes

### us-central1 → europe-west1 Migration (v0.2.0)

**IMPORTANT**: If you were using the us-central1 infrastructure, all resources have been migrated to europe-west1.

**Action Required**:
1. Update local .env file with new region: `VERTEX_AI_LOCATION=europe-west1`
2. Update GCS bucket: `GCS_BUCKET=merlin-ai-agency-artifacts-eu`
3. Update Docker auth: `gcloud auth configure-docker europe-west1-docker.pkg.dev`
4. Update deployment commands to use europe-west1

**Old resources** (us-central1):
- No longer active
- Can be safely deleted

**New resources** (europe-west1):
- Cloud Run: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- Artifact Registry: europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency
- GCS: merlin-ai-agency-artifacts-eu
- Cloud SQL: ai-agency-db (europe-west1)

---

## Upcoming Changes

### Wave 2 - Core Services (Planned)
- Database repositories and models
- LLM provider adapters (OpenAI, Vertex AI)
- Multi-LLM support testing
- Database migration execution

### Future Infrastructure
- Additional regions for multi-region deployment
- Load balancing and auto-scaling configuration
- Production database backups and disaster recovery
- Monitoring and alerting setup

---

For detailed documentation, see:
- [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md) - Development workflow
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
