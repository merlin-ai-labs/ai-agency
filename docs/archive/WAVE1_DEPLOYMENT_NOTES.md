# Wave 1 Deployment Notes

**Date**: 2025-10-28
**Status**: Deployed to Production
**URL**: https://ai-agency-4ebxrg4hdq-ew.a.run.app

---

## Deployment Summary

Wave 1 deployment was successfully completed to Google Cloud Run. The service is live and operational with the following components:

### Infrastructure

- **Platform**: Google Cloud Run
- **Region**: europe-west1 (Belgium)
- **Service Name**: ai-agency
- **Service URL**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Container Image**: europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency/app:latest

### Database

- **Type**: Cloud SQL PostgreSQL 15
- **Instance Name**: ai-agency-db
- **Region**: europe-west1
- **Connection**: Unix socket via Cloud SQL Proxy
- **Extensions**: pgvector enabled

### Storage

- **GCS Bucket**: merlin-ai-agency-artifacts-eu
- **Region**: europe-west1
- **Purpose**: Artifact storage (future use)

### Secrets Management

All secrets stored in Secret Manager:
- `cloudsql-database-url`: Database connection string
- `openai-api-key`: OpenAI API key

### Service Account

- **Name**: ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com
- **Permissions**: Cloud Run, Cloud SQL, Secret Manager access

---

## Deployment Verification

### Working Endpoints

All of the following endpoints are operational:

```bash
# Swagger UI - OpenAPI documentation
curl -I https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
# Status: 200 OK

# OpenAPI Schema
curl -I https://ai-agency-4ebxrg4hdq-ew.a.run.app/openapi.json
# Status: 200 OK

# Create Run (POST)
curl -X POST https://ai-agency-4ebxrg4hdq-ew.a.run.app/runs \
  -H "Content-Type: application/json" \
  -d '{"flow_name":"test","tenant_id":"test","input_data":{}}'
# Status: 200 OK (returns stub response)

# Get Run (GET)
curl -I https://ai-agency-4ebxrg4hdq-ew.a.run.app/runs/test-run-id
# Status: 200 OK (returns stub response)
```

### Non-Working Endpoints

#### /healthz Endpoint Issue

**Problem**: The `/healthz` endpoint returns a 404 error.

**Evidence**:
```bash
$ curl -I https://ai-agency-4ebxrg4hdq-ew.a.run.app/healthz
HTTP/2 404
content-type: text/html; charset=UTF-8
```

The response is Google's generic 404 page, not a FastAPI 404 response:
```html
<!DOCTYPE html>
<html lang=en>
  <title>Error 404 (Not Found)!!1</title>
  ...
  <p>The requested URL <code>/healthz</code> was not found on this server.
```

**Verification that Endpoint Exists**:

1. **Code Verification** (`app/main.py`):
```python
@app.get("/healthz")
async def healthz():
    """Health check endpoint for Cloud Run."""
    return {"status": "ok", "service": "ai-agency"}
```

2. **OpenAPI Schema Verification**:
```bash
$ curl -s https://ai-agency-4ebxrg4hdq-ew.a.run.app/openapi.json | jq '.paths | keys'
[
  "/healthz",
  "/runs",
  "/runs/{run_id}"
]
```

The endpoint is defined in code and appears in the OpenAPI spec, but returns 404.

---

## Issue Analysis: /healthz 404 Error

### What We Know

1. **Endpoint is defined**: Present in `app/main.py`
2. **OpenAPI recognizes it**: Listed in `/openapi.json`
3. **Other endpoints work**: `/docs`, `/openapi.json`, `/runs` all return 200
4. **404 is from Cloud Run**: Returns Google's generic 404 page, not FastAPI's

### Possible Causes

#### 1. Cloud Run Health Check Path Conflict

Cloud Run may have special handling for certain paths like `/healthz`. If Cloud Run's own health check system intercepts this path before it reaches the container, it could cause a 404.

**Evidence**:
- Cloud Run uses `/` as default health check path
- Some platforms reserve `/health` or `/healthz` for their own use

**Investigation needed**:
- Check Cloud Run service YAML for health check configuration
- Try alternative health check paths (`/health`, `/api/health`, `/ready`)

#### 2. ASGI/Uvicorn Routing Issue

The FastAPI application might not be properly handling the `/healthz` route in the Cloud Run environment, even though it works locally.

**Evidence**:
- `/docs` and other routes work fine
- Problem is specific to `/healthz`

**Investigation needed**:
- Check Uvicorn startup logs in Cloud Run
- Test with alternative ASGI servers (hypercorn, daphne)
- Add logging to the healthz handler

#### 3. Cloud Run Service Configuration

The Cloud Run service might have a configuration issue that prevents certain paths from reaching the container.

**Investigation needed**:
- Review clouddeploy.yaml configuration
- Check Cloud Run revision settings
- Verify container port mapping (8080)

#### 4. FastAPI Route Priority

FastAPI might be handling routes in an unexpected order, though this seems unlikely given `/docs` works.

**Investigation needed**:
- Check if any middleware is filtering the request
- Verify route registration order
- Test with explicit route prefix

### Immediate Impact

**Low Priority**: The issue does not affect core functionality:
- API endpoints work correctly (`/runs`)
- Documentation is accessible (`/docs`)
- Service is operational

**Workarounds Available**:
1. Use `/docs` for service availability checks (returns 200 when healthy)
2. Use `/openapi.json` for automated health checks
3. Use `/runs` endpoint with GET request (returns 200)

---

## Recommendations for Wave 2

### Investigation Steps

1. **Review Cloud Run logs**:
```bash
gcloud run services logs read ai-agency \
  --region=europe-west1 \
  --limit=100 \
  | grep -i health
```

2. **Test alternative health check paths**:
- Try `/health` instead of `/healthz`
- Try `/api/v1/health`
- Try `/ready` and `/live` (Kubernetes convention)

3. **Add debug logging**:
```python
@app.get("/healthz")
async def healthz():
    logger.info("healthz_endpoint_called")
    return {"status": "ok", "service": "ai-agency"}
```

4. **Check Cloud Run service configuration**:
```bash
gcloud run services describe ai-agency \
  --region=europe-west1 \
  --format=yaml > service-config.yaml
```

5. **Test locally with Cloud Run emulator**:
```bash
docker run -p 8080:8080 \
  europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency/app:latest
curl http://localhost:8080/healthz
```

### Potential Fixes

1. **Change endpoint name**:
```python
@app.get("/health")  # Try without 'z'
async def health():
    return {"status": "ok"}
```

2. **Add explicit startup health check in Dockerfile**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

3. **Configure Cloud Run health check explicitly**:
```yaml
# In clouddeploy.yaml
annotations:
  run.googleapis.com/health-check-path: "/health"
```

4. **Use FastAPI's lifespan events**:
```python
@app.on_event("startup")
async def startup_event():
    logger.info("app_started", endpoints=app.routes)
```

---

## Testing After Deployment

### Manual Testing Checklist

- [x] Swagger UI accessible
- [x] OpenAPI schema accessible
- [x] POST /runs returns stub response
- [x] GET /runs/{run_id} returns stub response
- [ ] GET /healthz returns 200 (FAILING)

### Automated Testing

GitHub Actions workflows are running successfully:
- **CI Pipeline** (`ci.yml`): All tests pass
- **Deployment Pipeline** (`deploy.yml`): Deploys successfully
- **E2E Tests** (`e2e-tests.yml`): Smoke tests pass (using /docs as health check)

---

## Documentation Updates Completed

- [x] Updated README.md with production URL
- [x] Added Known Issues section to README.md
- [x] Updated DEPLOYMENT.md with production status
- [x] Documented /healthz issue in DEPLOYMENT.md
- [x] Updated ARCHITECTURE.md with deployment info
- [x] Updated CHANGELOG.md with v0.2.1 release notes
- [x] Created this deployment notes document

---

## Next Steps (Wave 2)

1. **Resolve /healthz issue**:
   - Investigate root cause
   - Implement fix
   - Update documentation

2. **Enhanced health checks**:
   - Add database connectivity check
   - Add LLM provider availability check
   - Add more detailed status response

3. **Monitoring setup**:
   - Configure Cloud Monitoring alerts
   - Set up log-based metrics
   - Create uptime checks

4. **Database implementation**:
   - Connect to Cloud SQL
   - Run Alembic migrations
   - Test database operations

---

## Contact

For questions about this deployment:
- Review [DEPLOYMENT.md](DEPLOYMENT.md)
- Check [ARCHITECTURE.md](ARCHITECTURE.md)
- See [CHANGELOG.md](CHANGELOG.md) for version history

---

**Document Status**: Complete
**Next Review**: Wave 2 kickoff
