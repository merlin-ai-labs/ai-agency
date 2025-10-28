# Cloud Run Deployment Checklist

Quick reference checklist for troubleshooting Cloud Run deployments.

## Pre-Deployment Verification

### 1. Service Account Permissions ✓

Run the verification script:
```bash
./scripts/verify_permissions.sh
```

**Required roles on project** (`merlin-notebook-lm`):
- [x] `roles/run.admin`
- [x] `roles/run.developer`
- [x] `roles/artifactregistry.writer`
- [x] `roles/iam.serviceAccountUser`
- [x] `roles/cloudsql.client`
- [x] `roles/secretmanager.secretAccessor`
- [x] `roles/storage.objectAdmin`
- [x] `roles/aiplatform.user`

**Self-impersonation permission**:
- [x] `roles/iam.serviceAccountTokenCreator` on the service account itself

### 2. GitHub Secrets Configuration

**Repository Settings → Secrets and variables → Actions**

| Secret Name | Description | Status |
|-------------|-------------|--------|
| `GCP_PROJECT_ID` | Project ID: `merlin-notebook-lm` | ✓ |
| `GCP_SA_KEY` | Complete JSON key for `ai-agency-runner@...` | ⚠️ Verify |

**To verify GCP_SA_KEY is correct:**
1. Download a fresh key if unsure
2. Ensure it's the complete JSON (starts with `{` and ends with `}`)
3. Verify it's for the correct service account

### 3. GCP APIs Enabled ✓

```bash
gcloud services list --enabled --filter="name:(run OR artifactregistry OR iam)"
```

Required APIs:
- [x] `run.googleapis.com`
- [x] `artifactregistry.googleapis.com`
- [x] `iam.googleapis.com`
- [x] `cloudresourcemanager.googleapis.com`

### 4. Cloud Run Service Configuration

**Service details:**
- Name: `ai-agency`
- Region: `europe-west1`
- URL: https://ai-agency-4ebxrg4hdq-ew.a.run.app

**Verify service exists:**
```bash
gcloud run services describe ai-agency --region=europe-west1
```

## Deployment Process

### Manual Deployment Test

```bash
# 1. Build image locally
docker build -t europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency/app:test .

# 2. Configure Docker auth
gcloud auth configure-docker europe-west1-docker.pkg.dev

# 3. Push image
docker push europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency/app:test

# 4. Update clouddeploy.yaml with test tag
sed 's/:latest/:test/' clouddeploy.yaml > clouddeploy-test.yaml

# 5. Deploy
gcloud run services replace clouddeploy-test.yaml --region=europe-west1
```

### GitHub Actions Deployment

**Trigger options:**
1. **Automatic**: Push to `main` branch
2. **Manual**: Actions → Deploy to Cloud Run → Run workflow

**Monitoring:**
```bash
# Watch GitHub Actions logs in real-time
gh run watch

# Or view in browser
gh run view --web
```

## Common Issues & Solutions

### Issue 1: PERMISSION_DENIED
**Symptom**: `Permission 'run.services.get' denied`

**Solutions:**
1. ✓ Verify service account has `roles/run.admin` and `roles/run.developer`
2. ⏰ Wait 2-3 minutes for IAM propagation
3. 🔑 Verify `GCP_SA_KEY` secret is correct
4. 🔄 Re-generate and update service account key

### Issue 2: Image Not Found
**Symptom**: `Container image not found` or `Failed pulling image`

**Solutions:**
1. Verify image was pushed: `gcloud artifacts docker images list europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency`
2. Check Artifact Registry permissions: `roles/artifactregistry.writer`
3. Verify Docker authentication in workflow

### Issue 3: Service Timeout
**Symptom**: `Revision failed with health check timeout`

**Solutions:**
1. Check application starts within 240 seconds
2. Verify `/healthz` endpoint responds
3. Review Cloud Run logs: `gcloud run services logs read ai-agency --region=europe-west1`

### Issue 4: Database Connection Failed
**Symptom**: `Connection refused` or `could not connect to server`

**Solutions:**
1. Verify Cloud SQL connection annotation in `clouddeploy.yaml`
2. Check `DATABASE_URL` secret is correct
3. Confirm `roles/cloudsql.client` permission

### Issue 5: Secret Access Denied
**Symptom**: `Permission denied on secret` or `Secret not found`

**Solutions:**
1. Verify secrets exist: `gcloud secrets list`
2. Check `roles/secretmanager.secretAccessor` permission
3. Confirm secret names in `clouddeploy.yaml` match actual secrets

## Verification After Deployment

### 1. Service Health Check
```bash
SERVICE_URL=$(gcloud run services describe ai-agency \
  --region=europe-west1 \
  --format='value(status.url)')

curl -f "$SERVICE_URL/healthz"
```

Expected response: `200 OK`

### 2. Check Service Logs
```bash
gcloud run services logs read ai-agency \
  --region=europe-west1 \
  --limit=50
```

### 3. Test API Endpoints
```bash
# Health check
curl "$SERVICE_URL/healthz"

# API docs
curl "$SERVICE_URL/docs"

# Test endpoint (if available)
curl "$SERVICE_URL/api/v1/health"
```

### 4. Monitor Metrics
```bash
# Open Cloud Console metrics
gcloud run services describe ai-agency \
  --region=europe-west1 \
  --format='value(status.url)' | \
  sed 's|https://||' | \
  xargs -I {} echo "https://console.cloud.google.com/run/detail/europe-west1/ai-agency/metrics"
```

## Rollback Procedure

If deployment fails and service is down:

```bash
# List revisions
gcloud run revisions list --service=ai-agency --region=europe-west1

# Rollback to previous revision
PREVIOUS_REVISION=$(gcloud run revisions list \
  --service=ai-agency \
  --region=europe-west1 \
  --format='value(name)' \
  --limit=2 | tail -1)

gcloud run services update-traffic ai-agency \
  --to-revisions=$PREVIOUS_REVISION=100 \
  --region=europe-west1
```

## Emergency Contacts

- **GCP Console**: https://console.cloud.google.com/run?project=merlin-notebook-lm
- **GitHub Actions**: https://github.com/<org>/<repo>/actions
- **Cloud Run Logs**: https://console.cloud.google.com/logs/query?project=merlin-notebook-lm

## Quick Commands

```bash
# Check deployment status
gcloud run services describe ai-agency --region=europe-west1

# View recent logs
gcloud run services logs read ai-agency --region=europe-west1 --limit=50

# List recent revisions
gcloud run revisions list --service=ai-agency --region=europe-west1 --limit=5

# View service URL
gcloud run services describe ai-agency --region=europe-west1 --format='value(status.url)'

# Check IAM policy
gcloud run services get-iam-policy ai-agency --region=europe-west1
```
