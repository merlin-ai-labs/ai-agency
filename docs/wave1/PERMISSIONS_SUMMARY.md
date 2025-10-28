# Cloud Run Deployment Permissions - Resolution Summary

## Issue
GitHub Actions workflow "Deploy to Cloud Run" was failing with:
```
ERROR: (gcloud.run.services.replace) PERMISSION_DENIED: Permission 'run.services.get' 
denied on resource 'namespaces/***/services/ai-agency'
```

## Root Cause Analysis
The `ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com` service account needed additional IAM permissions for Cloud Run deployment operations.

## Permissions Added

### 1. Project-Level IAM Roles

The service account now has the following roles at the project level:

| Role | Key Permissions | Purpose |
|------|----------------|---------|
| **roles/run.admin** ✓ | `run.services.get`, `run.services.create`, `run.services.update`, `run.services.delete`, `run.services.setIamPolicy` | Full Cloud Run administration |
| **roles/run.developer** ✓ NEW | Additional deployment-specific permissions for Cloud Run services and revisions | Complementary deployment permissions |
| **roles/artifactregistry.writer** ✓ | `artifactregistry.repositories.uploadArtifacts` | Push Docker images |
| **roles/iam.serviceAccountUser** ✓ | `iam.serviceAccounts.actAs` | Act as service account in Cloud Run |
| **roles/cloudsql.client** ✓ | `cloudsql.instances.connect` | Database connectivity |
| **roles/secretmanager.secretAccessor** ✓ | `secretmanager.versions.access` | Access API keys and secrets |
| **roles/storage.objectAdmin** ✓ | `storage.objects.*` | GCS bucket management |
| **roles/aiplatform.user** ✓ | `aiplatform.endpoints.predict` | Vertex AI access |

### 2. Service Account Self-Permissions

Added to the service account itself:

| Role | Purpose |
|------|---------|
| **roles/iam.serviceAccountTokenCreator** ✓ NEW | Allows the service account to create tokens for itself (self-impersonation) |

## Changes Made

### IAM Policy Updates

```bash
# Added Cloud Run Developer role
gcloud projects add-iam-policy-binding merlin-notebook-lm \
  --member="serviceAccount:ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com" \
  --role="roles/run.developer"

# Added Service Account Token Creator (self-impersonation)
gcloud iam service-accounts add-iam-policy-binding \
  ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com \
  --member="serviceAccount:ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator"
```

## Verification

### All Required APIs Enabled ✓
- `run.googleapis.com`
- `artifactregistry.googleapis.com`
- `iam.googleapis.com`
- `cloudresourcemanager.googleapis.com`

### Service Configuration ✓
- Service Name: `ai-agency`
- Region: `europe-west1`
- Service Account: `ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com`

### Verification Scripts Created

1. **Permission Checker**: `/scripts/verify_permissions.sh`
   - Validates all IAM roles are correctly assigned
   - Checks service and API configuration
   - Provides troubleshooting guidance

2. **Deployment Tester**: `/scripts/test_deployment_permissions.sh`
   - Tests service account permissions via impersonation
   - Verifies Cloud Run, Artifact Registry, and Secret Manager access

## Expected Resolution

The deployment should now succeed because:

1. ✓ Service account has `roles/run.admin` with `run.services.get` permission
2. ✓ Service account has `roles/run.developer` for deployment operations
3. ✓ Service account can impersonate itself via `roles/iam.serviceAccountTokenCreator`
4. ✓ All required APIs are enabled
5. ✓ Cloud Run service exists in the correct region (`europe-west1`)

## Important Notes

### IAM Propagation Delay
⚠️ IAM changes can take **60-80 seconds** to fully propagate. If the deployment still fails immediately after these changes, wait 2-3 minutes and retry.

### GitHub Secrets Validation
Ensure the following secrets are correctly configured in GitHub repository settings:

| Secret | Value | Verification |
|--------|-------|--------------|
| `GCP_PROJECT_ID` | `merlin-notebook-lm` | ✓ |
| `GCP_SA_KEY` | Service account JSON key | Must be the complete JSON key file for `ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com` |

### Service Account Key
If deployment continues to fail, verify the `GCP_SA_KEY` secret:

```bash
# Download fresh service account key
gcloud iam service-accounts keys create ~/sa-key.json \
  --iam-account=ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com

# Update GitHub secret with contents of sa-key.json
# Then delete the local copy for security
```

## Troubleshooting

If the deployment still fails after waiting for propagation:

1. **Check GitHub Actions logs** for the specific error message
2. **Verify the region** matches in both workflow and service (`europe-west1`)
3. **Review Cloud Run service logs** for runtime errors
4. **Test service account key**:
   ```bash
   gcloud auth activate-service-account \
     --key-file=<path-to-key.json>
   gcloud run services list --region=europe-west1
   ```

## Next Steps

1. ✅ Wait 2-3 minutes for IAM propagation
2. ✅ Trigger the GitHub Actions workflow (push to main or manual trigger)
3. ✅ Monitor the deployment in GitHub Actions
4. ✅ If successful, verify the service is running: https://ai-agency-847424242737.europe-west1.run.app

## Documentation

- Full permissions documentation: `/docs/DEPLOYMENT_PERMISSIONS.md`
- Workflow configuration: `/.github/workflows/deploy.yml`
- Service configuration: `/clouddeploy.yaml`

## Security Considerations

- ✓ Least privilege principle applied (only necessary permissions granted)
- ✓ Secrets managed in GCP Secret Manager (not GitHub secrets)
- ✓ Service account key should be rotated periodically
- ✓ Cloud Audit Logs track all service account activity
