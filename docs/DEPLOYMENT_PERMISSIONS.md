# Cloud Run Deployment Permissions

## Overview
This document describes the IAM permissions configured for the `ai-agency-runner` service account to enable GitHub Actions deployment to Cloud Run.

## Service Account
- **Email**: `ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com`
- **Project**: `merlin-notebook-lm`
- **Purpose**: CI/CD deployment to Cloud Run via GitHub Actions

## Permissions Granted

### Project-Level IAM Roles

| Role | Purpose | Permissions Included |
|------|---------|---------------------|
| `roles/run.admin` | Full Cloud Run administration | `run.services.get`, `run.services.create`, `run.services.update`, `run.services.delete`, `run.services.setIamPolicy`, etc. |
| `roles/run.developer` | Cloud Run development and deployment | Deployment-focused permissions for services and revisions |
| `roles/artifactregistry.writer` | Push Docker images to Artifact Registry | `artifactregistry.repositories.uploadArtifacts`, `artifactregistry.repositories.downloadArtifacts` |
| `roles/iam.serviceAccountUser` | Act as service account | `iam.serviceAccounts.actAs`, required for Cloud Run to use the service account |
| `roles/cloudsql.client` | Connect to Cloud SQL | `cloudsql.instances.connect`, required for database access |
| `roles/secretmanager.secretAccessor` | Access secrets | `secretmanager.versions.access`, required for OPENAI_API_KEY and DATABASE_URL |
| `roles/storage.objectAdmin` | Manage GCS objects | `storage.objects.create`, `storage.objects.delete`, `storage.objects.get` |
| `roles/aiplatform.user` | Use Vertex AI | `aiplatform.endpoints.predict`, required for Vertex AI LLM calls |

### Service Account Self-Permissions

| Role | Purpose |
|------|---------|
| `roles/iam.serviceAccountTokenCreator` | Create tokens for itself (self-impersonation) |

## Key Permissions for Cloud Run Deployment

The following specific permissions are required for deployment and are included in the roles above:

- `run.services.get` - Read service configuration
- `run.services.create` - Create new services
- `run.services.update` - Update existing services
- `run.services.setIamPolicy` - Configure service IAM (if needed)
- `run.revisions.get` - Read revision information
- `run.revisions.list` - List revisions
- `resourcemanager.projects.get` - Access project metadata
- `artifactregistry.repositories.uploadArtifacts` - Push Docker images

## Verification

Run the verification script to confirm all permissions are correctly configured:

```bash
./scripts/verify_permissions.sh
```

Expected output: All checks should pass with ✓ marks.

## Troubleshooting

### Permission Denied Errors

If you see `PERMISSION_DENIED` errors after granting permissions:

1. **Wait for propagation**: IAM changes can take 60-80 seconds to propagate
2. **Verify service account key**: Ensure the `GCP_SA_KEY` GitHub secret contains the correct JSON key
3. **Check region**: Verify the service exists in the correct region (`europe-west1`)
4. **Review logs**: Check Cloud Run logs for additional details

```bash
# Test service account permissions manually
gcloud run services describe ai-agency --region=europe-west1 \
  --impersonate-service-account=ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com
```

### Missing Permissions

To check what permissions the service account has:

```bash
# List all project-level roles
gcloud projects get-iam-policy merlin-notebook-lm \
  --flatten="bindings[].members" \
  --filter="bindings.members:ai-agency-runner@merlin-notebook-lm.iam.gserviceaccount.com" \
  --format="table(bindings.role)"
```

## GitHub Actions Configuration

### Required Secrets

The following secrets must be configured in GitHub repository settings:

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `GCP_PROJECT_ID` | Google Cloud project ID | `gcloud config get-value project` |
| `GCP_SA_KEY` | Service account JSON key | Download from Cloud Console → IAM & Admin → Service Accounts |

### Workflow Configuration

The deployment workflow is configured in `.github/workflows/deploy.yml`:

- **Trigger**: Push to `main` branch or manual dispatch
- **Region**: `europe-west1`
- **Service**: `ai-agency`
- **Authentication**: Uses `google-github-actions/auth@v2` with service account JSON key

## Security Best Practices

1. **Least Privilege**: Service account has only the permissions needed for deployment
2. **Secret Management**: Sensitive values (API keys, database URLs) are stored in GCP Secret Manager, not GitHub secrets
3. **Service Account Key Rotation**: Rotate the service account key periodically
4. **Audit Logging**: Enable Cloud Audit Logs to track service account usage

## Related Documentation

- [Cloud Run IAM Roles](https://cloud.google.com/run/docs/reference/iam/roles)
- [Artifact Registry IAM](https://cloud.google.com/artifact-registry/docs/access-control)
- [GitHub Actions for GCP](https://github.com/google-github-actions)

## Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2025-10-28 | Added `roles/run.developer` | Additional deployment permissions |
| 2025-10-28 | Added `roles/iam.serviceAccountTokenCreator` on service account | Enable self-impersonation if needed |
| Previous | Initial roles granted | `roles/run.admin`, `roles/artifactregistry.writer`, etc. |
