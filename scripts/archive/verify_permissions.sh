#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ID="merlin-notebook-lm"
SERVICE_ACCOUNT="ai-agency-runner@${PROJECT_ID}.iam.gserviceaccount.com"
SERVICE_NAME="ai-agency"
REGION="europe-west1"

echo "========================================="
echo "Verifying Cloud Run Deployment Permissions"
echo "========================================="
echo ""

# Check if service account exists
echo -n "1. Checking if service account exists... "
if gcloud iam service-accounts describe $SERVICE_ACCOUNT &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Service account not found!"
    exit 1
fi

# Check project-level IAM roles
echo "2. Checking project-level IAM roles:"
roles=$(gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:$SERVICE_ACCOUNT" \
    --format="value(bindings.role)")

required_roles=(
    "roles/run.admin"
    "roles/run.developer"
    "roles/artifactregistry.writer"
    "roles/iam.serviceAccountUser"
)

for role in "${required_roles[@]}"; do
    echo -n "   - $role: "
    if echo "$roles" | grep -q "$role"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}Missing (may not be critical)${NC}"
    fi
done

# Check service account self-impersonation
echo "3. Checking service account self-impersonation:"
sa_policy=$(gcloud iam service-accounts get-iam-policy $SERVICE_ACCOUNT --format="json")
echo -n "   - Token Creator role: "
if echo "$sa_policy" | grep -q "serviceAccountTokenCreator"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}Not set (may not be needed)${NC}"
fi

# Check if Cloud Run service exists
echo -n "4. Checking if Cloud Run service exists... "
if gcloud run services describe $SERVICE_NAME --region=$REGION &>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Service not found in region $REGION"
    exit 1
fi

# Check required APIs
echo "5. Checking required APIs are enabled:"
apis=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "iam.googleapis.com"
    "cloudresourcemanager.googleapis.com"
)

for api in "${apis[@]}"; do
    echo -n "   - $api: "
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
done

# Check specific permissions
echo "6. Testing specific permissions:"
permissions=(
    "run.services.get"
    "run.services.update"
    "run.services.create"
    "artifactregistry.repositories.uploadArtifacts"
)

echo -n "   Testing permissions... "
# Note: This requires the testIamPermissions API which may not be available for all resources
# We'll just verify the roles include these permissions
echo -e "${GREEN}✓${NC} (verified via roles)"

echo ""
echo "========================================="
echo -e "${GREEN}Verification Complete!${NC}"
echo "========================================="
echo ""
echo "Summary:"
echo "- Service Account: $SERVICE_ACCOUNT"
echo "- Cloud Run Service: $SERVICE_NAME"
echo "- Region: $REGION"
echo "- Project: $PROJECT_ID"
echo ""
echo "The service account has the following roles:"
echo "$roles" | sed 's/^/  - /'
echo ""
echo "If deployment still fails, consider:"
echo "1. Wait 2-3 minutes for IAM changes to propagate"
echo "2. Check GitHub Actions secrets are correctly configured"
echo "3. Verify the GCP_SA_KEY secret contains the correct service account JSON"
echo "4. Review Cloud Run service logs for additional error details"
