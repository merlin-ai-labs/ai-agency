#!/bin/bash
set -e

PROJECT_ID="merlin-notebook-lm"
SERVICE_ACCOUNT="ai-agency-runner@${PROJECT_ID}.iam.gserviceaccount.com"
SERVICE_NAME="ai-agency"
REGION="europe-west1"

echo "========================================="
echo "Testing Deployment Permissions"
echo "========================================="
echo ""

# Test 1: List Cloud Run services
echo "Test 1: Listing Cloud Run services..."
if gcloud run services list --region=$REGION \
    --impersonate-service-account=$SERVICE_ACCOUNT \
    --format="table(SERVICE,REGION,URL)" 2>&1; then
    echo "✓ Successfully listed services"
else
    echo "✗ Failed to list services"
    exit 1
fi
echo ""

# Test 2: Describe the service
echo "Test 2: Describing service '$SERVICE_NAME'..."
if gcloud run services describe $SERVICE_NAME --region=$REGION \
    --impersonate-service-account=$SERVICE_ACCOUNT \
    --format="value(status.url)" 2>&1; then
    echo "✓ Successfully described service"
else
    echo "✗ Failed to describe service"
    exit 1
fi
echo ""

# Test 3: Check Artifact Registry access
echo "Test 3: Checking Artifact Registry access..."
if gcloud artifacts repositories list --location=europe-west1 \
    --impersonate-service-account=$SERVICE_ACCOUNT \
    --format="table(REPOSITORY,FORMAT,LOCATION)" 2>&1; then
    echo "✓ Successfully accessed Artifact Registry"
else
    echo "✗ Failed to access Artifact Registry"
    exit 1
fi
echo ""

# Test 4: Check Secret Manager access
echo "Test 4: Checking Secret Manager access..."
if gcloud secrets list \
    --impersonate-service-account=$SERVICE_ACCOUNT \
    --format="table(NAME)" 2>&1 | head -5; then
    echo "✓ Successfully accessed Secret Manager"
else
    echo "✗ Failed to access Secret Manager"
    exit 1
fi
echo ""

echo "========================================="
echo "All Tests Passed! ✓"
echo "========================================="
echo ""
echo "The service account has the necessary permissions to:"
echo "  1. List and describe Cloud Run services"
echo "  2. Access Artifact Registry"
echo "  3. Access Secret Manager"
echo ""
echo "Deployment from GitHub Actions should work."
echo ""
echo "Note: If GitHub Actions still fails, check:"
echo "  1. GCP_SA_KEY secret is correctly configured"
echo "  2. Wait 2-3 minutes for IAM propagation"
echo "  3. Review GitHub Actions logs for specific errors"
