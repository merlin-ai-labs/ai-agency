#!/bin/bash
set -e

# AI Agency Platform - Quick Deploy Script
# Builds and deploys to Cloud Run in one command

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================="
echo "AI Agency Platform - Quick Deploy"
echo "========================================="
echo ""

# Configuration
PROJECT_ID="merlin-notebook-lm"
REGION="us-central1"
ARTIFACT_REPO="ai-agency"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/app"

# Get version tag (use git tag or default to latest)
VERSION=$(git describe --tags --always 2>/dev/null || echo "latest")
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"
IMAGE_LATEST="${IMAGE_NAME}:latest"

echo "Deploying version: $VERSION"
echo "Image: $IMAGE_TAG"
echo ""

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo -e "${RED}✗ Not authenticated with gcloud${NC}"
    echo "  Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID --quiet

# Step 1: Build
echo "========================================="
echo "Step 1/3: Building Docker image"
echo "========================================="
docker build \
  --platform linux/amd64 \
  -t $IMAGE_TAG \
  -t $IMAGE_LATEST \
  .
echo -e "${GREEN}✓ Image built${NC}"
echo ""

# Step 2: Push
echo "========================================="
echo "Step 2/3: Pushing to Artifact Registry"
echo "========================================="
docker push $IMAGE_TAG
docker push $IMAGE_LATEST
echo -e "${GREEN}✓ Image pushed${NC}"
echo ""

# Step 3: Deploy
echo "========================================="
echo "Step 3/3: Deploying to Cloud Run"
echo "========================================="

# Update clouddeploy.yaml with new image
sed -i.bak "s|image: .*|image: ${IMAGE_LATEST}|" clouddeploy.yaml
rm clouddeploy.yaml.bak

# Deploy
gcloud run services replace clouddeploy.yaml --region=$REGION

echo -e "${GREEN}✓ Deployed successfully${NC}"
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe ai-agency \
  --region=$REGION \
  --format='value(status.url)')

echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test endpoints:"
echo "  Health check: curl $SERVICE_URL/health"
echo "  API docs:     $SERVICE_URL/docs"
echo ""
echo "View logs:"
echo "  gcloud run services logs read ai-agency --region=$REGION --limit=50"
echo ""
