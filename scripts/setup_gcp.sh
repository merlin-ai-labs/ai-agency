#!/bin/bash
set -e

# AI Agency Platform - GCP Setup Script
# Sets up GCP infrastructure for team deployment

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================="
echo "AI Agency Platform - GCP Setup"
echo "========================================="
echo ""

# Configuration
PROJECT_ID="merlin-notebook-lm"
REGION="us-central1"
BUCKET_NAME="merlin-ai-agency-artifacts"
ARTIFACT_REPO="ai-agency"

echo "Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Bucket: $BUCKET_NAME"
echo "  Artifact Registry: $ARTIFACT_REPO"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo -e "${RED}✗ Not authenticated with gcloud${NC}"
    echo "  Run: gcloud auth login"
    exit 1
fi
echo -e "${GREEN}✓ Authenticated with gcloud${NC}"

# Set project
echo ""
echo "Setting GCP project..."
gcloud config set project $PROJECT_ID
echo -e "${GREEN}✓ Project set to $PROJECT_ID${NC}"

# Enable required APIs
echo ""
echo "Enabling required GCP APIs..."
APIS=(
    "run.googleapis.com"                  # Cloud Run
    "artifactregistry.googleapis.com"     # Artifact Registry
    "storage.googleapis.com"              # Cloud Storage
    "secretmanager.googleapis.com"        # Secret Manager
    "aiplatform.googleapis.com"           # Vertex AI
    "sqladmin.googleapis.com"             # Cloud SQL (for future)
    "cloudbuild.googleapis.com"           # Cloud Build
)

for api in "${APIS[@]}"; do
    echo "  Enabling $api..."
    gcloud services enable $api --quiet
done
echo -e "${GREEN}✓ All APIs enabled${NC}"

# Create Artifact Registry repository
echo ""
echo "Creating Artifact Registry repository..."
if gcloud artifacts repositories describe $ARTIFACT_REPO --location=$REGION &>/dev/null; then
    echo -e "${YELLOW}⚠ Artifact Registry repository already exists${NC}"
else
    gcloud artifacts repositories create $ARTIFACT_REPO \
        --repository-format=docker \
        --location=$REGION \
        --description="AI Agency Platform container images" \
        --quiet
    echo -e "${GREEN}✓ Artifact Registry repository created${NC}"
fi

# Configure Docker authentication
echo ""
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
echo -e "${GREEN}✓ Docker authentication configured${NC}"

# Create GCS bucket
echo ""
echo "Creating GCS bucket for artifacts..."
if gsutil ls -b gs://$BUCKET_NAME &>/dev/null; then
    echo -e "${YELLOW}⚠ GCS bucket already exists${NC}"
else
    gsutil mb -l $REGION gs://$BUCKET_NAME
    # Enable versioning for safety
    gsutil versioning set on gs://$BUCKET_NAME
    echo -e "${GREEN}✓ GCS bucket created with versioning enabled${NC}"
fi

# Create Secret Manager secrets
echo ""
echo "Setting up Secret Manager..."

# Check if OPENAI_API_KEY is in .env
if [ -f ".env" ] && grep -q "OPENAI_API_KEY" .env; then
    OPENAI_KEY=$(grep "OPENAI_API_KEY" .env | cut -d '=' -f2 | tr -d '"' | tr -d "'")

    if [ -n "$OPENAI_KEY" ] && [ "$OPENAI_KEY" != "your-key-here" ]; then
        echo "  Storing OpenAI API key in Secret Manager..."
        if gcloud secrets describe openai-api-key &>/dev/null; then
            echo -e "${YELLOW}  ⚠ Secret already exists, creating new version${NC}"
            echo -n "$OPENAI_KEY" | gcloud secrets versions add openai-api-key --data-file=-
        else
            echo -n "$OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=-
        fi
        echo -e "${GREEN}  ✓ OpenAI API key stored${NC}"
    else
        echo -e "${YELLOW}  ⚠ No valid OpenAI API key in .env, skipping${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ .env file not found or no OpenAI key, skipping${NC}"
fi

# Service account for Cloud Run (optional but recommended)
echo ""
echo "Creating service account for Cloud Run..."
SERVICE_ACCOUNT="ai-agency-runner"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL &>/dev/null; then
    echo -e "${YELLOW}⚠ Service account already exists${NC}"
else
    gcloud iam service-accounts create $SERVICE_ACCOUNT \
        --display-name="AI Agency Cloud Run Service Account" \
        --quiet
    echo -e "${GREEN}✓ Service account created${NC}"
fi

# Grant necessary permissions
echo "  Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --quiet &>/dev/null

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.user" \
    --quiet &>/dev/null

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet &>/dev/null

echo -e "${GREEN}  ✓ Permissions granted${NC}"

# Update .env.gcp template
echo ""
echo "Creating .env.gcp configuration file..."
cat > .env.gcp <<EOF
# GCP Configuration for AI Agency Platform
# This file is used for Cloud Run deployment

# GCP Project
GCP_PROJECT_ID=$PROJECT_ID
GCS_BUCKET=$BUCKET_NAME

# Database (update when Cloud SQL is created)
DATABASE_URL=postgresql+psycopg://user:pass@/dbname?host=/cloudsql/PROJECT:REGION:INSTANCE

# LLM Provider (openai or vertex)
LLM_PROVIDER=openai

# OpenAI (stored in Secret Manager)
# OPENAI_API_KEY will be injected from Secret Manager

# Vertex AI
VERTEX_AI_LOCATION=$REGION

# Application
LOG_LEVEL=INFO
ENVIRONMENT=production

# Artifact Registry
ARTIFACT_REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}
EOF
echo -e "${GREEN}✓ .env.gcp created${NC}"

# Create deployment configuration
echo ""
echo "Creating Cloud Run deployment configuration..."
cat > clouddeploy.yaml <<EOF
# Cloud Run Deployment Configuration
# Deploy with: gcloud run services replace clouddeploy.yaml

apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ai-agency
  labels:
    cloud.googleapis.com/location: $REGION
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: '0'
        autoscaling.knative.dev/maxScale: '10'
        run.googleapis.com/startup-cpu-boost: 'true'
    spec:
      serviceAccountName: $SERVICE_ACCOUNT_EMAIL
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/app:latest
        ports:
        - name: http1
          containerPort: 8080
        env:
        - name: GCP_PROJECT_ID
          value: "$PROJECT_ID"
        - name: GCS_BUCKET
          value: "$BUCKET_NAME"
        - name: LLM_PROVIDER
          value: "openai"
        - name: VERTEX_AI_LOCATION
          value: "$REGION"
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENVIRONMENT
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-api-key
              key: latest
        resources:
          limits:
            cpu: '2'
            memory: 2Gi
EOF
echo -e "${GREEN}✓ clouddeploy.yaml created${NC}"

# Summary
echo ""
echo "========================================="
echo "GCP Setup Complete!"
echo "========================================="
echo ""
echo "✓ APIs enabled"
echo "✓ Artifact Registry ready: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}"
echo "✓ GCS bucket ready: gs://${BUCKET_NAME}"
echo "✓ Secret Manager configured"
echo "✓ Service account created with permissions"
echo "✓ Configuration files created"
echo ""
echo "Next steps:"
echo "  1. Build and push container:"
echo "     docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/app:latest ."
echo "     docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/app:latest"
echo ""
echo "  2. Deploy to Cloud Run:"
echo "     gcloud run services replace clouddeploy.yaml --region=$REGION"
echo ""
echo "  3. For other developers:"
echo "     - Share .env.example (they add their own OPENAI_API_KEY)"
echo "     - They run: gcloud auth login"
echo "     - They run: gcloud auth configure-docker ${REGION}-docker.pkg.dev"
echo ""
echo "Environment configured for: merlin-notebook-lm (shared R&D)"
echo ""
