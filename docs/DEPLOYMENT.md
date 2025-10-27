# Deployment Guide

This guide covers deployment of the AI Agency Platform to various cloud providers and local development environments.

## Table of Contents

1. [Database Portability](#database-portability)
2. [Local Development](#local-development)
3. [Google Cloud Platform (GCP)](#google-cloud-platform-gcp)
4. [Amazon Web Services (AWS)](#amazon-web-services-aws)
5. [Microsoft Azure](#microsoft-azure)
6. [Database Migrations](#database-migrations)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Troubleshooting](#troubleshooting)

---

## Database Portability

The AI Agency Platform is designed for **easy database portability** across different PostgreSQL providers. Switching between providers requires only updating the `DATABASE_URL` environment variable.

### Key Features

- **Standard PostgreSQL**: Uses only standard PostgreSQL features and pgvector extension
- **Single Configuration**: Change `DATABASE_URL` to switch providers
- **Consistent Schema**: Same migrations work across all providers
- **Vector Support**: pgvector extension required (1536 dimensions for OpenAI embeddings)

### Connection String Format

All providers use the standard PostgreSQL connection string format:

```
postgresql://username:password@host:port/database?options
```

### Provider-Specific Connection Strings

#### Local PostgreSQL
```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_agency
```

#### GCP Cloud SQL
```bash
# Public IP
DATABASE_URL=postgresql://user:password@34.123.45.67:5432/ai_agency

# Unix socket (with Cloud SQL Proxy)
DATABASE_URL=postgresql://user:password@/cloudsql/project-id:region:instance-name/ai_agency

# Private IP (from VPC)
DATABASE_URL=postgresql://user:password@10.0.0.3:5432/ai_agency
```

#### AWS RDS
```bash
# Public endpoint
DATABASE_URL=postgresql://user:password@mydb.abc123.us-east-1.rds.amazonaws.com:5432/ai_agency

# With SSL
DATABASE_URL=postgresql://user:password@mydb.abc123.us-east-1.rds.amazonaws.com:5432/ai_agency?sslmode=require

# Read replica
DATABASE_URL=postgresql://user:password@mydb-ro.abc123.us-east-1.rds.amazonaws.com:5432/ai_agency
```

#### Azure Database for PostgreSQL
```bash
# Single Server
DATABASE_URL=postgresql://user@servername:password@servername.postgres.database.azure.com:5432/ai_agency?sslmode=require

# Flexible Server
DATABASE_URL=postgresql://user:password@servername.postgres.database.azure.com:5432/ai_agency?sslmode=require
```

### Testing Database Portability

```bash
# Test connection
psql "$DATABASE_URL" -c "SELECT version();"

# Test pgvector
psql "$DATABASE_URL" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Run migrations
export DATABASE_URL="your-connection-string"
./scripts/run_migrations.sh upgrade head
```

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL client tools (optional, for debugging)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ConsultingAgency
   ```

2. **Run automated setup**
   ```bash
   ./scripts/setup_dev.sh
   ```

   This script will:
   - Check Python version
   - Create/activate virtual environment
   - Install dependencies
   - Start PostgreSQL with pgvector
   - Run database migrations
   - Install pre-commit hooks

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start the application**

   Option A: Docker Compose (full stack)
   ```bash
   docker-compose up
   ```

   Option B: Local Python (for debugging)
   ```bash
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

5. **Access the application**
   - API: http://localhost:8080
   - Health check: http://localhost:8080/health
   - API docs: http://localhost:8080/docs

### Manual Setup (if needed)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e .
pip install -e ".[dev]"

# 3. Start PostgreSQL
docker-compose up -d db

# 4. Run migrations
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_agency
alembic upgrade head

# 5. Install pre-commit hooks
pre-commit install
```

### Development Workflow

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Linting and formatting
ruff check app tests
ruff format app tests

# Type checking
mypy app

# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

---

## Google Cloud Platform (GCP)

### Prerequisites

- GCP Project with billing enabled
- `gcloud` CLI installed and configured
- Required APIs enabled:
  - Cloud Run API
  - Cloud SQL Admin API
  - Artifact Registry API
  - Secret Manager API

### 1. Create Cloud SQL Instance

```bash
# Set variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export INSTANCE_NAME="ai-agency-db"

# Create PostgreSQL instance with pgvector
gcloud sql instances create $INSTANCE_NAME \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=$REGION \
  --database-flags=cloudsql.enable_pgvector=on

# Create database
gcloud sql databases create ai_agency \
  --instance=$INSTANCE_NAME

# Create user
gcloud sql users create ai_agency_user \
  --instance=$INSTANCE_NAME \
  --password=YOUR_SECURE_PASSWORD

# Enable pgvector extension
gcloud sql connect $INSTANCE_NAME --user=postgres
# In psql:
# \c ai_agency
# CREATE EXTENSION IF NOT EXISTS vector;
# \q
```

### 2. Configure Cloud SQL Proxy (for migrations)

```bash
# Download Cloud SQL Proxy
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# Start proxy
./cloud-sql-proxy $PROJECT_ID:$REGION:$INSTANCE_NAME --port 5432 &

# Run migrations
export DATABASE_URL=postgresql://ai_agency_user:password@localhost:5432/ai_agency
alembic upgrade head
```

### 3. Configure Secrets

```bash
# Store OpenAI API key
echo -n "sk-your-key" | gcloud secrets create openai-api-key \
  --data-file=- \
  --replication-policy=automatic

# Store database password
echo -n "password" | gcloud secrets create database-password \
  --data-file=- \
  --replication-policy=automatic
```

### 4. Build and Push Docker Image

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build image
docker build -t gcr.io/$PROJECT_ID/ai-agency:latest .

# Push image
docker push gcr.io/$PROJECT_ID/ai-agency:latest
```

### 5. Deploy to Cloud Run

```bash
# Get Cloud SQL connection name
INSTANCE_CONNECTION=$(gcloud sql instances describe $INSTANCE_NAME --format='value(connectionName)')

# Deploy
gcloud run deploy ai-agency \
  --image gcr.io/$PROJECT_ID/ai-agency:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-cloudsql-instances=$INSTANCE_CONNECTION \
  --set-env-vars DATABASE_URL=postgresql://ai_agency_user:password@/cloudsql/$INSTANCE_CONNECTION/ai_agency \
  --set-env-vars GCP_PROJECT_ID=$PROJECT_ID \
  --set-env-vars GCS_BUCKET=your-bucket-name \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest \
  --min-instances 1 \
  --max-instances 10 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

### 6. Set up CI/CD

Configure GitHub secrets:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_SA_KEY`: Service account JSON key
- `DATABASE_URL`: Cloud SQL connection string
- `OPENAI_API_KEY`: OpenAI API key

Enable deployment in `.github/workflows/deploy.yml` by setting `if: true`.

---

## Amazon Web Services (AWS)

### Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured
- Docker installed

### 1. Create RDS PostgreSQL Instance

```bash
# Set variables
export DB_INSTANCE="ai-agency-db"
export DB_NAME="ai_agency"
export DB_USER="ai_agency_user"
export DB_PASSWORD="YOUR_SECURE_PASSWORD"
export REGION="us-east-1"

# Create DB subnet group (if not exists)
aws rds create-db-subnet-group \
  --db-subnet-group-name ai-agency-subnet \
  --db-subnet-group-description "AI Agency DB Subnet" \
  --subnet-ids subnet-xxxxx subnet-yyyyy

# Create security group
aws ec2 create-security-group \
  --group-name ai-agency-db-sg \
  --description "AI Agency Database Security Group" \
  --vpc-id vpc-xxxxx

# Allow PostgreSQL access
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 5432 \
  --cidr 0.0.0.0/0

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier $DB_INSTANCE \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.4 \
  --master-username $DB_USER \
  --master-user-password $DB_PASSWORD \
  --allocated-storage 20 \
  --db-name $DB_NAME \
  --vpc-security-group-ids sg-xxxxx \
  --db-subnet-group-name ai-agency-subnet \
  --backup-retention-period 7 \
  --publicly-accessible

# Wait for instance to be available
aws rds wait db-instance-available --db-instance-identifier $DB_INSTANCE

# Get endpoint
DB_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier $DB_INSTANCE \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "Database endpoint: $DB_ENDPOINT"
```

### 2. Install pgvector Extension

```bash
# Connect to RDS
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_ENDPOINT:5432/$DB_NAME?sslmode=require"

# In psql:
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

### 3. Run Migrations

```bash
export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_ENDPOINT:5432/$DB_NAME?sslmode=require"
alembic upgrade head
```

### 4. Deploy to AWS App Runner / ECS

#### Option A: AWS App Runner

```bash
# Build and push to ECR
aws ecr create-repository --repository-name ai-agency

# Get ECR URI
ECR_URI=$(aws ecr describe-repositories --repository-names ai-agency --query 'repositories[0].repositoryUri' --output text)

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Build and push
docker build -t $ECR_URI:latest .
docker push $ECR_URI:latest

# Create App Runner service
aws apprunner create-service \
  --service-name ai-agency \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$ECR_URI':latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": {
          "DATABASE_URL": "postgresql://'$DB_USER':'$DB_PASSWORD'@'$DB_ENDPOINT':5432/'$DB_NAME'?sslmode=require",
          "OPENAI_API_KEY": "your-key"
        }
      }
    },
    "AutoDeploymentsEnabled": true
  }' \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }'
```

### 5. Configure Environment Variables

Store sensitive data in AWS Secrets Manager:

```bash
# Store OpenAI API key
aws secretsmanager create-secret \
  --name ai-agency/openai-key \
  --secret-string "sk-your-key"

# Store database password
aws secretsmanager create-secret \
  --name ai-agency/db-password \
  --secret-string "$DB_PASSWORD"
```

---

## Microsoft Azure

### Prerequisites

- Azure account with active subscription
- Azure CLI installed and configured
- Docker installed

### 1. Create Azure Database for PostgreSQL

```bash
# Set variables
export RESOURCE_GROUP="ai-agency-rg"
export LOCATION="eastus"
export SERVER_NAME="ai-agency-db"
export DB_NAME="ai_agency"
export ADMIN_USER="ai_agency_admin"
export ADMIN_PASSWORD="YOUR_SECURE_PASSWORD"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create PostgreSQL Flexible Server
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $SERVER_NAME \
  --location $LOCATION \
  --admin-user $ADMIN_USER \
  --admin-password "$ADMIN_PASSWORD" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --version 15 \
  --storage-size 32 \
  --public-access 0.0.0.0-255.255.255.255

# Create database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $SERVER_NAME \
  --database-name $DB_NAME

# Get connection string
SERVER_FQDN=$(az postgres flexible-server show \
  --resource-group $RESOURCE_GROUP \
  --name $SERVER_NAME \
  --query fullyQualifiedDomainName \
  --output tsv)

echo "Connection string: postgresql://$ADMIN_USER:$ADMIN_PASSWORD@$SERVER_FQDN:5432/$DB_NAME?sslmode=require"
```

### 2. Install pgvector Extension

```bash
# Enable extension
az postgres flexible-server parameter set \
  --resource-group $RESOURCE_GROUP \
  --server-name $SERVER_NAME \
  --name azure.extensions \
  --value vector

# Connect and create extension
psql "postgresql://$ADMIN_USER:$ADMIN_PASSWORD@$SERVER_FQDN:5432/$DB_NAME?sslmode=require" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Run Migrations

```bash
export DATABASE_URL="postgresql://$ADMIN_USER:$ADMIN_PASSWORD@$SERVER_FQDN:5432/$DB_NAME?sslmode=require"
alembic upgrade head
```

### 4. Deploy to Azure Container Instances / App Service

```bash
# Create container registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name aiagencyacr \
  --sku Basic

# Login to ACR
az acr login --name aiagencyacr

# Build and push image
ACR_LOGIN_SERVER=$(az acr show --name aiagencyacr --query loginServer --output tsv)
docker build -t $ACR_LOGIN_SERVER/ai-agency:latest .
docker push $ACR_LOGIN_SERVER/ai-agency:latest

# Create container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name ai-agency \
  --image $ACR_LOGIN_SERVER/ai-agency:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server $ACR_LOGIN_SERVER \
  --registry-username $(az acr credential show --name aiagencyacr --query username --output tsv) \
  --registry-password $(az acr credential show --name aiagencyacr --query passwords[0].value --output tsv) \
  --dns-name-label ai-agency \
  --ports 8080 \
  --environment-variables \
    DATABASE_URL="postgresql://$ADMIN_USER:$ADMIN_PASSWORD@$SERVER_FQDN:5432/$DB_NAME?sslmode=require" \
    OPENAI_API_KEY="your-key"
```

---

## Database Migrations

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add new table"

# Create empty migration (for manual SQL)
alembic revision -m "Custom migration"

# Edit the generated file in app/db/migrations/versions/
```

### Applying Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade abc123

# Upgrade one version forward
alembic upgrade +1
```

### Rolling Back Migrations

```bash
# Downgrade one version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade abc123

# Downgrade to base (caution!)
alembic downgrade base
```

### Migration Best Practices

1. **Always review auto-generated migrations** before applying
2. **Test migrations on a copy** of production data first
3. **Make migrations reversible** with proper downgrade logic
4. **Use transactions** for data migrations
5. **Backup before migrating** production databases

### Helper Script

Use the migrations helper script:

```bash
# Upgrade to latest
./scripts/run_migrations.sh upgrade head

# Create new migration
./scripts/run_migrations.sh revision "Add user table"

# Show current version
./scripts/run_migrations.sh current

# Show migration history
./scripts/run_migrations.sh history
```

---

## CI/CD Pipeline

### GitHub Actions Workflows

The repository includes two workflows:

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Runs on every PR and push to main/develop
   - Tests against real PostgreSQL with pgvector
   - Runs linting (Ruff), formatting, and type checking
   - Requires 70% code coverage

2. **Deployment Pipeline** (`.github/workflows/deploy.yml`)
   - Currently disabled (manual trigger only)
   - Deploys to Google Cloud Run
   - Runs migrations before deployment

### Setting Up CI/CD

1. **Configure GitHub Secrets**

   Go to repository Settings > Secrets and add:

   ```
   GCP_PROJECT_ID      # Your GCP project ID
   GCP_SA_KEY          # Service account JSON key
   DATABASE_URL        # Cloud SQL connection string
   OPENAI_API_KEY      # OpenAI API key
   GCS_BUCKET          # GCS bucket name
   CODECOV_TOKEN       # Codecov token (optional)
   ```

2. **Enable Deployment**

   Edit `.github/workflows/deploy.yml`:
   ```yaml
   # Change this line:
   if: false
   # To:
   if: true
   ```

3. **Trigger Deployment**

   Push to main branch or manually trigger via GitHub Actions UI.

### Local CI Testing

Test the CI pipeline locally:

```bash
# Start PostgreSQL
docker-compose up -d db

# Run migrations
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_agency
alembic upgrade head

# Run all checks
ruff check app tests
ruff format --check app tests
mypy app
pytest --cov=app --cov-fail-under=70
```

---

## Troubleshooting

### Common Issues

#### 1. pgvector Extension Not Found

**Problem**: `ERROR: extension "vector" is not available`

**Solutions**:
- GCP: Enable `cloudsql.enable_pgvector=on` flag
- AWS: Ensure PostgreSQL 15+ and install from RDS extensions
- Azure: Add `vector` to `azure.extensions` parameter
- Local: Use `ankane/pgvector` Docker image

```bash
# Check if extension is available
psql "$DATABASE_URL" -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

# Install extension
psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### 2. Database Connection Failed

**Problem**: `FATAL: password authentication failed`

**Solutions**:
- Verify DATABASE_URL format
- Check firewall rules / security groups
- Verify SSL mode (use `sslmode=require` for cloud providers)
- Check Cloud SQL Proxy is running (GCP)

```bash
# Test connection
psql "$DATABASE_URL" -c "SELECT 1;"

# Test with verbose output
psql "$DATABASE_URL?sslmode=require" -c "\conninfo"
```

#### 3. Migration Conflicts

**Problem**: `alembic.util.exc.CommandError: Multiple head revisions`

**Solutions**:
```bash
# Check heads
alembic heads

# Merge heads
alembic merge heads -m "Merge branches"

# Apply merge
alembic upgrade head
```

#### 4. Cloud Run Deployment Timeout

**Problem**: Service deployment times out or fails health checks

**Solutions**:
- Increase `timeout` in `gcloud run deploy`
- Check Cloud SQL connection (use Unix socket)
- Verify environment variables are set
- Check Cloud Run logs: `gcloud run services logs read ai-agency --limit=50`

#### 5. Docker Build Fails

**Problem**: `ERROR: Could not install packages due to an EnvironmentError`

**Solutions**:
- Clear Docker cache: `docker builder prune`
- Build without cache: `docker build --no-cache`
- Check `.dockerignore` is not excluding required files

#### 6. Pre-commit Hooks Fail

**Problem**: `ruff: command not found`

**Solutions**:
```bash
# Reinstall pre-commit
pip install pre-commit
pre-commit install

# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files
```

### Getting Help

- Check application logs: `docker-compose logs app`
- Check database logs: `docker-compose logs db`
- Run migrations with verbose output: `alembic -x verbose=true upgrade head`
- Enable debug logging: Set `LOG_LEVEL=DEBUG` in environment

### Performance Optimization

#### Database Query Optimization

```sql
-- Check slow queries
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - query_start > interval '1 second';

-- Analyze vector search performance
EXPLAIN ANALYZE
SELECT * FROM document_chunks
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

#### Index Tuning

```sql
-- Rebuild vector index for better performance (after data load)
DROP INDEX IF EXISTS ix_document_chunks_embedding;
CREATE INDEX ix_document_chunks_embedding
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Analyze table
ANALYZE document_chunks;
```

---

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [AWS RDS PostgreSQL Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_PostgreSQL.html)
- [Azure Database for PostgreSQL Documentation](https://docs.microsoft.com/en-us/azure/postgresql/)

---

**Last Updated**: 2025-10-27
