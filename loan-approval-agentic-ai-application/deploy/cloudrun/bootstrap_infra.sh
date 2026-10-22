#!/usr/bin/env bash
# One-time GCP infrastructure for LoanFlow (idempotent-ish; review before running).
#   PROJECT_ID=my-bank-proj REGION=asia-south1 ./deploy/cloudrun/bootstrap_infra.sh
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-south1}"          # Mumbai - keeps Indian customer data in-country (RBI data localisation)
NET="${NET:-default}"
gcloud config set project "$PROJECT_ID"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com sqladmin.googleapis.com redis.googleapis.com vpcaccess.googleapis.com \
  pubsub.googleapis.com cloudtrace.googleapis.com monitoring.googleapis.com

# --- Artifact Registry
gcloud artifacts repositories create loanflow --repository-format=docker --location="$REGION" || true

# --- Serverless VPC connector (Cloud Run -> Memorystore / Cloud SQL private IP)
gcloud compute networks vpc-access connectors create loanflow-conn --region="$REGION" --network="$NET" \
  --range=10.8.0.0/28 --min-instances=2 --max-instances=10 || true

# --- Memorystore for Redis (short-term memory, semantic cache, rate limits, job streams)
gcloud redis instances create loanflow-redis --size=1 --region="$REGION" --tier=STANDARD_HA \
  --redis-version=redis_7_2 --network="$NET" --transit-encryption-mode=SERVER_AUTHENTICATION || true

# --- Cloud SQL Postgres 16 (pgvector is a supported extension) - long-term memory + LangGraph checkpoints
gcloud sql instances create loanflow-pg --database-version=POSTGRES_16 --region="$REGION" --tier=db-custom-2-7680 \
  --availability-type=REGIONAL --network="projects/$PROJECT_ID/global/networks/$NET" --no-assign-ip \
  --backup --enable-point-in-time-recovery || true
gcloud sql databases create loanflow --instance=loanflow-pg || true
echo ">> connect once and run db/init.sql (CREATE EXTENSION vector; tables)"

# --- Service accounts: one per service for least-privilege invoker grants
gcloud iam service-accounts create sa-supervisor --display-name="LoanFlow supervisor" || true
for a in fetch-document validate-document check-credit assess-risk check-compliance compile-report compile-rejection; do
  gcloud iam service-accounts create "sa-agent-$a" --display-name="LoanFlow agent $a" || true
done
gcloud iam service-accounts create sa-mcp --display-name="LoanFlow MCP servers" || true

# --- Secrets (values from your vault; never commit them)
for s in JWT_SECRET MESSAGE_HMAC_KEY INTERNAL_SERVICE_TOKEN GROQ_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY POSTGRES_DSN REDIS_URL; do
  gcloud secrets create "$s" --replication-policy=user-managed --locations="$REGION" || true
done
echo ">> add versions: printf '%s' \"\$VALUE\" | gcloud secrets versions add JWT_SECRET --data-file=-"

# --- Pub/Sub (optional alternative to Redis Streams for the job queue, push to /internal/jobs)
gcloud pubsub topics create loan-jobs || true
gcloud pubsub topics create loan-jobs-dlq || true
echo "bootstrap done"
