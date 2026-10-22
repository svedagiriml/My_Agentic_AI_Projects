#!/usr/bin/env bash
# One-time project setup: APIs, Artifact Registry, bucket, service accounts, IAM, secrets, Redis.
set -euo pipefail
source "$(dirname "$0")/env.sh"
gcloud config set project "$PROJECT_ID"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com eventarc.googleapis.com redis.googleapis.com \
  cloudscheduler.googleapis.com monitoring.googleapis.com logging.googleapis.com pubsub.googleapis.com

gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" \
  --description="Bank RAG images" || true

gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access || true

# ---- least-privilege service accounts, one per service ---------------------
for sa in retriever generator guardrail gateway ingestion; do
  gcloud iam service-accounts create "rag-${sa}" --display-name="Bank RAG ${sa}" || true
done
SA() { echo "rag-$1@${PROJECT_ID}.iam.gserviceaccount.com"; }

# bucket access: ingestion read/write, retriever read-only
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:$(SA ingestion)" --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:$(SA retriever)" --role=roles/storage.objectViewer

# ---- secrets (values read from your local .env) ---------------------------
set -a; source .env; set +a
create_secret() {  # name value
  [ -z "$2" ] && { echo "skip secret $1 (empty)"; return; }
  printf '%s' "$2" | gcloud secrets create "$1" --data-file=- --replication-policy=automatic 2>/dev/null \
    || printf '%s' "$2" | gcloud secrets versions add "$1" --data-file=-
}
create_secret pinecone-api-key   "${PINECONE_API_KEY:-}"
create_secret groq-api-key       "${GROQ_API_KEY:-}"
create_secret google-api-key     "${GOOGLE_API_KEY:-}"
create_secret langfuse-public-key "${LANGFUSE_PUBLIC_KEY:-}"
create_secret langfuse-secret-key "${LANGFUSE_SECRET_KEY:-}"
create_secret prompt-canary      "${PROMPT_CANARY:-$(openssl rand -hex 16)}"

grant() { gcloud secrets add-iam-policy-binding "$1" --member="serviceAccount:$(SA "$2")" \
          --role=roles/secretmanager.secretAccessor >/dev/null 2>&1 || true; }
for s in pinecone-api-key langfuse-public-key langfuse-secret-key; do grant "$s" retriever; grant "$s" ingestion; done
for s in groq-api-key google-api-key langfuse-public-key langfuse-secret-key prompt-canary; do grant "$s" generator; done
for s in langfuse-public-key langfuse-secret-key prompt-canary; do grant "$s" guardrail; grant "$s" gateway; done

# ---- semantic cache: Memorystore for Redis 7.2 (supports vector search) ---------
# Smallest basic tier; private IP reachable from Cloud Run via Direct VPC egress.
gcloud redis instances create "$REDIS_INSTANCE" --region="$REGION" --tier=basic --size=1 \
  --redis-version=redis_7_2 --network="$VPC_NETWORK" || true
echo "Redis host: $(gcloud redis instances describe "$REDIS_INSTANCE" --region="$REGION" --format='value(host)')"
echo "Setup done."
