#!/usr/bin/env bash
# Shared variables for the GCP deployment scripts. Edit, then `source deploy/gcp/env.sh`.
export PROJECT_ID="${PROJECT_ID:-your-gcp-project}"
export REGION="${REGION:-asia-south1}"              # Mumbai; Pinecone/Groq latency is fine from here
export REPO="${REPO:-bank-rag}"                     # Artifact Registry repo
export BUCKET="${BUCKET:-${PROJECT_ID}-bank-rag-kb}" # PDFs + chunk store + CDC state
export VPC_NETWORK="${VPC_NETWORK:-default}"
export VPC_SUBNET="${VPC_SUBNET:-default}"
export REDIS_INSTANCE="${REDIS_INSTANCE:-bank-rag-cache}"
export TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
export IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"
export SERVICES="retriever generator guardrail gateway"
