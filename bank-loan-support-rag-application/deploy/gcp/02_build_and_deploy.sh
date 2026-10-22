#!/usr/bin/env bash
# Build all images with Cloud Build and deploy each microservice as its own Cloud Run service.
#
# Network/security model
#   * retriever, generator, guardrail: --ingress=internal + --no-allow-unauthenticated.
#     Only the gateway service account holds roles/run.invoker on them, and it calls them
#     with a Google-signed ID token (USE_GCP_ID_TOKEN=true) over Direct VPC egress.
#   * gateway: public (--allow-unauthenticated); put Cloud Armor / API Gateway in front for prod.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SA() { echo "rag-$1@${PROJECT_ID}.iam.gserviceaccount.com"; }

gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_REPO=${REPO},_TAG=${TAG}" .

COMMON_ENV="ENVIRONMENT=prod,LOG_LEVEL=INFO,CHUNK_STORE_URI=gs://${BUCKET}/state/chunks.jsonl,CDC_STATE_URI=gs://${BUCKET}/state/cdc_state.json,PDF_SOURCE_URI=gs://${BUCKET}/pdfs,MARKDOWN_OUT_DIR=gs://${BUCKET}/markdown,RETRIEVAL_LOG_PATH="
LF_SECRETS="LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest,LANGFUSE_SECRET_KEY=langfuse-secret-key:latest"
NET="--network=${VPC_NETWORK} --subnet=${VPC_SUBNET} --vpc-egress=all-traffic"

deploy() {  # name extra-args...
  local name=$1; shift
  gcloud run deploy "rag-${name}" --image="${IMAGE_BASE}/${name}:${TAG}" --region="$REGION" \
    --service-account="$(SA "$name")" --port=8080 --execution-environment=gen2 "$@"
}

# ---- internal services ------------------------------------------------------
deploy retriever --ingress=internal --no-allow-unauthenticated $NET \
  --cpu=2 --memory=2Gi --concurrency=40 --min-instances=1 --max-instances=5 --cpu-boost \
  --set-env-vars="${COMMON_ENV}" \
  --set-secrets="PINECONE_API_KEY=pinecone-api-key:latest,${LF_SECRETS}"

deploy generator --ingress=internal --no-allow-unauthenticated $NET \
  --cpu=1 --memory=512Mi --concurrency=80 --min-instances=0 --max-instances=3 --timeout=300 \
  --set-env-vars="${COMMON_ENV},LLM_PROVIDER=groq,LLM_FALLBACK_PROVIDER=gemini" \
  --set-secrets="GROQ_API_KEY=groq-api-key:latest,GOOGLE_API_KEY=google-api-key:latest,PROMPT_CANARY=prompt-canary:latest,${LF_SECRETS}"

deploy guardrail --ingress=internal --no-allow-unauthenticated $NET \
  --cpu=1 --memory=512Mi --concurrency=80 --min-instances=0 --max-instances=5 \
  --set-env-vars="${COMMON_ENV}" \
  --set-secrets="PROMPT_CANARY=prompt-canary:latest,${LF_SECRETS}"

URL() { gcloud run services describe "rag-$1" --region="$REGION" --format='value(status.url)'; }
RETRIEVER_URL=$(URL retriever); GENERATOR_URL=$(URL generator); GUARDRAIL_URL=$(URL guardrail)

for s in retriever generator guardrail; do
  gcloud run services add-iam-policy-binding "rag-${s}" --region="$REGION" \
    --member="serviceAccount:$(SA gateway)" --role=roles/run.invoker >/dev/null
done
# ingestion job/service may call retriever /admin/reload
gcloud run services add-iam-policy-binding rag-retriever --region="$REGION" \
  --member="serviceAccount:$(SA ingestion)" --role=roles/run.invoker >/dev/null

# ---- public gateway ------------------------------------------------------------
REDIS_HOST=$(gcloud redis instances describe "$REDIS_INSTANCE" --region="$REGION" --format='value(host)')
deploy gateway --allow-unauthenticated $NET \
  --cpu=1 --memory=512Mi --concurrency=100 --min-instances=1 --max-instances=10 --timeout=300 \
  --set-env-vars="${COMMON_ENV},RETRIEVER_URL=${RETRIEVER_URL},GENERATOR_URL=${GENERATOR_URL},GUARDRAIL_URL=${GUARDRAIL_URL},USE_GCP_ID_TOKEN=true,CACHE_BACKEND=redis,REDIS_URL=redis://${REDIS_HOST}:6379/0" \
  --set-secrets="PROMPT_CANARY=prompt-canary:latest,${LF_SECRETS}"

echo "Gateway: $(URL gateway)"
