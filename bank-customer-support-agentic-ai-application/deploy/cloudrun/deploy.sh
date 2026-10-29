#!/usr/bin/env bash
# Deploy the platform to Google Cloud Run as 18 microservices:
#   supervisor (public, behind HTTPS LB + Cloud Armor) | 8 sub-agents | 8 MCP servers | memory-worker
# Each sub-agent and each MCP server is its OWN Cloud Run service with its OWN service account.
# IAM enforces state isolation: agent-X's identity can invoke ONLY mcp-X; only the supervisor can
# invoke agents. Everything except the supervisor uses ingress=internal.
#
# Prereqs: gcloud auth, APIs enabled (run, artifactregistry, secretmanager, sqladmin, redis, pubsub,
# vpcaccess, cloudbuild), Cloud SQL Postgres 16 with pgvector, Memorystore Redis, a Serverless VPC
# connector, and secrets created (see README "Deploying to Cloud Run").
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-asia-south1}"                    # Mumbai; data residency for Indian customers
REPO="${REPO:-bank-agentic}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bank-agentic:${TAG}"
VPC_CONNECTOR="${VPC_CONNECTOR:-bank-connector}"
REDIS_URL="${REDIS_URL:?set REDIS_URL e.g. redis://10.0.0.3:6379/0}"
CLOUDSQL_INSTANCE="${CLOUDSQL_INSTANCE:?set PROJECT:REGION:INSTANCE}"
AGENTS=(account payments cards fraud loans investments knowledge compliance)

echo ">> building ${IMAGE}"
gcloud builds submit --project "$PROJECT" --tag "$IMAGE" --gcs-log-dir "gs://${PROJECT}_cloudbuild/logs" \
  --config <(printf 'steps:\n- name: gcr.io/cloud-builders/docker\n  args: [build, -f, docker/Dockerfile, -t, %s, .]\nimages: [%s]\n' "$IMAGE" "$IMAGE") .

sa() { echo "$1@${PROJECT}.iam.gserviceaccount.com"; }
ensure_sa() {
  gcloud iam service-accounts describe "$(sa "$1")" --project "$PROJECT" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$1" --project "$PROJECT" --display-name "bank-agentic $1"
}

COMMON_FLAGS=(--project "$PROJECT" --region "$REGION" --image "$IMAGE" --platform managed
  --vpc-connector "$VPC_CONNECTOR" --vpc-egress private-ranges-only
  --add-cloudsql-instances "$CLOUDSQL_INSTANCE" --execution-environment gen2 --cpu-boost
  --set-secrets "JWT_SECRET=jwt-secret:latest,HANDOFF_HMAC_SECRET=handoff-secret:latest,DELEGATION_SECRET=delegation-secret:latest,APPROVAL_SECRET=approval-secret:latest,POSTGRES_DSN=postgres-dsn:latest,GROQ_API_KEY=groq-api-key:latest,LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest,LANGFUSE_SECRET_KEY=langfuse-secret-key:latest"
  --labels "app=bank-agentic,tag=${TAG}")
BASE_ENV="ENVIRONMENT=production,SERVICE_AUTH_MODE=gcp,REDIS_URL=${REDIS_URL},EVENT_BUS_BACKEND=pubsub,GCP_PROJECT=${PROJECT},LLM_PROVIDER=${LLM_PROVIDER:-vllm},LLM_FALLBACK_PROVIDERS=[\"groq\"],VLLM_BASE_URL=${VLLM_BASE_URL:-http://10.0.0.10:8000/v1},EMBEDDING_BACKEND=${EMBEDDING_BACKEND:-hash},LANGFUSE_HOST=${LANGFUSE_HOST:-https://langfuse.internal.example}"

url_of() { gcloud run services describe "$1" --project "$PROJECT" --region "$REGION" --format 'value(status.url)'; }

# ------------------------------------------------------------------ MCP servers
for a in "${AGENTS[@]}"; do
  ensure_sa "mcp-$a"
  gcloud run deploy "mcp-$a" "${COMMON_FLAGS[@]}" --service-account "$(sa "mcp-$a")" \
    --no-allow-unauthenticated --ingress internal \
    --concurrency 80 --min-instances 1 --max-instances 50 --cpu 1 --memory 512Mi --timeout 30 \
    --set-env-vars "${BASE_ENV},SERVICE=mcp,MCP_DOMAIN=${a}"
  MCP_URL="$(url_of "mcp-$a")"
  gcloud run services update "mcp-$a" --project "$PROJECT" --region "$REGION" --update-env-vars "SELF_URL=${MCP_URL}"
done

# --------------------------------------------------------------- sub-agents
AGENT_URLS="{"
for a in "${AGENTS[@]}"; do
  ensure_sa "agent-$a"
  MCP_URL="$(url_of "mcp-$a")/mcp/"
  # IAM state isolation: ONLY agent-$a may invoke mcp-$a
  gcloud run services add-iam-policy-binding "mcp-$a" --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:$(sa "agent-$a")" --role roles/run.invoker
  MAXI=100; [ "$a" = "fraud" ] && MAXI=200                       # fraud gets headroom for incident spikes
  MINI=1; [ "$a" = "fraud" ] && MINI=2
  gcloud run deploy "agent-$a" "${COMMON_FLAGS[@]}" --service-account "$(sa "agent-$a")" \
    --no-allow-unauthenticated --ingress internal \
    --concurrency 40 --min-instances "$MINI" --max-instances "$MAXI" --cpu 1 --memory 1Gi --timeout 60 \
    --set-env-vars "^@^${BASE_ENV//,/@}@SERVICE=agent@AGENT_NAME=${a}@MCP_URLS={\"${a}\":\"${MCP_URL}\"}"
  URL="$(url_of "agent-$a")"
  gcloud run services update "agent-$a" --project "$PROJECT" --region "$REGION" --update-env-vars "SELF_URL=${URL}"
  AGENT_URLS+="\"${a}\":\"${URL}\","
done
AGENT_URLS="${AGENT_URLS%,}}"

# --------------------------------------------------------------- supervisor
ensure_sa supervisor
for a in "${AGENTS[@]}"; do
  gcloud run services add-iam-policy-binding "agent-$a" --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:$(sa supervisor)" --role roles/run.invoker
done
gcloud run deploy supervisor "${COMMON_FLAGS[@]}" --service-account "$(sa supervisor)" \
  --allow-unauthenticated --ingress internal-and-cloud-load-balancing \
  --concurrency 60 --min-instances 2 --max-instances 200 --cpu 2 --memory 2Gi --timeout 120 \
  --set-env-vars "^@^${BASE_ENV//,/@}@SERVICE=supervisor@AGENT_URLS=${AGENT_URLS}"

# ------------------------------------------------------------ memory worker
ensure_sa memory-worker
gcloud run deploy memory-worker "${COMMON_FLAGS[@]}" --service-account "$(sa memory-worker)" \
  --no-allow-unauthenticated --ingress internal --concurrency 20 --min-instances 0 --max-instances 20 \
  --set-env-vars "${BASE_ENV},SERVICE=memory-worker"
WORKER_URL="$(url_of memory-worker)"
bash "$(dirname "$0")/pubsub.sh" "$PROJECT" "$REGION" "$WORKER_URL"

echo ">> done. Supervisor: $(url_of supervisor)"
