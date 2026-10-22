#!/usr/bin/env bash
# Deploy LoanFlow as 15 Cloud Run services (one per subagent, one per MCP server, supervisor API + worker).
#   PROJECT_ID=my-bank-proj REGION=asia-south1 TAG=$(git rev-parse --short HEAD) ./deploy/cloudrun/deploy.sh
#
# Security model
#   * Every service except the supervisor API: --no-allow-unauthenticated + --ingress internal
#   * Least-privilege invoker matrix: supervisor SA may invoke agents only; each agent SA may invoke
#     ONLY the MCP servers in its tool allow-list (e.g. check_credit -> credit_bureau).
#   * Services authenticate with Google-signed ID tokens (USE_GCP_ID_TOKENS=true) + internal token.
#   * Secrets come from Secret Manager, never env files.
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-asia-south1}"
TAG="${TAG:-latest}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/loanflow/loanflow:$TAG"
CONN="loanflow-conn"
SECRETS="JWT_SECRET=JWT_SECRET:latest,MESSAGE_HMAC_KEY=MESSAGE_HMAC_KEY:latest,INTERNAL_SERVICE_TOKEN=INTERNAL_SERVICE_TOKEN:latest,GROQ_API_KEY=GROQ_API_KEY:latest,LANGFUSE_PUBLIC_KEY=LANGFUSE_PUBLIC_KEY:latest,LANGFUSE_SECRET_KEY=LANGFUSE_SECRET_KEY:latest,POSTGRES_DSN=POSTGRES_DSN:latest,REDIS_URL=REDIS_URL:latest"
COMMON_ENV=(APP_ENV=prod TRANSPORT_MODE=http USE_GCP_ID_TOKENS=true LLM_CHAIN=groq,ollama,rules LOG_LEVEL=INFO)

MCP=(document credit_bureau risk_engine compliance reporting)
AGENTS=(fetch_document validate_document check_credit assess_risk check_compliance compile_report compile_rejection)
declare -A AGENT_MCP=(
  [fetch_document]=document [validate_document]=document [check_credit]=credit_bureau
  [assess_risk]=risk_engine [check_compliance]=compliance [compile_report]=reporting [compile_rejection]=reporting
)
dash() { echo "${1//_/-}"; }
sa() { echo "$1@$PROJECT_ID.iam.gserviceaccount.com"; }
url() { gcloud run services describe "$1" --region "$REGION" --format='value(status.url)'; }

envfile() {  # KEY=VALUE ... -> YAML env file (values may contain commas / JSON)
  local f; f=$(mktemp --suffix=.yaml)
  for kv in "$@"; do printf "%s: '%s'\n" "${kv%%=*}" "${kv#*=}" >> "$f"; done
  echo "$f"
}

deploy() {  # name, service-account, env-file, extra flags...
  local name=$1 svc_acct=$2 env_file=$3; shift 3
  gcloud run deploy "$name" --image "$IMAGE" --region "$REGION" --service-account "$svc_acct" \
    --vpc-connector "$CONN" --vpc-egress private-ranges-only --set-secrets "$SECRETS" \
    --env-vars-file "$env_file" --cpu 1 --memory 1Gi --timeout 300 --concurrency 40 \
    --execution-environment gen2 --cpu-boost "$@" --quiet
}

echo "== 1/4 MCP servers (internal only)"
for m in "${MCP[@]}"; do
  deploy "loanflow-mcp-$(dash "$m")" "$(sa sa-mcp)" "$(envfile "${COMMON_ENV[@]}" SERVICE_KIND=mcp MCP_SERVER="$m" SERVICE_NAME="mcp-$m")" \
    --no-allow-unauthenticated --ingress internal --min-instances 0 --max-instances 20
done
MCP_URLS="{$(for m in "${MCP[@]}"; do printf '"%s":"%s",' "$m" "$(url "loanflow-mcp-$(dash "$m")")"; done | sed 's/,$//')}"

echo "== 2/4 subagents (one Cloud Run service each)"
for a in "${AGENTS[@]}"; do
  deploy "loanflow-agent-$(dash "$a")" "$(sa "sa-agent-$(dash "$a")")" \
    "$(envfile "${COMMON_ENV[@]}" SERVICE_KIND=agent AGENT_NAME="$a" SERVICE_NAME="agent-$a" MCP_URLS="$MCP_URLS")" \
    --no-allow-unauthenticated --ingress internal --min-instances 0 --max-instances 50
  # least privilege: this agent may call only its MCP server
  gcloud run services add-iam-policy-binding "loanflow-mcp-$(dash "${AGENT_MCP[$a]}")" --region "$REGION" \
    --member "serviceAccount:$(sa "sa-agent-$(dash "$a")")" --role roles/run.invoker --quiet >/dev/null
done
AGENT_URLS="{$(for a in "${AGENTS[@]}"; do printf '"%s":"%s",' "$a" "$(url "loanflow-agent-$(dash "$a")")"; done | sed 's/,$//')}"
for a in "${AGENTS[@]}"; do
  gcloud run services add-iam-policy-binding "loanflow-agent-$(dash "$a")" --region "$REGION" \
    --member "serviceAccount:$(sa sa-supervisor)" --role roles/run.invoker --quiet >/dev/null
done

echo "== 3/4 supervisor worker (consumes Redis Stream; always-on CPU, no public ingress)"
deploy loanflow-supervisor-worker "$(sa sa-supervisor)" \
  "$(envfile "${COMMON_ENV[@]}" SERVICE_KIND=supervisor SERVICE_NAME=supervisor-worker WORKER_ENABLED=true AGENT_URLS="$AGENT_URLS")" \
  --no-allow-unauthenticated --ingress internal --no-cpu-throttling --min-instances 1 --max-instances 10 --memory 2Gi

echo "== 4/4 supervisor API (behind external HTTPS LB + Cloud Armor WAF/rate limiting)"
deploy loanflow-supervisor "$(sa sa-supervisor)" \
  "$(envfile "${COMMON_ENV[@]}" SERVICE_KIND=supervisor SERVICE_NAME=supervisor WORKER_ENABLED=false AGENT_URLS="$AGENT_URLS")" \
  --allow-unauthenticated --ingress internal-and-cloud-load-balancing --min-instances 1 --max-instances 20

echo "Deployed. Supervisor: $(url loanflow-supervisor)"
