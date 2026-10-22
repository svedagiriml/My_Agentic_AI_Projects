#!/usr/bin/env bash
# Ingestion with CDC on GCP — two complementary triggers:
#   A) Cloud Run *Job* `rag-ingestion` (batch, incremental) + nightly Cloud Scheduler safety net
#   B) Cloud Run *service* `rag-ingestion-events` + Eventarc: any PDF finalized/deleted in the
#      bucket triggers an incremental CDC run within seconds.
set -euo pipefail
source "$(dirname "$0")/env.sh"
SA() { echo "rag-$1@${PROJECT_ID}.iam.gserviceaccount.com"; }
URL() { gcloud run services describe "rag-$1" --region="$REGION" --format='value(status.url)'; }

ENV="ENVIRONMENT=prod,CHUNK_STORE_URI=gs://${BUCKET}/state/chunks.jsonl,CDC_STATE_URI=gs://${BUCKET}/state/cdc_state.json,PDF_SOURCE_URI=gs://${BUCKET}/pdfs,MARKDOWN_OUT_DIR=gs://${BUCKET}/markdown,RETRIEVER_URL=$(URL retriever),USE_GCP_ID_TOKEN=true"
SECRETS="PINECONE_API_KEY=pinecone-api-key:latest"
NET="--network=${VPC_NETWORK} --subnet=${VPC_SUBNET} --vpc-egress=all-traffic"

# Upload the knowledge base
gcloud storage cp data/pdfs/*.pdf "gs://${BUCKET}/pdfs/"

# ---- A) batch job --------------------------------------------------------------
gcloud run jobs deploy rag-ingestion --image="${IMAGE_BASE}/ingestion:${TAG}" --region="$REGION" \
  --service-account="$(SA ingestion)" --cpu=2 --memory=4Gi --task-timeout=3600 --max-retries=1 \
  --parallelism=1 --tasks=1 $NET --set-env-vars="$ENV" --set-secrets="$SECRETS"
gcloud run jobs execute rag-ingestion --region="$REGION" --wait

gcloud iam service-accounts create rag-scheduler --display-name="Bank RAG scheduler" || true
gcloud run jobs add-iam-policy-binding rag-ingestion --region="$REGION" \
  --member="serviceAccount:rag-scheduler@${PROJECT_ID}.iam.gserviceaccount.com" --role=roles/run.invoker
gcloud scheduler jobs create http rag-ingestion-nightly --location="$REGION" --schedule="30 1 * * *" \
  --time-zone="Asia/Kolkata" --http-method=POST \
  --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/rag-ingestion:run" \
  --oauth-service-account-email="rag-scheduler@${PROJECT_ID}.iam.gserviceaccount.com" || true

# ---- B) event-driven service ---------------------------------------------------
gcloud run deploy rag-ingestion-events --image="${IMAGE_BASE}/ingestion:${TAG}" --region="$REGION" \
  --service-account="$(SA ingestion)" --cpu=2 --memory=4Gi --no-allow-unauthenticated $NET \
  --min-instances=0 --max-instances=1 --concurrency=1 --timeout=3600 --no-cpu-throttling \
  --command=uvicorn --args="services.ingestion.main:app,--host,0.0.0.0,--port,8080" \
  --set-env-vars="$ENV" --set-secrets="$SECRETS"

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
GCS_SA=$(gcloud storage service-agent --project="$PROJECT_ID")
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${GCS_SA}" --role=roles/pubsub.publisher >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$(SA ingestion)" --role=roles/eventarc.eventReceiver >/dev/null
gcloud run services add-iam-policy-binding rag-ingestion-events --region="$REGION" \
  --member="serviceAccount:$(SA ingestion)" --role=roles/run.invoker >/dev/null

for ev in finalized deleted; do
  gcloud eventarc triggers create "rag-pdf-${ev}" --location="$REGION" \
    --destination-run-service=rag-ingestion-events --destination-run-path=/events/gcs \
    --destination-run-region="$REGION" \
    --event-filters="type=google.cloud.storage.object.v1.${ev}" --event-filters="bucket=${BUCKET}" \
    --service-account="$(SA ingestion)" || true
done
echo "CDC triggers ready (project number ${PROJECT_NUMBER})."
