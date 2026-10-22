#!/usr/bin/env bash
# Async messaging: Pub/Sub topics + OIDC push subscriptions to the memory worker, with retry
# back-off and dead-letter topics (fault tolerance for off-path work).
set -euo pipefail
PROJECT="$1"; REGION="$2"; WORKER_URL="$3"
PUSH_SA="pubsub-push@${PROJECT}.iam.gserviceaccount.com"
gcloud iam service-accounts describe "$PUSH_SA" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud iam service-accounts create pubsub-push --project "$PROJECT"
gcloud run services add-iam-policy-binding memory-worker --project "$PROJECT" --region "$REGION" \
  --member "serviceAccount:${PUSH_SA}" --role roles/run.invoker
for t in memory-consolidate audit-events hitl-requested hitl-decided; do
  gcloud pubsub topics describe "$t" --project "$PROJECT" >/dev/null 2>&1 || gcloud pubsub topics create "$t" --project "$PROJECT"
  gcloud pubsub topics describe "$t-dlq" --project "$PROJECT" >/dev/null 2>&1 || gcloud pubsub topics create "$t-dlq" --project "$PROJECT"
done
for t in memory-consolidate audit-events; do
  gcloud pubsub subscriptions describe "$t-push" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud pubsub subscriptions create "$t-push" --project "$PROJECT" --topic "$t" \
    --push-endpoint "${WORKER_URL}/pubsub/push" --push-auth-service-account "$PUSH_SA" \
    --ack-deadline 60 --min-retry-delay 10s --max-retry-delay 600s \
    --dead-letter-topic "$t-dlq" --max-delivery-attempts 10 --message-retention-duration 7d
done
