#!/usr/bin/env bash
#
# Deploy travel-buddy-backend to Cloud Run.
#
# WHY Cloud Run: the Gemini APIs (both the Developer API and Vertex AI) edge-block
# Render's datacenter egress IP with an HTML 403 — proven this is purely the
# source IP (same SA/project/model returns 200 from a residential IP, see
# memory project_gemini_migration). Cloud Run egress is inside Google's network,
# so it is not edge-blocked, AND ADC resolves natively from the runtime service
# account — no GOOGLE_SERVICE_ACCOUNT_JSON secret to manage.
#
# Run scripts/setup_cloudrun_iam_secrets.sh ONCE first (enables APIs, grants the
# runtime SA the roles it needs, and creates the Secret Manager secrets). Then
# run this for every deploy.
#
# Usage:  ./scripts/deploy_cloudrun.sh
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-travel-buddy-500309}"
REGION="${REGION:-europe-west2}"                       # London — closest to the Supabase DB (eu-west-2)
SERVICE="${SERVICE:-travel-buddy-backend}"
RUNTIME_SA="${RUNTIME_SA:-gemini-vertex-422@travel-buddy-500309.iam.gserviceaccount.com}"

# CORS origin(s) the web app is served from. Override before deploy.
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-https://travel-buddy-web-2h4g.onrender.com,https://travel-buddy-mobile.onrender.com,http://localhost:3000}"

cd "$(dirname "$0")/.."

# Non-secret config → plain env vars. Vertex mode + the project/location proven
# to work (travel-buddy-500309 / global). No GEMINI_API_KEY and no
# GOOGLE_SERVICE_ACCOUNT_JSON: the runtime SA supplies Vertex auth via ADC.
#
# Pairs are joined with '##' and the value is prefixed with '^##^' so gcloud uses
# '##' (not its default comma) as the pair delimiter — ALLOWED_ORIGINS itself
# contains commas (multiple origins) which would otherwise be misparsed.
ENV_VARS="GEMINI_USE_VERTEX=true"
ENV_VARS+="##GOOGLE_CLOUD_PROJECT=${PROJECT}"
ENV_VARS+="##GOOGLE_CLOUD_LOCATION=global"
ENV_VARS+="##GOOGLE_PLACES_ENABLED=true"
ENV_VARS+="##GOOGLE_DIRECTIONS_ENABLED=true"
ENV_VARS+="##LOG_FORMAT=json"
ENV_VARS+="##SENTRY_ENVIRONMENT=production"
ENV_VARS+="##ALLOWED_ORIGINS=${ALLOWED_ORIGINS}"

# Secrets → mounted from Secret Manager (created by setup script). KEY=secret:latest
SECRETS="SUPABASE_URL=SUPABASE_URL:latest"
SECRETS+=",SUPABASE_ANON_KEY=SUPABASE_ANON_KEY:latest"
SECRETS+=",SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest"
SECRETS+=",DATABASE_URL=DATABASE_URL:latest"
SECRETS+=",COHERE_API_KEY=COHERE_API_KEY:latest"
SECRETS+=",GOOGLE_API_KEY=GOOGLE_API_KEY:latest"

echo "Deploying ${SERVICE} → ${PROJECT}/${REGION} as ${RUNTIME_SA}"

gcloud run deploy "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --source . \
  --service-account "${RUNTIME_SA}" \
  --allow-unauthenticated \
  --set-env-vars "^##^${ENV_VARS}" \
  --set-secrets "${SECRETS}" \
  --cpu 1 \
  --memory 512Mi \
  --concurrency 40 \
  --timeout 600 \
  --min-instances 0 \
  --max-instances 4

URL=$(gcloud run services describe "${SERVICE}" --project "${PROJECT}" --region "${REGION}" --format='value(status.url)')
echo
echo "Deployed: ${URL}"
echo "Smoke test:"
echo "  curl -s ${URL}/health/ready | python3 -m json.tool"
echo
echo "Then point the web app at it:"
echo "  travel-buddy-web/.env.local → NEXT_PUBLIC_API_BASE_URL=${URL}"
echo "  (and add that origin to ALLOWED_ORIGINS above if it changes)"
