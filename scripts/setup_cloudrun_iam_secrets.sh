#!/usr/bin/env bash
#
# ONE-TIME Cloud Run bootstrap for travel-buddy-backend: enable APIs, grant the
# runtime service account the roles it needs, and create Secret Manager secrets
# from values in ../.env. Idempotent — safe to re-run (adds a new secret VERSION
# if a value changed). Run before the first scripts/deploy_cloudrun.sh.
#
# Usage:  ./scripts/setup_cloudrun_iam_secrets.sh
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-travel-buddy-500309}"
RUNTIME_SA="${RUNTIME_SA:-gemini-vertex-422@travel-buddy-500309.iam.gserviceaccount.com}"

cd "$(dirname "$0")/.."
ENV_FILE="${ENV_FILE:-.env}"
[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found (run from backend dir)"; exit 1; }

echo "== Enabling APIs on ${PROJECT} =="
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  --project "${PROJECT}"

echo "== Granting runtime SA roles =="
# Vertex AI access (the whole point). Idempotent if already granted.
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role "roles/aiplatform.user" --condition=None >/dev/null
# Read the secrets mounted at deploy time.
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role "roles/secretmanager.secretAccessor" --condition=None >/dev/null

# Let the human/CI deployer impersonate the runtime SA (required by
# `gcloud run deploy --service-account`). Uses the active gcloud account.
DEPLOYER="$(gcloud config get-value account 2>/dev/null)"
if [ -n "${DEPLOYER}" ]; then
  echo "== Granting ${DEPLOYER} actAs on ${RUNTIME_SA} =="
  gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" \
    --project "${PROJECT}" \
    --member "user:${DEPLOYER}" \
    --role "roles/iam.serviceAccountUser" >/dev/null
fi

# Read one KEY from .env, stripping surrounding quotes/whitespace.
read_env() { grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed -e 's/^["'"'"']//' -e 's/["'"'"']$//'; }

# Create-or-update a secret from a value (new version each run; no-op if empty).
put_secret() {
  local name="$1" value="$2"
  [ -z "$value" ] && { echo "  skip ${name} (empty in ${ENV_FILE})"; return; }
  if gcloud secrets describe "$name" --project "${PROJECT}" >/dev/null 2>&1; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --project "${PROJECT}" --data-file=- >/dev/null
    echo "  updated ${name}"
  else
    printf '%s' "$value" | gcloud secrets create "$name" --project "${PROJECT}" --replication-policy=automatic --data-file=- >/dev/null
    echo "  created ${name}"
  fi
}

echo "== Creating/updating secrets from ${ENV_FILE} =="
put_secret SUPABASE_URL          "$(read_env SUPABASE_URL)"
put_secret SUPABASE_ANON_KEY     "$(read_env SUPABASE_ANON_KEY)"
put_secret SUPABASE_SERVICE_KEY  "$(read_env SUPABASE_SERVICE_KEY)"
put_secret DATABASE_URL          "$(read_env DATABASE_URL)"
put_secret COHERE_API_KEY        "$(read_env COHERE_API_KEY)"
put_secret GOOGLE_API_KEY        "$(read_env GOOGLE_API_KEY)"

echo
echo "Done. Now deploy:  ./scripts/deploy_cloudrun.sh"
