# Deploying the backend to Cloud Run

## Why we moved off Render

Both Gemini surfaces — the Developer API (`generativelanguage.googleapis.com`)
**and** Vertex AI (`aiplatform.googleapis.com`) — edge-block Render's datacenter
egress IP with a generic HTML `403 Forbidden`. This was proven to be purely the
source IP: the identical service account / project / model returns `200` from a
residential IP but `403` from Render (see `memory/project_gemini_migration.md`).
The block is at Google's front-end edge, *before* auth, so no key, billing,
IAM, or surface change fixes it.

Cloud Run egress originates **inside Google's network**, so it is not
edge-blocked — and Application Default Credentials resolve natively from the
service the container runs as, so there's **no service-account JSON secret to
manage**.

## Prerequisites (once)

1. Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
2. Authenticate as a user with Owner/Editor on `travel-buddy-500309`:
   ```
   gcloud auth login
   gcloud config set project travel-buddy-500309
   ```

## One-time setup

Enables APIs, grants the runtime SA its roles, and creates the Secret Manager
secrets from your local `.env`:

```
cd travel-buddy-backend
./scripts/setup_cloudrun_iam_secrets.sh
```

The runtime SA is `gemini-vertex-422@travel-buddy-500309.iam.gserviceaccount.com`
— it already has **Vertex AI User** (the Gemini access). The script also gives it
**Secret Manager Secret Accessor** and lets your gcloud account impersonate it.

## Deploy

```
./scripts/deploy_cloudrun.sh
```

This builds the Dockerfile via Cloud Build and deploys, running the service **as**
the Vertex SA with `GEMINI_USE_VERTEX=true`, `GOOGLE_CLOUD_PROJECT=travel-buddy-500309`,
`GOOGLE_CLOUD_LOCATION=global`. Auth comes from the runtime SA via ADC — the code's
`_vertex_credentials()` returns `None` and the SDK uses the metadata server.

## Verify

```
curl -s https://<service-url>/health/ready | python3 -m json.tool
```

Expect `database.ok: true` and `config.gemini_backend: "vertex"`. On Cloud Run
`gemini_vertex_creds` reports `"adc"` — and that is **correct here** (unlike on a
non-GCP host): ADC resolves to the runtime SA. Then send one real chat message
and confirm tokens stream with no `gemini_access_denied` in the logs.

## Cut the web app over

In `travel-buddy-web/.env.local`:
```
NEXT_PUBLIC_API_BASE_URL=https://<service-url>
```
If the web origin isn't already in `ALLOWED_ORIGINS` (set in
`scripts/deploy_cloudrun.sh`), add it and redeploy.

## Notes / knobs

- **Cold starts:** `--min-instances 0` (cheapest) means the first request after
  idle pays a few seconds of cold start. Set `--min-instances 1` for a warm
  instance (costs ~1 always-on instance).
- **Streaming:** `/chat/stream` SSE works on Cloud Run; request `--timeout` is
  600s, raise if turns can run longer.
- **DB:** `DATABASE_URL` points at the Supabase pooler (public host) — reachable
  from Cloud Run with no VPC connector.
- **Region:** `europe-west2` (London), closest to the Supabase DB in `eu-west-2`.
  Vertex stays `global` regardless (the gemini-3.x tier is global-only;
  `us-central1`/regional 404s).
