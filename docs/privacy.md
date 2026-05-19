# Travel Buddy — Privacy

**Status:** Draft. Placeholders marked `TODO` need values from the operator before public release. Last revised 2026-05-19 as part of docs/07 item 8.

This document describes what Travel Buddy collects, where it lives, who it's shared with, how long it's kept, and how a user can request its deletion. It is not a legal-quality privacy policy; treat it as the technical inventory a lawyer will turn into one.

---

## 1. Data we collect

| Category | What | Source | Required? |
|---|---|---|---|
| Account identity | Supabase auth UID, email | Sign-in via Supabase Auth (email/password or OAuth) | Yes — gates per-user data |
| Personalization | Persona text, language, travel style, accessibility preferences | The persona-picker onboarding screen + free-form refinement in chat | No — anonymous chat works without it |
| Chat content | Messages exchanged with the Buddy assistant; attached images | Created via `POST /chat/stream` | No — voluntary feature use |
| Trips | Itineraries: title, summary, dates, days, stops (name / coords / notes), routes | Created via the Buddy "save trip" action or direct trip-editor edits | No — voluntary |
| Uploaded photos | JPG / PNG / WebP attached in the Buddy chat | Uploaded directly from device via `supabase_flutter`'s Storage client to bucket `chat-attachments/<auth_uid>/<uuid>.<ext>` | No — voluntary |
| Saved conversations | Hydrated message thread per `conversations` row | Persisted server-side on `[DONE]` of each turn | Yes when chat is used (so we can resume) |

We do **not** collect: phone numbers, payment data, contacts, calendars, precise device location (we use city-level location only via Google Places lookups against text we already have).

## 2. Where it lives

- **Postgres (Supabase project)** — accounts, preferences, conversations + messages, trips and their days/stops/routes, places cache.
- **Supabase Storage** — uploaded photos (bucket `chat-attachments`), curated catalog images (buckets `place-photos`, `artifact-photos`), generated trip-export PDFs (bucket `trip-exports`).
- **Render** — the FastAPI backend that serves the mobile client. Request logs are JSON; the access log captures method + path + status + latency + request_id but no body content.
- **Sentry** (when `SENTRY_DSN` is set) — exception traces with PII redacted by `app/observability/sentry.py::_scrub_pii` (strips message / persona / email / stop name leaves before send).
- **Braintrust** (when `BRAINTRUST_ENABLED=true`) — LLM trace logs. When `BRAINTRUST_LOG_USER_PAYLOADS=false`, message + assistant text are stripped from spans; token counts and structural metadata still ship.

**Region:** `TODO(operator)` — confirm the Supabase project region. Current default is the project's home region (likely EU west, but verify in the Supabase dashboard).

## 3. Third parties

The backend forwards subsets of user data to these processors:

| Vendor | What we send | Purpose | Where |
|---|---|---|---|
| **Anthropic** (Claude) | Chat messages, persona, trip context, retrieved RAG chunks | LLM completions for the Buddy chat | API call from backend; no data retained by us, retained per Anthropic's policy |
| **Cohere** | Chat-turn user query + candidate chunks | Multilingual reranking of RAG retrieval results | Same — backend-only call |
| **Google Maps Platform** | Stop name + lat/lng (no user content) | Place Details (photos, ratings), Directions (polylines), Static Maps (thumbnail PNG) | Photos + place data cached in our DB for 90 days |
| **Supabase** | Everything in §2 above | Auth, Postgres, Storage | Hosted at the Supabase project region |
| **Sentry** | Exception traces post-`_scrub_pii` | Production error visibility | Conditional on `SENTRY_DSN` |
| **Braintrust** | LLM I/O spans (subject to `LOG_USER_PAYLOADS` flag) | LLM eval + trace dashboards | Conditional on `BRAINTRUST_ENABLED` |

We do **not** ship user data to any analytics product (no Google Analytics, no Mixpanel, no Amplitude, no Segment).

## 4. Retention

- **Account-scoped data** (trips, conversations, messages, prefs, uploaded photos): kept until the user invokes `DELETE /users/me` (the "Delete account" action in the mobile Profile tab — see `app/routers/users.py::delete_me`). On invocation we:
  1. Remove every file under `chat-attachments/<auth_uid>/`.
  2. Delete the local `users` row — FK CASCADE wipes `trips → trip_days → trip_stops → trip_routes`, `conversations → messages`, `user_preferences`.
  3. Call `supabase.auth.admin.delete_user(uid)` to remove the auth row and invalidate the JWT.
- **Postgres backups** (Supabase managed): retained per Supabase's plan default. `TODO(operator)` — confirm retention window in the dashboard and add it here.
- **Caches** (`places_cache`, `directions_cache`): keyed by `place_id` / route hash, NOT by user. Not affected by account deletion.
- **Logs**: Render rotates logs per its standard retention; we don't extend it.
- **Sentry / Braintrust**: retained per each vendor's plan defaults.

## 5. Known privacy considerations

Documented so the operator + legal know what we're shipping:

- **`chat-attachments` is public-read.** The Storage RLS policy (`supabase/migrations/004_create_storage_rls_policies.sql`) gates INSERT to the owner's folder but allows any holder of the URL to fetch. Photos live at `chat-attachments/<auth_uid>/<uuid>.<ext>`; the per-file UUID makes guessing infeasible but the URL itself is not access-controlled. Tightening this to signed URLs requires plumbing auth headers through `Image.network` / `CachedNetworkImage` and is tracked as a separate follow-up.
- **No DELETE RLS policy on `chat-attachments`.** Users cannot delete individual uploaded photos from the mobile client; the backend's service_role key handles cleanup during account delete. Add a per-row owner-scoped DELETE policy if/when a "delete this photo" UI ships.
- **PII scrubbing is allow-list-by-key, not classifier-based.** Sentry / Braintrust scrubbers (`app/observability/sentry.py`, `lib/utils/sentry_scrubber.dart`) redact known PII keys (`message`, `persona`, `content`, `email`, …). New endpoints adding user-text fields under unknown key names will silently leak until the key list is updated. The scrubber tests in `tests/test_observability_sentry.py` and `test/utils/sentry_scrubber_test.dart` document the contract.

## 6. Contact

For data-deletion requests outside the in-app flow, or for any other privacy question:

- **Data Protection Contact:** `TODO(operator)` — e.g. `privacy@<your-domain>`.
- **Operator entity:** `TODO(operator)` — legal name + address that appears on the App Store / Play Store listing.

## 7. Change log

- 2026-05-19 — initial draft (docs/07 item 8).
