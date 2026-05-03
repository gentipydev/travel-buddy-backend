"""Idempotent upsert of `scripts/data/artifacts.json` into the artifacts
table. Run after editing the catalog or after adding/removing rows. The
trigger on `search_text` updates automatically on every INSERT / UPDATE.

Usage (from the project root, with deps installed):
    python -m scripts.seed_artifacts

This does NOT upload images — drop those into the `artifact-photos`
Supabase bucket at the `image_path` you set in the JSON. The mobile app
reads images by URL via the manifest router, not via this script.
"""
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.db.engine import async_session
from app.db.models import Artifact


CATALOG_PATH = Path(__file__).parent / "data" / "artifacts.json"

ARTIFACT_FIELDS = {
    "destination_slug",
    "category",
    "sort_order",
    "title",
    "date_text",
    "period",
    "country",
    "culture",
    "medium",
    "dimension",
    "classification",
    "object_type",
    "object_begin_year",
    "object_end_year",
    "image_path",
    "icon_name",
}


def _load_catalog() -> list[dict]:
    raw = json.loads(CATALOG_PATH.read_text())
    entries = raw.get("artifacts", [])
    if not isinstance(entries, list):
        raise ValueError("artifacts.json: top-level `artifacts` must be a list")
    for e in entries:
        missing = {"id", "destination_slug", "category", "title", "image_path"} - e.keys()
        if missing:
            raise ValueError(f"artifact {e.get('id', '?')} missing required keys: {missing}")
        if e["category"] not in {"highlight", "collectible"}:
            raise ValueError(f"artifact {e['id']}: category must be highlight|collectible")
    return entries


async def seed():
    entries = _load_catalog()
    inserted = updated = unchanged = 0

    async with async_session() as db:
        for entry in entries:
            artifact_id = entry["id"]
            existing = (await db.execute(
                select(Artifact).where(Artifact.id == artifact_id)
            )).scalar_one_or_none()

            payload = {k: entry.get(k) for k in ARTIFACT_FIELDS if k in entry}
            payload.setdefault("sort_order", 0)

            if existing is None:
                db.add(Artifact(id=artifact_id, **payload))
                inserted += 1
                continue

            changed = False
            for k, v in payload.items():
                if getattr(existing, k) != v:
                    setattr(existing, k, v)
                    changed = True
            if changed:
                updated += 1
            else:
                unchanged += 1

        await db.commit()

    total = inserted + updated + unchanged
    print(
        f"seed_artifacts: inserted={inserted} updated={updated} "
        f"unchanged={unchanged} total={total}"
    )


if __name__ == "__main__":
    asyncio.run(seed())
