"""Idempotent upsert of `scripts/data/artifacts.json` into the artifacts
table. Run after editing the catalog or after adding/removing rows. The
trigger on `search_text` updates automatically on every INSERT / UPDATE.

Usage (from the project root, with deps installed):
    python -m scripts.seed_artifacts             # upsert
    python -m scripts.seed_artifacts --dry-run   # do everything, then roll back
    python -m scripts.seed_artifacts --prune     # also delete rows the JSON no longer has

`--prune` lists the rows it would delete and asks first.

This does NOT upload images. Upload them first with
`python -m scripts.upload_artifact_photos` (it reads each entry's `photo`
block), so no row points at a missing file. The mobile app reads images by
URL via the images router, not via this script.
"""
import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import delete, select

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
# Collectible icons bundled in the mobile app (assets/images/collectibles/).
ICON_NAMES = {"jewelry", "picture", "scroll", "statue", "textile", "vase"}


def _load_catalog() -> list[dict]:
    raw = json.loads(CATALOG_PATH.read_text())
    destinations = set(raw.get("_destinations", []))
    entries = raw.get("artifacts", [])
    if not isinstance(entries, list):
        raise ValueError("artifacts.json: top-level `artifacts` must be a list")
    ids, paths, slots = set(), set(), set()
    for e in entries:
        missing = {"id", "destination_slug", "category", "title", "image_path"} - e.keys()
        if missing:
            raise ValueError(f"artifact {e.get('id', '?')} missing required keys: {missing}")
        aid, slug = e["id"], e["destination_slug"]
        if e["category"] not in {"highlight", "collectible"}:
            raise ValueError(f"artifact {aid}: category must be highlight|collectible")
        if slug not in destinations:
            raise ValueError(f"artifact {aid}: unknown destination_slug {slug!r}")
        if not e["image_path"].startswith(f"{slug}/"):
            raise ValueError(f"artifact {aid}: image_path must start with {slug}/")
        if e["category"] == "collectible" and e.get("icon_name") not in ICON_NAMES:
            raise ValueError(f"artifact {aid}: icon_name must be one of {sorted(ICON_NAMES)}")
        # The API orders by (category, sort_order) with no tiebreaker.
        slot = (slug, e["category"], e.get("sort_order", 0))
        for seen, value, what in ((ids, aid, "id"), (paths, e["image_path"], "image_path"),
                                  (slots, slot, "slug/category/sort_order")):
            if value in seen:
                raise ValueError(f"artifact {aid}: duplicate {what} {value!r}")
            seen.add(value)
    return entries


async def _rows_to_prune(keep_ids: set[str]) -> list[str]:
    async with async_session() as db:
        rows = (await db.execute(
            select(Artifact.id, Artifact.destination_slug, Artifact.title)
        )).all()
    stale = [r for r in rows if r.id not in keep_ids]
    if not stale:
        print("prune: nothing to remove")
        return []
    print("prune: in the database but not in artifacts.json:")
    for r in stale:
        print(f"  {r.destination_slug:<14} {r.id:<40} {r.title}")
    answer = input(f"Delete these {len(stale)} rows? [y/N] ")
    if answer.strip().lower() != "y":
        print("prune: skipped")
        return []
    return [r.id for r in stale]


async def seed(dry_run: bool, do_prune: bool) -> None:
    entries = _load_catalog()
    # Ask before opening the write transaction, so it isn't held open on input().
    to_delete = await _rows_to_prune({e["id"] for e in entries}) if do_prune else []
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

        if to_delete:
            await db.execute(delete(Artifact).where(Artifact.id.in_(to_delete)))

        if dry_run:
            await db.rollback()
        else:
            await db.commit()

    total = inserted + updated + unchanged
    print(
        f"seed_artifacts: inserted={inserted} updated={updated} "
        f"unchanged={unchanged} deleted={len(to_delete)} total={total} dry_run={dry_run}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="roll back instead of committing")
    parser.add_argument("--prune", action="store_true", help="also delete rows not in the JSON")
    args = parser.parse_args()
    asyncio.run(seed(args.dry_run, args.prune))


if __name__ == "__main__":
    main()
