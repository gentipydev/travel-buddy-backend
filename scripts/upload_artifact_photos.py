"""Process and upload the artifact pictures listed in `scripts/data/artifacts.json`
to the `artifact-photos` bucket, each at its entry's `image_path`. Run this
before `scripts.seed_artifacts`, so no row ever points at a missing file.

Only entries with a `photo` block (the picture's Wikimedia source and
licence) are handled. The original is read from
`scripts/data/artifact_photos/{slug}/` (git-ignored) and downloaded from
`photo.download_url` if it isn't there yet. Processing is the same as for
the gallery photos (see `upload_destination_photos.py`): upright, sRGB,
4000 px long edge, progressive JPEG under 4.9 MB, no metadata, plus a small
copy at `{slug}/thumbs/{file}` for the carousel. Uploads use upsert, so
re-running is safe.

Usage (from the project root, with dev deps installed):
    python -m scripts.upload_artifact_photos              # all places
    python -m scripts.upload_artifact_photos --slug berat
    python -m scripts.upload_artifact_photos --dry-run --preview-dir /tmp/previews
    python -m scripts.upload_artifact_photos --slug berat --prune

`--preview-dir` writes each processed JPEG plus `<name>.card.jpg`, its 2:3
centre crop (what the carousel card shows). `--prune` also deletes files in
a place's folder that no catalog entry points at, after printing them and
asking (Supabase has no undo). Apollonia is never pruned.
"""
import argparse
import asyncio
import io
import json
import re
from pathlib import Path

import httpx
from PIL import Image

from app.config import get_supabase
from app.integrations.supabase_storage import ARTIFACT_PHOTOS_BUCKET
from scripts.upload_destination_photos import (
    PROTECTED_SLUGS,
    USER_AGENT,
    ensure_original,
    process,
    prune,
    upload_with_thumbnail,
)

DATA_DIR = Path(__file__).parent / "data"
CATALOG_PATH = DATA_DIR / "artifacts.json"
ORIGINALS_DIR = DATA_DIR / "artifact_photos"

BUCKET_FILE_LIMIT = 10_485_760  # artifact-photos file_size_limit (setup_storage.py)
PHOTO_KEYS = {"source", "page_url", "download_url", "author", "license"}
CARD_PREVIEW_WIDTH = 600


def load_catalog() -> tuple[list[str], list[dict]]:
    raw = json.loads(CATALOG_PATH.read_text())
    slugs, entries = raw["_destinations"], raw["artifacts"]
    for e in entries:
        photo = e.get("photo")
        if photo is None:
            continue
        missing = PHOTO_KEYS - photo.keys()
        if missing:
            raise ValueError(f"{e['id']}: photo block is missing {sorted(missing)}")
        slug = e["destination_slug"]
        if not re.fullmatch(rf"{re.escape(slug)}/[a-z0-9-]+\.jpg", e["image_path"]):
            raise ValueError(f"{e['id']}: image_path {e['image_path']!r} must be {slug}/<kebab>.jpg")
    return slugs, entries


def write_previews(preview_dir: Path, image_path: str, data: bytes) -> None:
    out = preview_dir / image_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        cw, ch = (round(h * 2 / 3), h) if w / h > 2 / 3 else (w, round(w * 3 / 2))
        left, top = (w - cw) // 2, (h - ch) // 2
        card = im.crop((left, top, left + cw, top + ch))
        card.thumbnail((CARD_PREVIEW_WIDTH, CARD_PREVIEW_WIDTH * 3 // 2))
        card.save(out.with_name(f"{out.stem}.card.jpg"), quality=85)


async def check_uploaded(bucket, slugs: list[str], entries: list[dict]) -> None:
    missing = []
    for slug in slugs:
        expected = {
            Path(e["image_path"]).name
            for e in entries
            if e["destination_slug"] == slug and "photo" in e
        }
        if not expected:
            continue
        present = {f["name"] for f in await bucket.list(path=slug) if f.get("id")}
        missing += [f"{slug}/{name}" for name in sorted(expected - present)]
    if missing:
        raise SystemExit(f"missing from the bucket after upload: {', '.join(missing)}")
    print("check: every expected file is in the bucket")


async def run(slug_filter: str | None, dry_run: bool, do_prune: bool, preview_dir: Path | None) -> None:
    slugs, entries = load_catalog()
    if slug_filter:
        if slug_filter not in slugs:
            raise SystemExit(f"{slug_filter!r} is not in {CATALOG_PATH.name} _destinations")
        slugs = [slug_filter]
    else:
        slugs = [s for s in slugs if s not in PROTECTED_SLUGS]

    bucket = None if dry_run else (await get_supabase()).storage.from_(ARTIFACT_PHOTOS_BUCKET)
    uploaded = 0
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
    ) as client:
        for slug in slugs:
            todo = [e for e in entries if e["destination_slug"] == slug and "photo" in e]
            print(f"{slug}:")
            for e in todo:
                src = await ensure_original(client, slug, e["photo"], originals_dir=ORIGINALS_DIR)
                data, (w, h), q = process(src)
                if len(data) >= BUCKET_FILE_LIMIT:
                    raise SystemExit(f"  {e['image_path']}: {len(data)} bytes, over the bucket limit")
                if preview_dir:
                    write_previews(preview_dir, e["image_path"], data)
                line = f"  {e['image_path']}: {w}x{h}, q{q}, {len(data) / 1e6:.2f} MB  <- {src.name}"
                if dry_run:
                    print(f"{line}  (dry run)")
                    continue
                thumb_bytes = await upload_with_thumbnail(bucket, e["image_path"], data)
                uploaded += 1
                print(f"{line}  + thumb {thumb_bytes / 1e3:.0f} KB")
            if do_prune:
                if slug in PROTECTED_SLUGS:
                    print("  prune: skipped, protected place")
                elif dry_run:
                    print("  prune: skipped in dry run")
                else:
                    keep = {Path(e["image_path"]).name for e in entries if e["destination_slug"] == slug}
                    await prune(bucket, slug, keep, folder=slug)
        if not dry_run:
            await check_uploaded(bucket, slugs, entries)
    print(f"upload_artifact_photos: uploaded={uploaded} dry_run={dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", help="only this destination")
    parser.add_argument("--dry-run", action="store_true", help="process and print, upload nothing")
    parser.add_argument("--prune", action="store_true", help="also delete files no entry points at")
    parser.add_argument("--preview-dir", type=Path, help="write processed JPEGs + 2:3 card crops here")
    args = parser.parse_args()
    asyncio.run(run(args.slug, args.dry_run, args.prune, args.preview_dir))


if __name__ == "__main__":
    main()
