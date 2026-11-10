"""Add a small copy next to every photo already in Storage.

The upload scripts now write `.../thumbs/{file}` alongside each full-size
photo. This one-off covers the files uploaded before that: for each place it
lists `place-photos/destinations/{slug}/` and `artifact-photos/{slug}/`,
downloads every image, and uploads `make_thumbnail()` of it to the matching
`thumbs/` path. It only adds files — the full-size photos are never changed —
so it is safe for protected places like Apollonia, and re-running it just
overwrites the same small copies.

Usage (from the project root, with deps installed):
    python -m scripts.backfill_thumbnails --dry-run
    python -m scripts.backfill_thumbnails
    python -m scripts.backfill_thumbnails --slug berat
"""
import argparse
import asyncio

from app.config import get_supabase
from app.integrations.supabase_storage import (
    ARTIFACT_PHOTOS_BUCKET,
    PLACE_PHOTOS_BUCKET,
    is_image_file,
    thumb_path,
)
from scripts.upload_artifact_photos import load_catalog
from scripts.upload_destination_photos import make_thumbnail, upload_options


async def backfill_folder(storage, bucket_name: str, folder: str, dry_run: bool) -> int:
    bucket = storage.from_(bucket_name)
    names = sorted(f["name"] for f in await bucket.list(path=folder) if is_image_file(f))
    for name in names:
        key = f"{folder}/{name}"
        data = await bucket.download(key)
        thumb = make_thumbnail(data)
        line = f"  {bucket_name}/{key}: {len(data) / 1e6:.2f} MB -> {len(thumb) / 1e3:.0f} KB"
        if dry_run:
            print(f"{line}  (dry run)")
            continue
        await bucket.upload(thumb_path(key), thumb, upload_options())
        print(line)
    return len(names)


async def run(slug_filter: str | None, dry_run: bool) -> None:
    slugs, _ = load_catalog()
    if slug_filter:
        if slug_filter not in slugs:
            raise SystemExit(f"{slug_filter!r} is not a known destination")
        slugs = [slug_filter]

    storage = (await get_supabase()).storage
    total = 0
    for slug in slugs:
        print(f"{slug}:")
        total += await backfill_folder(storage, PLACE_PHOTOS_BUCKET, f"destinations/{slug}", dry_run)
        total += await backfill_folder(storage, ARTIFACT_PHOTOS_BUCKET, slug, dry_run)
    print(f"backfill_thumbnails: files={total} dry_run={dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", help="only this destination")
    parser.add_argument("--dry-run", action="store_true", help="download and resize, upload nothing")
    args = parser.parse_args()
    asyncio.run(run(args.slug, args.dry_run))


if __name__ == "__main__":
    main()
