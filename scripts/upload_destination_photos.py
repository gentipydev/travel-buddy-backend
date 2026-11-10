"""Process and upload the destination gallery photos listed in
`scripts/data/destination_photos.json` to the `place-photos` bucket at
`destinations/{slug}/{file}`. The gallery endpoint lists that folder, so
the mobile and web apps pick the photos up with no code change.

For each manifest entry the original is read from
`scripts/data/destination_photos/{slug}/` (git-ignored); if it isn't
there yet it's downloaded from the entry's `download_url`. Each original
is turned upright, converted to sRGB, shrunk to a 4000 px long edge,
saved as a progressive JPEG with all metadata removed, and squeezed
under 4.9 MB if needed. A small copy (see `make_thumbnail`) is uploaded
next to it at `destinations/{slug}/thumbs/{file}` for the gallery grid.
Uploads use upsert, so re-running is safe.

Usage (from the project root, with deps installed):
    python -m scripts.upload_destination_photos              # all places
    python -m scripts.upload_destination_photos --slug berat
    python -m scripts.upload_destination_photos --dry-run    # no upload
    python -m scripts.upload_destination_photos --slug berat --prune

`--prune` also deletes files in the folder that aren't in the manifest,
after printing them and asking for confirmation (Supabase has no undo).
Apollonia is skipped unless it's named with `--slug apollonia`.
"""
import argparse
import asyncio
import io
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from PIL import Image, ImageCms, ImageOps

from app.config import get_supabase
from app.integrations.supabase_storage import thumb_path

DATA_DIR = Path(__file__).parent / "data"
MANIFEST_PATH = DATA_DIR / "destination_photos.json"
ORIGINALS_DIR = DATA_DIR / "destination_photos"

BUCKET = "place-photos"
BUCKET_FILE_LIMIT = 5_242_880  # place-photos file_size_limit (setup_storage.py)
TARGET_MAX_BYTES = 4_900_000   # margin under the bucket limit
MAX_LONG_EDGE = 4000           # the phone never shows more than ~3300 px across
START_QUALITY = 88
MIN_QUALITY = 80
MAX_PHOTOS_PER_PLACE = 9       # names sort as text: x-10.jpg would sort before x-2.jpg
PROTECTED_SLUGS = {"apollonia"}
# Small copies for grids and carousels. The app decodes images at screen
# width (~1180 px on a 3x phone), so 1280 px wide looks the same there at a
# fraction of the size. Full-size files are kept for the fullscreen views.
THUMB_MAX_SIZE = (1280, 1920)
THUMB_QUALITY = 80
# Seconds browsers, the CDN and the app's disk cache may reuse a file before
# re-checking it. Without this storage3 sends a header Supabase ignores and
# files are served `no-cache`.
CACHE_SECONDS = "86400"
USER_AGENT = "TravelBuddyPhotoUpload/0.1 (https://travel-buddy-web-2h4g.onrender.com) python-httpx"  # Wikimedia wants contact info

REQUIRED_KEYS = {"file", "source", "page_url", "download_url", "author", "license", "subject"}

Image.MAX_IMAGE_PIXELS = 200_000_000  # some Commons originals are very large


def load_manifest() -> dict[str, list[dict]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for slug, entries in manifest.items():
        if len(entries) > MAX_PHOTOS_PER_PLACE:
            raise ValueError(f"{slug}: {len(entries)} photos, max is {MAX_PHOTOS_PER_PLACE}")
        names = set()
        for e in entries:
            missing = REQUIRED_KEYS - e.keys()
            if missing:
                raise ValueError(f"{slug}/{e.get('file', '?')}: missing keys {missing}")
            if not re.fullmatch(rf"{re.escape(slug)}-[1-9]\.jpg", e["file"]):
                raise ValueError(f"{slug}: file name {e['file']!r} must be {slug}-<1..9>.jpg")
            if e["file"] in names:
                raise ValueError(f"{slug}: duplicate file name {e['file']}")
            names.add(e["file"])
    return manifest


def original_path(slug: str, entry: dict, *, originals_dir: Path = ORIGINALS_DIR) -> Path:
    name = unquote(Path(urlparse(entry["download_url"]).path).name)
    name = re.sub(r"[^\w.\-]+", "_", name)
    return originals_dir / slug / name


async def ensure_original(
    client: httpx.AsyncClient, slug: str, entry: dict, *, originals_dir: Path = ORIGINALS_DIR
) -> Path:
    path = original_path(slug, entry, originals_dir=originals_dir)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        r = await client.get(entry["download_url"])
        if r.status_code in (403, 429, 503):
            await asyncio.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"  downloaded {path.name} ({len(r.content) / 1e6:.1f} MB)")
        return path
    raise RuntimeError(f"could not download {entry['download_url']}")


def to_srgb(img: Image.Image) -> Image.Image:
    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            mode = "RGB" if img.mode not in ("RGB", "CMYK") else img.mode
            img = ImageCms.profileToProfile(img.convert(mode), src, dst, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError):
            pass  # unreadable profile: fall back to a plain conversion
    return img.convert("RGB")


def encode(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    # No exif / icc_profile arguments: the output carries no metadata.
    img.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def process(path: Path) -> tuple[bytes, tuple[int, int], int]:
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im)
        img = to_srgb(img)
    if max(img.size) > MAX_LONG_EDGE:
        img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)

    quality = START_QUALITY
    data = encode(img, quality)
    while len(data) > TARGET_MAX_BYTES and quality > MIN_QUALITY:
        quality -= 2
        data = encode(img, quality)
    while len(data) > TARGET_MAX_BYTES:
        w, h = img.size
        img = img.resize((int(w * 0.9), int(h * 0.9)), Image.Resampling.LANCZOS)
        data = encode(img, quality)
    return data, img.size, quality


def make_thumbnail(data: bytes) -> bytes:
    """Small copy of an already-processed JPEG: at most 1280 px wide and
    1920 px tall, progressive, no metadata."""
    with Image.open(io.BytesIO(data)) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail(THUMB_MAX_SIZE, Image.Resampling.LANCZOS)
    return encode(img, THUMB_QUALITY)


def upload_options() -> dict[str, str]:
    return {"content-type": "image/jpeg", "upsert": "true", "cache-control": CACHE_SECONDS}


async def upload_with_thumbnail(bucket, key: str, data: bytes) -> int:
    """Uploads the full JPEG and its small copy. Returns the copy's size."""
    thumb = make_thumbnail(data)
    await bucket.upload(key, data, upload_options())
    await bucket.upload(thumb_path(key), thumb, upload_options())
    return len(thumb)


async def prune(bucket, slug: str, keep: set[str], *, folder: str | None = None) -> None:
    folder = folder or f"destinations/{slug}"
    files = await bucket.list(path=folder)
    extra = sorted(f["name"] for f in files if f.get("id") and f["name"] not in keep)
    if not extra:
        print("  prune: nothing to remove")
        return
    print(f"  prune: not in manifest: {', '.join(extra)}")
    answer = input(f"  Delete these {len(extra)} files from {folder}/? [y/N] ")
    if answer.strip().lower() != "y":
        print("  prune: skipped")
        return
    await bucket.remove([f"{folder}/{n}" for n in extra])
    print(f"  prune: removed {len(extra)}")


async def run(slug_filter: str | None, dry_run: bool, do_prune: bool) -> None:
    manifest = load_manifest()
    if slug_filter:
        if slug_filter not in manifest:
            raise SystemExit(f"{slug_filter!r} is not in {MANIFEST_PATH.name}")
        slugs = [slug_filter]
    else:
        slugs = [s for s in manifest if s not in PROTECTED_SLUGS]

    bucket = None if dry_run else (await get_supabase()).storage.from_(BUCKET)
    uploaded = 0
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True
    ) as client:
        for slug in slugs:
            print(f"{slug}:")
            for entry in manifest[slug]:
                src = await ensure_original(client, slug, entry)
                data, (w, h), q = process(src)
                if len(data) >= BUCKET_FILE_LIMIT:
                    raise SystemExit(f"  {entry['file']}: {len(data)} bytes, over the bucket limit")
                key = f"destinations/{slug}/{entry['file']}"
                line = f"  {entry['file']}: {w}x{h}, q{q}, {len(data) / 1e6:.2f} MB  <- {src.name}"
                if dry_run:
                    print(f"{line}  (dry run)")
                    continue
                thumb_bytes = await upload_with_thumbnail(bucket, key, data)
                uploaded += 1
                print(f"{line}  + thumb {thumb_bytes / 1e3:.0f} KB")
            if do_prune:
                if dry_run:
                    print("  prune: skipped in dry run")
                else:
                    await prune(bucket, slug, {e["file"] for e in manifest[slug]})
    print(f"upload_destination_photos: uploaded={uploaded} dry_run={dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slug", help="only this destination")
    parser.add_argument("--dry-run", action="store_true", help="process and print, upload nothing")
    parser.add_argument("--prune", action="store_true", help="also delete files not in the manifest")
    args = parser.parse_args()
    asyncio.run(run(args.slug, args.dry_run, args.prune))


if __name__ == "__main__":
    main()
