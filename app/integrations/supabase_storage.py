from app.config import get_supabase

PLACE_PHOTOS_BUCKET = "place-photos"
TRIP_EXPORTS_BUCKET = "trip-exports"
CHAT_ATTACHMENTS_BUCKET = "chat-attachments"
ARTIFACT_PHOTOS_BUCKET = "artifact-photos"


async def upload_place_photo(
    place_id: str,
    file_data: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> str:
    supabase = await get_supabase()
    file_path = f"{place_id}/{filename}"
    await supabase.storage.from_(PLACE_PHOTOS_BUCKET).upload(
        file_path,
        file_data,
        {"content-type": content_type, "upsert": "true"},
    )
    return await supabase.storage.from_(PLACE_PHOTOS_BUCKET).get_public_url(file_path)


# Small copies of gallery and artifact photos live in a `thumbs/` subfolder
# next to the full-size file (written by the upload scripts).
THUMBS_FOLDER = "thumbs"


def thumb_path(path: str) -> str:
    """`a/b/name.jpg` -> `a/b/thumbs/name.jpg`."""
    folder, _, name = path.rpartition("/")
    return f"{folder}/{THUMBS_FOLDER}/{name}" if folder else f"{THUMBS_FOLDER}/{name}"


def is_image_file(entry: dict) -> bool:
    """True for an image object in a Storage listing.

    Subfolder entries have a `name` but no `id` (only file objects get a
    UUID). Also drops `.emptyFolderPlaceholder` (the 0-byte sentinel
    Supabase Studio writes when you create a folder via the dashboard) and
    any other non-image objects by checking the stored mimetype.
    """
    return bool(entry.get("id")) and (entry.get("metadata") or {}).get(
        "mimetype", ""
    ).startswith("image/")


async def list_image_names(bucket: str, prefix: str) -> list[str]:
    """Names of the image files directly under `prefix`, sorted."""
    supabase = await get_supabase()
    files = await supabase.storage.from_(bucket).list(path=prefix)
    return sorted(f["name"] for f in files if is_image_file(f))


async def public_url(bucket: str, path: str) -> str:
    supabase = await get_supabase()
    return await supabase.storage.from_(bucket).get_public_url(path)


async def list_bucket_folder(bucket: str, prefix: str) -> list[str]:
    """Public URLs for every image file directly under `prefix` in `bucket`,
    sorted by filename for a stable order without persisting ordering
    metadata."""
    return [await public_url(bucket, f"{prefix}/{name}") for name in await list_image_names(bucket, prefix)]


async def get_place_photos(place_id: str) -> list[str]:
    return await list_bucket_folder(PLACE_PHOTOS_BUCKET, place_id)


async def upload_trip_export(user_id: str, trip_id: str, pdf_data: bytes) -> str:
    supabase = await get_supabase()
    file_path = f"{user_id}/{trip_id}.pdf"
    await supabase.storage.from_(TRIP_EXPORTS_BUCKET).upload(
        file_path,
        pdf_data,
        {"content-type": "application/pdf", "upsert": "true"},
    )
    signed = await supabase.storage.from_(TRIP_EXPORTS_BUCKET).create_signed_url(
        file_path, 3600
    )
    return signed["signedURL"]


async def delete_place_photo(place_id: str, filename: str) -> None:
    supabase = await get_supabase()
    await supabase.storage.from_(PLACE_PHOTOS_BUCKET).remove([f"{place_id}/{filename}"])
