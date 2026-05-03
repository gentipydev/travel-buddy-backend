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


async def list_bucket_folder(bucket: str, prefix: str) -> list[str]:
    """Public URLs for every image file directly under `prefix` in `bucket`.

    Subfolder/prefix entries in Supabase Storage's listing have a `name`
    but no `id` (only file objects get a UUID), so we filter on `id`.
    Also drops `.emptyFolderPlaceholder` (the 0-byte sentinel Supabase
    Studio writes when you create a folder via the dashboard) and any
    other non-image objects by checking the stored mimetype.
    Sorted by filename for a stable order without persisting ordering
    metadata.
    """
    supabase = await get_supabase()
    files = await supabase.storage.from_(bucket).list(path=prefix)
    names = sorted(
        f["name"]
        for f in files
        if f.get("id")
        and (f.get("metadata") or {}).get("mimetype", "").startswith("image/")
    )
    return [
        await supabase.storage.from_(bucket).get_public_url(f"{prefix}/{name}")
        for name in names
    ]


async def get_place_photos(place_id: str) -> list[str]:
    return await list_bucket_folder(PLACE_PHOTOS_BUCKET, place_id)


async def get_artifact_image_url(image_path: str) -> str:
    supabase = await get_supabase()
    return supabase.storage.from_(ARTIFACT_PHOTOS_BUCKET).get_public_url(image_path)


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
