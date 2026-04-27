from app.config import get_supabase

PLACE_PHOTOS_BUCKET = "place-photos"
TRIP_EXPORTS_BUCKET = "trip-exports"
CHAT_ATTACHMENTS_BUCKET = "chat-attachments"


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
    return supabase.storage.from_(PLACE_PHOTOS_BUCKET).get_public_url(file_path)


async def get_place_photos(place_id: str) -> list[str]:
    supabase = await get_supabase()
    files = await supabase.storage.from_(PLACE_PHOTOS_BUCKET).list(path=place_id)
    return [
        supabase.storage.from_(PLACE_PHOTOS_BUCKET).get_public_url(f"{place_id}/{f['name']}")
        for f in files if f.get("name")
    ]


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
