"""Create the Supabase Storage buckets. Idempotent — safe to re-run.

Usage (from the project root, with deps installed):
    python -m scripts.setup_storage
"""
import asyncio

from app.config import get_supabase


async def setup():
    supabase = await get_supabase()

    try:
        await supabase.storage.create_bucket(
            "place-photos",
            options={
                "public": True,
                "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"],
                "file_size_limit": 5242880,
            },
        )
        print("Created place-photos bucket")
    except Exception as e:
        print(f"place-photos: {e}")

    try:
        await supabase.storage.create_bucket(
            "trip-exports",
            options={
                "public": False,
                "allowed_mime_types": ["application/pdf"],
                "file_size_limit": 10485760,
            },
        )
        print("Created trip-exports bucket")
    except Exception as e:
        print(f"trip-exports: {e}")

    try:
        await supabase.storage.create_bucket(
            "chat-attachments",
            options={
                "public": True,
                "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"],
                "file_size_limit": 5242880,
            },
        )
        print("Created chat-attachments bucket")
    except Exception as e:
        print(f"chat-attachments: {e}")

    try:
        await supabase.storage.create_bucket(
            "artifact-photos",
            options={
                "public": True,
                "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"],
                "file_size_limit": 10485760,
            },
        )
        print("Created artifact-photos bucket")
    except Exception as e:
        print(f"artifact-photos: {e}")

    # Storage RLS for chat-attachments is tracked in
    # supabase/migrations/004_create_storage_rls_policies.sql — apply it via
    # the Supabase SQL editor (no CLI configured in this repo). Without it,
    # authenticated users can write into each other's folders.


if __name__ == "__main__":
    asyncio.run(setup())
