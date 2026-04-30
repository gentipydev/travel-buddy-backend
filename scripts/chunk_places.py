"""Chunk place descriptions into Chunk rows for RAG retrieval.

For each Place with a non-null description, ensures exactly one Chunk row
exists with `source="place_description"` whose `content` mirrors the latest
`Place.description`. Idempotent: re-runs sync existing chunks and clear the
stored embedding when content changes so Step 3 (embed) will re-embed them.

For now we emit one chunk per place. When descriptions grow, this script can
split into multiple chunks (e.g. by paragraph or sentence window) without
changing the rest of the pipeline.

Usage (from the project root, with deps installed):
    python -m scripts.chunk_places
"""
import asyncio

from sqlalchemy import select

from app.db.engine import async_session
from app.db.models import Chunk, Place


CHUNK_SOURCE = "place_description"


async def chunk():
    inserted = 0
    updated = 0
    unchanged = 0
    skipped = 0
    async with async_session() as db:
        places = (await db.execute(select(Place))).scalars().all()
        for place in places:
            if not place.description:
                skipped += 1
                continue
            existing = (
                await db.execute(
                    select(Chunk).where(
                        Chunk.place_id == place.id,
                        Chunk.source == CHUNK_SOURCE,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                db.add(
                    Chunk(
                        place_id=place.id,
                        content=place.description,
                        language="en",
                        source=CHUNK_SOURCE,
                        meta={"chunk_index": 0},
                    )
                )
                inserted += 1
            elif existing.content != place.description:
                existing.content = place.description
                existing.embedding = None  # stale — Step 3 will re-embed
                updated += 1
            else:
                unchanged += 1
        await db.commit()
    total = inserted + updated + unchanged + skipped
    print(
        f"chunk_places: inserted={inserted} updated={updated} "
        f"unchanged={unchanged} skipped_no_description={skipped} total={total}"
    )


if __name__ == "__main__":
    asyncio.run(chunk())
