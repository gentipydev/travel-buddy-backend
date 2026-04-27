"""Seed the `places` table with well-known Albania landmarks. Idempotent —
rows are matched on `name` and skipped if already present.

Usage (from the project root, with deps installed):
    python -m scripts.seed_places
"""
import asyncio

from sqlalchemy import select

from app.database import Place, async_session


PLACES: list[dict] = [
    {
        "name": "Berat",
        "city": "Berat",
        "region": "Berat County",
        "category": "city",
        "tags": ["unesco", "ottoman", "historic", "castle", "river"],
        "latitude": 40.7058,
        "longitude": 19.9522,
        "rating": 4.8,
        "price_level": None,
    },
    {
        "name": "Gjirokastër",
        "city": "Gjirokastër",
        "region": "Gjirokastër County",
        "category": "city",
        "tags": ["unesco", "ottoman", "historic", "castle", "stone-city"],
        "latitude": 40.0758,
        "longitude": 20.1389,
        "rating": 4.7,
        "price_level": None,
    },
    {
        "name": "Sarandë",
        "city": "Sarandë",
        "region": "Vlorë County",
        "category": "city",
        "tags": ["beach", "ionian", "coast", "summer"],
        "latitude": 39.8750,
        "longitude": 20.0050,
        "rating": 4.5,
        "price_level": None,
    },
    {
        "name": "Ksamil",
        "city": "Ksamil",
        "region": "Vlorë County",
        "category": "beach",
        "tags": ["beach", "ionian", "islands", "turquoise", "summer"],
        "latitude": 39.7667,
        "longitude": 20.0033,
        "rating": 4.7,
        "price_level": None,
    },
    {
        "name": "Butrint",
        "city": "Butrint",
        "region": "Vlorë County",
        "category": "archaeological-site",
        "tags": ["unesco", "ancient", "ruins", "roman", "greek"],
        "latitude": 39.7453,
        "longitude": 20.0225,
        "rating": 4.7,
        "price_level": None,
    },
    {
        "name": "Krujë",
        "city": "Krujë",
        "region": "Durrës County",
        "category": "city",
        "tags": ["castle", "skanderbeg", "historic", "bazaar"],
        "latitude": 41.5094,
        "longitude": 19.7931,
        "rating": 4.6,
        "price_level": None,
    },
    {
        "name": "Theth",
        "city": "Theth",
        "region": "Shkodër County",
        "category": "village",
        "tags": ["mountains", "alps", "hiking", "waterfall", "nature"],
        "latitude": 42.3925,
        "longitude": 19.7692,
        "rating": 4.8,
        "price_level": None,
    },
    {
        "name": "Apollonia",
        "city": "Fier",
        "region": "Fier County",
        "category": "archaeological-site",
        "tags": ["ancient", "ruins", "greek", "illyrian"],
        "latitude": 40.7181,
        "longitude": 19.4694,
        "rating": 4.5,
        "price_level": None,
    },
    {
        "name": "Shkodër",
        "city": "Shkodër",
        "region": "Shkodër County",
        "category": "city",
        "tags": ["lake", "castle", "historic", "cycling", "north"],
        "latitude": 42.0683,
        "longitude": 19.5126,
        "rating": 4.4,
        "price_level": None,
    },
    {
        "name": "Tirana",
        "city": "Tirana",
        "region": "Tirana County",
        "category": "city",
        "tags": ["capital", "nightlife", "cafes", "museums", "pyramid"],
        "latitude": 41.3275,
        "longitude": 19.8189,
        "rating": 4.5,
        "price_level": None,
    },
]


async def seed():
    inserted = 0
    skipped = 0
    async with async_session() as db:
        for data in PLACES:
            existing = (
                await db.execute(select(Place).where(Place.name == data["name"]))
            ).scalar_one_or_none()
            if existing:
                skipped += 1
                continue
            db.add(Place(**data, photos=[]))
            inserted += 1
        await db.commit()
    print(f"seed_places: inserted={inserted} skipped={skipped} total={inserted + skipped}")


if __name__ == "__main__":
    asyncio.run(seed())
