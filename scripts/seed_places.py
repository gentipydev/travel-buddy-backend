"""Seed the `places` table with well-known Albania landmarks. Idempotent —
rows are matched on `name`. New rows are inserted; existing rows have their
`description` re-synced from this file (other columns are preserved so that
photos uploaded later or manual edits are not clobbered).

Usage (from the project root, with deps installed):
    python -m scripts.seed_places
"""
import asyncio

from sqlalchemy import select

from app.db.engine import async_session
from app.db.models import Place


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
        "description": (
            "Berat is a UNESCO World Heritage city in central Albania, on the Osum "
            "River at the foot of Mount Tomorr. It is known as the City of a "
            "Thousand Windows for the rows of stacked white Ottoman houses climbing "
            "both banks of the river. Berat Castle (Kalaja e Beratit), inhabited "
            "since the 4th century BC, still has families living inside its medieval "
            "walls and shelters the Onufri National Iconographic Museum, named after "
            "the 16th-century Albanian icon painter. The Mangalem (historically "
            "Muslim) and Gorica (historically Christian) quarters face each other "
            "across the river, linked by the Gorica Bridge (1780). Notable monuments "
            "include the Lead Mosque, the King's Mosque (15th century), the "
            "Bachelors' Mosque, and the Holy Trinity Church inside the castle. Berat "
            "is roughly 2 hours by car south of Tirana. Nearby half-day add-ons "
            "within the Berat region include the Berat Wine Route and the Osum "
            "Canyon. Best visited in spring or autumn; summers regularly exceed "
            "35°C."
        ),
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
        "description": (
            "Gjirokastër is a UNESCO-listed Ottoman city in southern Albania's "
            "Drino Valley, often called the Stone City for its slate-roofed tower "
            "houses (kullas) cascading down a steep hillside. It is dominated by "
            "Gjirokastër Castle, one of the largest in the Balkans, which houses "
            "the National Museum of Armaments, a captured US Air Force plane from "
            "1957, and hosts the National Folklore Festival every five years. The "
            "Old Bazaar, with its cobbled streets and characteristic stone arches, "
            "sells traditional textiles, wood carvings, and qeleshe felt caps. "
            "Notable kullas open to visitors include the Skenduli House and the "
            "Zekate House (1812). The city was the birthplace of writer Ismail "
            "Kadare and former dictator Enver Hoxha. Nearby attractions include "
            "the Blue Eye spring (Syri i Kaltër, about 18 km away) and the "
            "Antigonea archaeological park. Gjirokastër lies on the SH4 highway, "
            "around 35 km from the Greek border crossing at Kakavija."
        ),
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
        "description": (
            "Sarandë is a coastal city on the Albanian Riviera, facing the Greek "
            "island of Corfu across a narrow strait of the Ionian Sea. Once the "
            "ancient Greek port of Onchesmos and later named after the Forty "
            "Saints (Agioi Saranda) of Sebaste, it grew rapidly after Albania's "
            "transition in the 1990s into the country's main southern beach "
            "resort. The pedestrian seafront promenade runs along a crescent bay "
            "lined with cafés, bars, and seafood restaurants. Key sights include "
            "the 5th-century Synagogue ruins, the 16th-century Lëkursi Castle on "
            "the hill above (sweeping views and a popular sunset restaurant), and "
            "the early Christian basilica preserved in the city museum. Sarandë "
            "functions as the gateway to the southern Riviera: ferries connect "
            "daily to Corfu (30–70 min) and the UNESCO archaeological site of "
            "Butrint lies 18 km south. Beaches inside the city are pebbly; the "
            "more famous turquoise coves of Ksamil and Pulëbardha are a short bus "
            "or taxi ride away. Peak season is July–August; late May, June, and "
            "September are quieter and noticeably cheaper."
        ),
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
        "description": (
            "Ksamil is a coastal village in southernmost Albania, part of "
            "Butrint National Park and famous for the four small islands lying "
            "just offshore in the Ionian Sea. Once a sleepy citrus-growing "
            "settlement under communism, it has become one of Albania's most "
            "photographed beach destinations thanks to its turquoise water, white "
            "pebble coves, and shallow bays where the islands are reachable by "
            "short swim, kayak, or pedalo in summer. The main beaches — Ksamil "
            "Beach, Pulëbardha, and Mirror Beach — are organized with sun-loungers "
            "and beach bars in season. Inland, the village retains stretches of "
            "olive groves and orange orchards. Ksamil sits about 4 km north of "
            "the UNESCO archaeological site of Butrint and 17 km south of "
            "Sarandë; the SH81 coastal road connects them. The wider area is part "
            "of a protected lagoon-and-wetland ecosystem (Lake Butrint, Vivari "
            "Channel) home to flamingos and herons. Best visited late May–June or "
            "September; July–August is extremely crowded and accommodation prices "
            "roughly triple."
        ),
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
        "description": (
            "Butrint is a UNESCO World Heritage archaeological site at the "
            "southern tip of Albania, on a forested peninsula between Lake "
            "Butrint and the Vivari Channel. Continuously inhabited for nearly "
            "2,500 years, it preserves layered remains from prehistoric, Greek, "
            "Roman, Byzantine, Venetian, and Ottoman periods, making it one of "
            "the most complete archaeological landscapes in the Mediterranean. "
            "According to Virgil's Aeneid, Butrint was founded by Trojan "
            "refugees; the historical record begins in the 7th century BC as a "
            "Corcyrean colony. Highlights include the 3rd-century BC Greek "
            "theatre (still used for performances), the 6th-century baptistery "
            "with one of the largest surviving early-Christian mosaic floors in "
            "the Balkans, the Lion Gate, the Venetian Triangular Castle, and the "
            "on-site museum housing finds from a century of excavation. The "
            "wider Butrint National Park also protects rare wetlands, dolphins, "
            "and migratory birds. Butrint is 18 km south of Sarandë, reachable "
            "by car, taxi, or the regular municipal bus. Allow 2–3 hours for the "
            "main site loop; bring water and mosquito repellent in summer."
        ),
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
        "description": (
            "Krujë is a hilltop town in central Albania, perched on a narrow "
            "ridge of Mount Krujë roughly 32 km north of Tirana. It was the "
            "capital of the medieval lord Gjergj Kastrioti Skanderbeg, who held "
            "off three Ottoman sieges (1450, 1466, 1467) from the town's castle "
            "and is now Albania's national hero. Krujë Castle dominates the "
            "ridge and contains the Skanderbeg Museum, designed by Pranvera "
            "Hoxha and Pirro Vaso and opened in 1982, with armour, manuscripts, "
            "and replicas of Skanderbeg's helmet and sword. Just below the "
            "castle wall, the Old Bazaar (Pazari i Vjetër) is a restored "
            "Ottoman-era cobbled street of around 60 wooden-shuttered shops "
            "selling antiques, copperware, kilims, and traditional embroidery — "
            "among the best handicraft markets in Albania. Other sights include "
            "the Ethnographic Museum housed in an 18th-century Ottoman mansion "
            "and the Sari Salltik teqe of the Bektashi order on the cliff above "
            "the town. Krujë is reachable from Tirana International Airport "
            "(about 15 km away); the town is small enough to cover in a "
            "half-day visit."
        ),
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
        "description": (
            "Theth is a remote mountain village in the Albanian Alps (Bjeshkët "
            "e Nemuna), in northern Albania's Shkodër County. Surrounded by "
            "limestone peaks rising over 2,500 m, it stayed isolated for "
            "centuries, reachable only by a rough road over the Thethi Pass. "
            "The village is known for its traditional stone-roofed houses, the "
            "Lock-in Tower (Kulla e Ngujimit) where men marked for blood feuds "
            "once sought refuge under the Kanun, and the Theth Church (1892). "
            "Hikers come for the Blue Eye of Theth — a turquoise karst spring — "
            "Grunas Waterfall, and the iconic Valbona–Theth trail over the "
            "Valbona Pass (around 7–9 hours one way). Theth sits inside Theth "
            "National Park and is a focal point of Albanian highland culture. "
            "Best visited June–September; the road is typically impassable in "
            "winter due to snow and most guesthouses close. Access has improved "
            "since the road from Shkodër was paved in 2021–2022, but the final "
            "descent into the valley remains steep and slow — allow about 3 "
            "hours by car from Shkodër."
        ),
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
        "description": (
            "Apollonia is an ancient Greek and Roman archaeological site near "
            "the village of Pojan, about 12 km west of Fier in southwestern "
            "Albania. Founded around 588 BC by Corinthian and Corcyrean "
            "colonists, it became one of the most important cities of the "
            "Adriatic coast, famous in antiquity for its school of philosophy — "
            "where the future emperor Octavian (Augustus) was studying when "
            "news arrived of Caesar's assassination in 44 BC. Cicero called it "
            "magna urbs et gravis, a great and weighty city. The site spreads "
            "across a low hill overlooking the Myzeqe plain and includes the "
            "partially restored Bouleuterion (city council monument, 2nd century "
            "AD), the odeon, the library façade, the Temple of Diana, and "
            "stretches of city wall. The 13th-century Byzantine Monastery of "
            "Saint Mary, built reusing ancient stone, still stands at the centre "
            "of the site and houses the on-site Archaeological Museum. Apollonia "
            "is part of an archaeological park covering around 130 hectares; "
            "only a fraction has been excavated. It is about an hour by car "
            "from Berat and 90 minutes from Tirana. Allow 1.5–2 hours; the site "
            "is shadeless, so avoid midday in summer."
        ),
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
        "description": (
            "Shkodër is the largest city in northern Albania, set between Lake "
            "Shkodër — the largest lake in the Balkans, shared with Montenegro "
            "— and the Buna and Drin rivers. One of the oldest continuously "
            "inhabited cities in Europe, it was the capital of the Illyrian "
            "Ardiaean Kingdom under Queen Teuta in the 3rd century BC and "
            "later an important Ottoman administrative centre. Rozafa Castle, "
            "on a rocky bluff at the junction of the rivers, takes its name "
            "from the Albanian legend of a woman walled into its foundations "
            "and houses a small museum on the city's Illyrian, Roman, "
            "Venetian, and Ottoman layers. Other highlights include the Marubi "
            "National Photography Museum, holding more than 500,000 historical "
            "images from the pioneering Marubi family of photographers (1858 "
            "onward), the Ebu Beker Mosque, and the Shkodër Cathedral. The "
            "pedestrianised Rruga Kolë Idromeno (the Pjaca) runs through the "
            "colourful Italian-style old town. Shkodër is the main staging "
            "point for travel to Theth and Valbona, and Lake Shkodër itself is "
            "a popular cycling, kayaking, and birdwatching destination. The "
            "city is about 95 km north of Tirana on the SH1."
        ),
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
        "description": (
            "Tirana is the capital and largest city of Albania, founded as an "
            "Ottoman bazaar town in 1614 and made the country's capital in "
            "1920. Skanderbeg Square (Sheshi Skënderbej), at the centre, is "
            "one of the largest pedestrian squares in the Balkans and is "
            "bordered by the Et'hem Bey Mosque (1789–1823), the Clock Tower "
            "(1822), the National History Museum with its monumental "
            "\"The Albanians\" mosaic façade, and the Palace of Culture. To "
            "the south, the Pyramid of Tirana — designed by Enver Hoxha's "
            "daughter and son-in-law as a museum to the dictator and "
            "reopened in 2023 as a tech and arts centre — has become a "
            "symbol of the city's transformation. Bunk'Art 1 and Bunk'Art 2 "
            "are Cold War nuclear bunkers converted into museums of "
            "communist-era life and surveillance. The Blloku district, once "
            "sealed off for the communist elite, is now the city's main "
            "nightlife and café area. Mount Dajti, reachable by cable car "
            "(Dajti Ekspres) in about 15 minutes, offers panoramic views and "
            "hiking trails. Tirana International Airport \"Nënë Tereza\" "
            "(TIA) is the country's main air gateway, 17 km northwest of the "
            "centre."
        ),
    },
]


async def seed():
    inserted = 0
    updated = 0
    unchanged = 0
    async with async_session() as db:
        for data in PLACES:
            existing = (
                await db.execute(select(Place).where(Place.name == data["name"]))
            ).scalar_one_or_none()
            if existing is None:
                db.add(Place(**data, photos=[]))
                inserted += 1
            elif existing.description != data["description"]:
                existing.description = data["description"]
                updated += 1
            else:
                unchanged += 1
        await db.commit()
    total = inserted + updated + unchanged
    print(
        f"seed_places: inserted={inserted} updated={updated} "
        f"unchanged={unchanged} total={total}"
    )


if __name__ == "__main__":
    asyncio.run(seed())
