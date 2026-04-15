from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import client
from database import User, UserPreferences, init_db, get_session

app = FastAPI(title="Travel Buddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_SYSTEM_PROMPT = """You are Travel Buddy, an AI travel assistant specializing
exclusively in Albania. You have deep knowledge of every city, village, beach,
mountain, restaurant, hotel, and hidden gem across the country.

You help travelers:
- Plan detailed itineraries across Albania
- Discover local restaurants, cafes, and nightlife
- Find beaches, hiking trails, and cultural sites
- Navigate transport (furgon schedules, car rentals, ferries)
- Understand local customs, tipping, and safety
- Get off-the-beaten-path recommendations locals would give

Personality: Warm, enthusiastic about Albania, like a knowledgeable local friend.
Not generic or corporate.

Rules:
- Only recommend real, verified places
- Always include practical details (hours, prices in LEK, location, how to get there)
- Flag seasonal closures or conditions when relevant
- If unsure about something, say so rather than guessing
- Respond in the language the user writes in"""


def build_system_prompt(preferences: UserPreferences | None) -> str:
    if not preferences:
        return BASE_SYSTEM_PROMPT
    return f"""{BASE_SYSTEM_PROMPT}

User preferences — tailor every recommendation to these:
{preferences.to_prompt()}"""


class ChatRequest(BaseModel):
    message: str
    user_id: int | None = None
    conversation_history: list = []


class CreateUserRequest(BaseModel):
    name: str
    email: str


class UpdatePreferencesRequest(BaseModel):
    user_id: int
    budget: str = "mid-range"
    travel_style: list[str] = []
    food: list[str] = []
    pace: str = "moderate"
    interests: list[str] = []
    language: str = "en"


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/users")
async def create_user(request: CreateUserRequest, db: AsyncSession = Depends(get_session)):
    user = User(name=request.name, email=request.email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email}


@app.post("/preferences")
async def update_preferences(
    request: UpdatePreferencesRequest, db: AsyncSession = Depends(get_session)
):
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == request.user_id)
    )
    prefs = result.scalar_one_or_none()

    if prefs:
        prefs.budget = request.budget
        prefs.travel_style = request.travel_style
        prefs.food = request.food
        prefs.pace = request.pace
        prefs.interests = request.interests
        prefs.language = request.language
    else:
        prefs = UserPreferences(
            user_id=request.user_id,
            budget=request.budget,
            travel_style=request.travel_style,
            food=request.food,
            pace=request.pace,
            interests=request.interests,
            language=request.language,
        )
        db.add(prefs)

    await db.commit()
    return {"status": "updated"}


@app.get("/preferences/{user_id}")
async def get_preferences(user_id: int, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user_id)
    )
    prefs = result.scalar_one_or_none()
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")

    return {
        "budget": prefs.budget,
        "travel_style": prefs.travel_style,
        "food": prefs.food,
        "pace": prefs.pace,
        "interests": prefs.interests,
        "language": prefs.language,
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_session)):
    preferences = None
    if request.user_id:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == request.user_id)
        )
        preferences = result.scalar_one_or_none()

    system_prompt = build_system_prompt(preferences)

    messages = request.conversation_history + [
        {"role": "user", "content": request.message}
    ]

    def stream():
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        ) as response:
            for text in response.text_stream:
                yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
