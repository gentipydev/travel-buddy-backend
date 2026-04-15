from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

from config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    preferences = relationship("UserPreferences", back_populates="user", uselist=False)


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    budget = Column(String, default="mid-range")
    travel_style = Column(JSON, default=list)
    food = Column(JSON, default=list)
    pace = Column(String, default="moderate")
    interests = Column(JSON, default=list)
    language = Column(String, default="en")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="preferences")

    def to_prompt(self) -> str:
        parts = []
        parts.append(f"- Budget: {self.budget}")
        if self.travel_style:
            parts.append(f"- Travel style: {', '.join(self.travel_style)}")
        if self.food:
            parts.append(f"- Food preferences: {', '.join(self.food)}")
        parts.append(f"- Pace: {self.pace}")
        if self.interests:
            parts.append(f"- Interests: {', '.join(self.interests)}")
        parts.append(f"- Preferred language: {self.language}")
        return "\n".join(parts)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
