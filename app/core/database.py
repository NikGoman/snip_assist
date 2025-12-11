from sqlalchemy import Column, Integer, String, DateTime, Boolean, Date, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, date
from app.core.config import settings

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True)
    first_query_date = Column(DateTime, default=datetime.utcnow)
    queries_used_today = Column(Integer, default=0)
    last_query_date = Column(Date, default=date.today)
    # --- Новые поля для статистики и подписки ---
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    pro_active = Column(Boolean, default=False)
    # --- /Новые поля ---

async def get_async_session():
    async with async_session() as session:
        yield session

engine = create_async_engine(settings.DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
