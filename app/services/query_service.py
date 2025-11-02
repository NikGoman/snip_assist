from app.core.rag import RAGService
from app.core.database import User, async_session
from sqlalchemy import select
from datetime import date, timedelta
from app.core.config import settings

class QueryService:
    def __init__(self):
        self.rag_service = RAGService()

    async def check_limits(self, telegram_id: str) -> dict:
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                user = User(telegram_id=telegram_id)
                session.add(user)
                await session.commit()

            today = date.today()
            if user.last_query_date != today:
                user.last_query_date = today
                user.queries_used_today = 0
                await session.commit()

            if user.queries_used_today >= settings.FREE_QUERIES_PER_DAY:
                return {"allowed": False, "message": f"Лимит запросов на сегодня исчерпан. Максимум: {settings.FREE_QUERIES_PER_DAY}/день."}

            return {"allowed": True, "message": ""}

    async def increment_usage(self, telegram_id: str):
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.queries_used_today += 1
                await session.commit()

    async def process_query(self, user_query: str) -> str:
        if len(user_query) > settings.MAX_QUERY_LENGTH:
            return f"Запрос слишком длинный. Максимум: {settings.MAX_QUERY_LENGTH} символов."

        response = await self.rag_service.query(user_query)
        return response