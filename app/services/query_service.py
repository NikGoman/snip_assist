"""
Query Service (Updated for Microservice Architecture)

This service handles the business logic for processing user queries.
It now interacts with the external API service for RAG functionality
and manages query limits using the local database.
"""

import httpx
from typing import Dict, Any
import logging

from app.core.database import User, async_session
from sqlalchemy import select
from datetime import date
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self):
        # Removed direct RAGService instantiation
        # Initialize HTTP client for API service
        # Using a single client instance or creating one per request are options.
        # For simplicity in this example, we'll create it per request using httpx directly.
        # If performance is critical, consider dependency injection or a managed client instance.
        self.api_base_url = settings.API_SERVICE_URL # Assumes this is added to config
        self.max_query_length = settings.MAX_QUERY_LENGTH
        self.free_queries_per_day = settings.FREE_QUERIES_PER_DAY

    async def check_limits(self, telegram_id: str) -> Dict[str, Any]:
        """
        Checks if the user is allowed to make a query based on daily limits.
        Updates the user's record if it's a new day.
        """
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

            if user.queries_used_today >= self.free_queries_per_day:
                return {
                    "allowed": False,
                    "message": f"Лимит запросов на сегодня исчерпан. Максимум: {self.free_queries_per_day}/день."
                }

            return {"allowed": True, "message": ""}

    async def increment_usage(self, telegram_id: str):
        """
        Increments the user's query count for the current day.
        """
        async with async_session() as session:
            result = await session.execute(
                select(User).where(User.telegram_id == telegram_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.queries_used_today += 1
                await session.commit()
            else:
                # This should ideally not happen if check_limits is called first
                logger.warning(f"Attempted to increment usage for unknown user: {telegram_id}")

    async def process_query(self, user_query: str) -> str:
        """
        Processes the user's query by checking limits and forwarding to the API service.
        """
        if len(user_query) > self.max_query_length:
            return f"Запрос слишком длинный. Максимум: {self.max_query_length} символов."

        # Note: In the new architecture, the bot service (app/) handles the limit check
        # before calling this method, or the API service handles it.
        # For this example, we'll assume the check happens *before* process_query is called
        # by the bot handler, or we could call check_limits here if it remains the bot's responsibility.
        # Let's assume the responsibility stays with the bot handler calling this service.
        # So, we proceed directly to the API call.

        try:
            # Prepare the payload for the API service
            payload = {
                "query_text": user_query,
                # Optionally pass top_k if the bot handler wants to control it,
                # otherwise, let the API/RAG service use its default.
                # "top_k": top_k_value
            }

            # Make an asynchronous request to the API service
            # Using httpx.AsyncClient context manager for safety
            async with httpx.AsyncClient(base_url=self.api_base_url, timeout=30.0) as client:
                response = await client.post("/query", json=payload)

            # Check the response status from the API service
            if response.status_code != 200:
                logger.error(f"API service returned status {response.status_code}: {response.text}")
                return f"Ошибка при обработке запроса: API service responded with status {response.status_code}."

            # Parse the JSON response from the API service
            api_response_data = response.json()

            # The API service should return a structure compatible with QueryResponse from api/main.py
            # which mirrors the RAG service's response.
            # Extract the response text. The structure is {"response_text": "...", "source_nodes": [...]}
            response_text = api_response_data.get("response_text", "Получен пустой ответ от системы.")
            # source_nodes = api_response_data.get("source_nodes", []) # Could be used for logging or detailed responses if needed

            logger.info("Query processed successfully via API service.")
            return response_text

        except httpx.RequestError as e:
            logger.error(f"Error making request to API service: {e}")
            return "Ошибка при подключении к сервису поиска. Пожалуйста, попробуйте позже."
        except httpx.HTTPStatusError as e:
            logger.error(f"API service responded with an error status: {e}")
            return f"Сервис поиска вернул ошибку: {e.response.status_code}."
        except KeyError as e:
            # Handle case where expected keys are missing in the API response
            logger.error(f"Malformed response from API service: missing key {e}")
            return "Ошибка: получен некорректный ответ от системы поиска."
        except Exception as e:
            logger.error(f"Unexpected error in process_query while calling API: {e}")
            return "Произошла непредвиденная ошибка при обработке запроса."

    # Optional helper method to encapsulate the API call if used in multiple places
    # async def _call_api_service(self, query_text: str, top_k: Optional[int] = None) -> Optional[Dict[str, Any]]:
    #     payload = {"query_text": query_text}
    #     if top_k is not None:
    #         payload["top_k"] = top_k
    #     try:
    #         async with httpx.AsyncClient(base_url=self.api_base_url, timeout=30.0) as client:
    #             response = await client.post("/query", json=payload)
    #             response.raise_for_status() # Raises an HTTPStatusError for 4xx/5xx responses
    #             return response.json()
    #     except httpx.RequestError as e:
    #         logger.error(f"Request error: {e}")
    #         return None
    #     except httpx.HTTPStatusError as e:
    #         logger.error(f"HTTP error from API: {e}")
    #         return None
