import logging
from typing import Any, Awaitable, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject
from app.services.query_service import QueryService
from app.core.config import settings

logger = logging.getLogger(__name__)

class LimitsMiddleware(BaseMiddleware):
    """
    Промежуточный слой для проверки лимитов запросов пользователя.
    Использует тот же QueryService, что и основной обработчик запросов.
    Если лимит превышен, отправляет сообщение пользователю и прерывает цепочку обработки.
    """

    def __init__(self):
        super().__init__()
        self.query_service = QueryService()

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """
        Основной метод middleware.
        Проверяет лимиты для текстовых сообщений.
        Если лимит не превышен, увеличивает счётчик и передаёт управление дальше.
        """
        # Проверяем, что событие — это сообщение
        if isinstance(event, Message) and event.text:
            user_id = str(event.from_user.id)

            # Проверяем лимиты через QueryService
            limit_check = await self.query_service.check_limits(user_id)

            if not limit_check["allowed"]:
                # Лимит превышен. Отправляем сообщение пользователю.
                await event.answer(limit_check["message"])
                # Прерываем цепочку обработки, возвращая None.
                return

            # Лимит не превышен. Увеличиваем счётчик.
            await self.query_service.increment_usage(user_id)

        # Передаём управление следующему обработчику
        return await handler(event, data)
