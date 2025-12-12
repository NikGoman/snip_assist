import logging
from typing import Any, Awaitable, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject, User
from app.core.config import settings

# Настраиваем логгер для middleware
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Простой обработчик в stdout, можно заменить на RotatingFileHandler или другой
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class LoggingMiddleware(BaseMiddleware):
    """
    Промежуточный слой для структурированного логирования событий:
    - Входящие сообщения
    - Выходящие ответы (опционально, через callback)
    - Ошибки (опционально, через callback)
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """
        Основной метод middleware.
        Логирует событие перед вызовом обработчика.
        """
        # Проверяем, что событие — это сообщение
        if isinstance(event, Message):
            user: User = event.from_user
            chat_id = event.chat.id
            # Безопасно получаем текст сообщения, проверяя на None
            text = event.text or (event.caption if hasattr(event, 'caption') else '<no_text>')

            # Безопасно обрезаем текст, если он не None
            text_preview = f"'{text[:50]}...'" if text else "'<no_text>'"

            logger.info(
                f"Получено сообщение от user_id={user.id} (@{user.username or 'N/A'}) "
                f"в чате chat_id={chat_id}. Текст: {text_preview}"
            )

        # Вызываем следующий обработчик
        try:
            result = await handler(event, data)
            # Здесь можно логировать успешный результат, если нужно
            # logger.debug(f"Обработчик завершён успешно для user_id={user.id}")
            return result
        except Exception as e:
            # Логируем ошибку, если она произошла в обработчике
            # Безопасно получаем user_id, если user существует
            user_id = getattr(data.get('event_from_user'), 'id', 'unknown') if data.get('event_from_user') else 'unknown'
            logger.error(f"Ошибка в обработчике для user_id={user_id}: {e}")
            # Передаём ошибку дальше, чтобы другие обработчики могли её обработать
            raise
