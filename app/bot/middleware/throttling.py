import time
from typing import Any, Awaitable, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject


class ThrottlingMiddleware(BaseMiddleware):
    """
    Промежуточный слой для ограничения частоты запросов от одного пользователя.
    Предотвращает флуд и чрезмерную нагрузку на бота и RAG-сервис.
    """

    def __init__(self, default_ttl: float = 1.0):
        """
        :param default_ttl: Время в секундах, в течение которого блокируются повторные сообщения от одного пользователя.
        """
        super().__init__()
        self.default_ttl = default_ttl
        # Словарь для хранения времени последнего сообщения от каждого пользователя
        # Ключ: user_id, Значение: timestamp
        self.last_message_time: Dict[int, float] = {}

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """
        Основной метод middleware.
        Проверяет, прошло ли достаточно времени с момента последнего сообщения от пользователя.
        Если нет — обработка сообщения прерывается (возврат None).
        Если да — обновляется время и вызывается следующий обработчик.
        """
        if isinstance(event, Message):
            user_id = event.from_user.id
            current_time = time.time()

            last_time = self.last_message_time.get(user_id)
            if last_time and (current_time - last_time) < self.default_ttl:
                # Пользователь отправляет сообщения слишком часто.
                # Возвращаем None, чтобы остановить обработку сообщения.
                # В реальных сценариях можно отправить пользователю предупреждение.
                return

            # Обновляем время последнего сообщения
            self.last_message_time[user_id] = current_time

        # Передаём управление следующему обработчику
        return await handler(event, data)