from aiogram import Router
from aiogram.types import Message
from app.services.query_service import QueryService
import logging

router = Router()
service = QueryService()

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

@router.message()
async def handle_query(message: Message):
    if not message.text:
        await message.answer("Пожалуйста, введите текст вопроса.")
        return

    # Проверка лимитов и инкремент счётчика теперь происходят в LimitsMiddleware
    # Удаляем вызовы service.check_limits и service.increment_usage

    # Обработка запроса
    try:
        response = await service.process_query(message.text)
    except Exception as e:
        # Логируем ошибку
        logger.error(f"Ошибка при обработке запроса от user_id={message.from_user.id}: {e}")
        # Отправляем пользователю сообщение об ошибке
        await message.answer(
            "Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте снова или обратитесь в поддержку."
        )
        return

    await message.answer(response)
    # Увеличение счётчика теперь происходит в LimitsMiddleware
