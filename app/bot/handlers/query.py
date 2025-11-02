from aiogram import Router
from aiogram.types import Message
from app.services.query_service import QueryService

router = Router()
service = QueryService()

@router.message()
async def handle_query(message: Message):
    if not message.text:
        await message.answer("Пожалуйста, введите текст вопроса.")
        return

    user_id = str(message.from_user.id)

    # Проверка лимитов
    limit_check = await service.check_limits(user_id)
    if not limit_check["allowed"]:
        await message.answer(limit_check["message"])
        return

    # Обработка запроса
    response = await service.process_query(message.text)

    await message.answer(response)

    # Увеличение счётчика
    await service.increment_usage(user_id)