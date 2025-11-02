from aiogram import Router
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from app.services.query_service import QueryService

router = Router()
service = QueryService()

@router.message(Command("limit"))
async def cmd_limit(message: Message):
    """
    Обработчик команды /limit.
    Показывает пользователю, сколько запросов он использовал сегодня и сколько осталось.
    """
    user_id = str(message.from_user.id)

    # Получаем текущий статус лимитов
    limit_check = await service.check_limits(user_id)

    # Так как мы внутри /limit, мы не увеличиваем счётчик
    # Проверим, был ли пользователь в базе, и сколько у него осталось
    async with service.rag_service.storage_context.db_session() as session:
        from app.core.database import User
        from sqlalchemy import select
        from datetime import date
        result = await session.execute(
            select(User).where(User.telegram_id == user_id)
        )
        user_db = result.scalar_one_or_none()

    if not user_db:
        await message.answer("Вы ещё не использовали ни одного запроса.")
        return

    used_today = user_db.queries_used_today
    max_free = settings.FREE_QUERIES_PER_DAY # Импортируем settings из config

    remaining = max(0, max_free - used_today)

    response_text = (
        f"📊 *Статус лимита*\n\n"
        f"Использовано сегодня: {used_today} / {max_free}\n"
        f"Осталось: {remaining}\n\n"
        f"💡 Подписка за $29/мес откроет неограниченный доступ."
    )
    await message.answer(response_text, parse_mode="Markdown")


@router.callback_query(lambda c: c.data == "my_limit")
async def callback_limit(callback_query: CallbackQuery):
    """
    Callback-обработчик для кнопки 'Мой лимит'.
    Вызывает ту же логику, что и команда /limit.
    """
    # Создаём фейковое сообщение для совместимости с cmd_limit
    fake_message = Message(
        message_id=callback_query.message.message_id,
        from_user=callback_query.from_user,
        chat=callback_query.message.chat,
        text="/limit",
        date=callback_query.message.date,
        bot=callback_query.bot
    )
    await cmd_limit(fake_message)
    await callback_query.answer()