from aiogram import Router
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from app.core.database import async_session, User
from sqlalchemy import select, func
from datetime import date, timedelta
from app.core.config import settings

router = Router()

# В реальном приложении список администраторов должен быть в настройках
ADMIN_USER_IDS = [123456789]  # Заменить на реальные ID администраторов

async def get_stats():
    """
    Внутренняя функция для получения статистики из базы данных.
    """
    async with async_session() as session:
        # Общее количество пользователей
        total_users_result = await session.execute(select(func.count(User.id)))
        total_users = total_users_result.scalar()

        # Количество активных пользователей за сегодня
        today_start = date.today()
        active_today_result = await session.execute(
            select(func.count(User.id)).where(User.last_active >= today_start)
        )
        active_today = active_today_result.scalar()

        # Количество активных пользователей за последние 7 дней
        week_start = date.today() - timedelta(days=7)
        active_week_result = await session.execute(
            select(func.count(User.id)).where(User.last_active >= week_start)
        )
        active_week = active_week_result.scalar()

        # Количество пользователей с активной подпиской (упрощённо)
        # В реальном приложении может быть отдельное поле или таблица
        pro_users_result = await session.execute(
            select(func.count(User.id)).where(User.pro_active == True)
        )
        pro_users = pro_users_result.scalar()

    return {
        "total_users": total_users,
        "active_today": active_today,
        "active_week": active_week,
        "pro_users": pro_users,
    }


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    """
    Обработчик команды /stats.
    Показывает статистику по боту. Доступно только администраторам.
    """
    if message.from_user.id not in ADMIN_USER_IDS:
        await message.answer("❌ У вас нет прав для просмотра статистики.")
        return

    stats = await get_stats()

    response_text = (
        "📊 **Статистика бота**\n\n"
        f"Всего пользователей: {stats['total_users']}\n"
        f"Активно сегодня: {stats['active_today']}\n"
        f"Активно за 7 дней: {stats['active_week']}\n"
        f"Подписчиков Pro: {stats['pro_users']}\n"
    )
    await message.answer(response_text, parse_mode="Markdown")


@router.callback_query(lambda c: c.data == "stats")
async def callback_stats(callback_query: CallbackQuery):
    """
    Callback-обработчик для кнопки 'Статистика'.
    Вызывает ту же логику, что и команда /stats.
    """
    if callback_query.from_user.id not in ADMIN_USER_IDS:
        await callback_query.answer("❌ У вас нет прав.", show_alert=True)
        return

    stats = await get_stats()

    response_text = (
        "📊 **Статистика бота**\n\n"
        f"Всего пользователей: {stats['total_users']}\n"
        f"Активно сегодня: {stats['active_today']}\n"
        f"Активно за 7 дней: {stats['active_week']}\n"
        f"Подписчиков Pro: {stats['pro_users']}\n"
    )
    await callback_query.message.edit_text(response_text, parse_mode="Markdown")
    await callback_query.answer()