import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from app.core.config import settings
from app.bot.handlers import start, query, limits, help, stats
from app.core.database import init_db
from app.core.rag import RAGService
from app.bot.middleware.throttling import ThrottlingMiddleware
from app.bot.middleware.logging import LoggingMiddleware
from app.bot.middleware.limits import LimitsMiddleware

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    # Инициализация базы данных
    await init_db()

    # Инициализация бота
    bot = Bot(
        token=settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2)
    )
    dp = Dispatcher()

    # --- Интеграция Middleware ---
    # Порядок важен: Logging -> Throttling -> Limits
    dp.message.middleware(LoggingMiddleware())
    dp.message.middleware(ThrottlingMiddleware(default_ttl=1.0))
    dp.message.middleware(LimitsMiddleware())

    # --- Регистрация роутеров ---
    dp.include_router(start.router)
    dp.include_router(query.router)
    dp.include_router(limits.router)
    dp.include_router(help.router)
    dp.include_router(stats.router) # Опционально, если нужна статистика админам

    # --- Инициализация RAG ---
    try:
        logger.info("Инициализация RAG-сервиса...")
        # Попытка инициализировать RAGService (проверит индекс)
        rag_service = RAGService()
        logger.info("RAG-сервис инициализирован успешно.")
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации RAG: {e}")
        raise # Прерываем запуск, если RAG не работает

    # --- Запуск поллинга ---
    try:
        logger.info("Запуск Telegram-бота...")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки. Завершение работы...")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в main: {e}")
    finally:
        await bot.session.close()
        logger.info("Сессия бота закрыта.")


if __name__ == "__main__":
    asyncio.run(main())
