"""
Main Application Entry Point (Updated for Microservice Architecture)

This module initializes and runs the Telegram bot.
It no longer initializes the RAG service directly, as RAG logic is handled by the 'rag' service
via the 'api' service. The bot focuses on Telegram interactions, database, and middleware.
"""

import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from app.core.config import settings
from app.bot.handlers import start, query, limits, help, stats # Import routers
from app.core.database import init_db
# Removed import: from app.core.rag import RAGService
from app.bot.middleware.throttling import ThrottlingMiddleware
from app.bot.middleware.logging import LoggingMiddleware
from app.bot.middleware.limits import LimitsMiddleware

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    # 1. Initialize the database
    await init_db()
    logger.info("Database initialized.")

    # 2. Initialize the bot
    bot = Bot(
        token=settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2)
    )
    dp = Dispatcher()

    # --- Setup Middleware ---
    # Order is important: Logging -> Throttling -> Limits
    dp.message.middleware(LoggingMiddleware())
    dp.message.middleware(ThrottlingMiddleware(default_ttl=1.0))
    dp.message.middleware(LimitsMiddleware())

    # --- Include Routers ---
    dp.include_router(start.router)
    dp.include_router(query.router)
    dp.include_router(limits.router)
    dp.include_router(help.router)
    dp.include_router(stats.router) # Optionally include if admin stats are needed

    # --- Removed RAG Initialization ---
    # RAG service is now external. The bot relies on the 'api' service for queries.
    # The startup check for RAG is no longer performed here.
    # It's the responsibility of the 'api' and 'rag' services to be ready.
    # Optionally, you could add a health check to the 'api' service on startup here,
    # but it's not strictly necessary if the 'api' service handles retries/failures gracefully.
    # For now, we assume the 'api' service is available when the bot starts.
    logger.info("Bot setup complete. Assuming API service is available.")

    # --- Start Polling ---
    try:
        logger.info("Starting Telegram bot polling...")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping...")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise # Re-raise to let the container/service manager handle it
    finally:
        await bot.session.close()
        logger.info("Bot session closed.")


if __name__ == "__main__":
    asyncio.run(main())
