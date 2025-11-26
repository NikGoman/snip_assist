"""
Query Handler (Updated for Microservice Architecture)

This handler processes text messages received by the bot.
It now relies on the QueryService, which interacts with the external API service,
and assumes limit checking/incrementing is handled by middleware.
"""

from aiogram import Router
from aiogram.types import Message
from app.services.query_service import QueryService
import logging

router = Router()
service = QueryService()

# Configure logging for this module
logger = logging.getLogger(__name__)

@router.message()
async def handle_query(message: Message):
    """
    Handles incoming text messages and processes them using the QueryService.
    Assumes limit checking and incrementing are handled by middleware (e.g., LimitsMiddleware).
    """
    if not message.text:
        await message.answer("Пожалуйста, введите текст вопроса.")
        return

    # Log the incoming query
    logger.info(f"Received query from user_id={message.from_user.id}: {message.text[:50]}...")

    # The LimitsMiddleware (or similar logic) is responsible for checking limits
    # and incrementing the usage counter before this handler runs.
    # Therefore, we proceed directly to processing the query.

    try:
        # Process the query using the updated QueryService
        # This service now calls the API/RAG service via HTTP
        response = await service.process_query(message.text)
    except Exception as e:
        # Log the error with user context
        logger.error(f"Unexpected error in handle_query for user_id={message.from_user.id}: {e}")
        # Send a generic error message to the user
        await message.answer(
            "Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте снова или обратитесь в поддержку."
        )
        return

    # Send the response received from the QueryService (originally from RAG via API)
    # to the user.
    await message.answer(response)
    logger.info(f"Response sent to user_id={message.from_user.id}.")

# Example of how it might look if limits were checked here (not the current approach):
# @router.message()
# async def handle_query_with_inline_limits(message: Message):
#     if not message.text:
#         await message.answer("Пожалуйста, введите текст вопроса.")
#         return
#
#     user_id = str(message.from_user.id)
#     limits_result = await service.check_limits(user_id)
#
#     if not limits_result["allowed"]:
#         await message.answer(limits_result["message"])
#         return
#
#     try:
#         response = await service.process_query(message.text)
#         await message.answer(response)
#         await service.increment_usage(user_id) # Increment AFTER successful response
#     except Exception as e:
#         logger.error(f"Error processing query for user {user_id}: {e}")
#         await message.answer("Произошла ошибка при обработке запроса.")
