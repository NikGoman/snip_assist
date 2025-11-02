import pytest
from aiogram import types
from app.bot.handlers.query import handle_query

@pytest.mark.asyncio
async def test_handle_query(mocker, query_service):
    # Мокаем сервис
    mocker.patch("app.bot.handlers.query.service", query_service)

    message = types.Message(
        message_id=1,
        from_user=types.User(id=123456, is_bot=False, first_name="Test"),
        chat=types.Chat(id=123456, type="private"),
        text="Какой ГОСТ?"
    )

    # Мокаем методы
    query_service.check_limits = AsyncMock(return_value={"allowed": True, "message": ""})
    query_service.process_query = AsyncMock(return_value="Mocked response")
    query_service.increment_usage = AsyncMock()

    # Мокаем вызов answer
    message.answer = AsyncMock()

    await handle_query(message)

    message.answer.assert_called_once_with("Mocked response")