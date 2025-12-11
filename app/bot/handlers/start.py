from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()

@router.message(Command("start"))
async def cmd_start(message: Message):
    # Используем HTML режим для корректного отображения специальных символов
    html_text = (
        "Привет! Это бот-ассистент прораба👷🏻\n\n"
        "Он поможет тебе с документами по строительству и ГОСТам.\n\n"
        "Отправь мне любой вопрос, и я постараюсь найти ответ в базе знаний!"
    )
    await message.answer(html_text, parse_mode="HTML")
