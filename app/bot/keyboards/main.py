from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


def get_main_keyboard() -> InlineKeyboardMarkup:
    """
    Возвращает основную инлайн-клавиатуру для бота.
    Предоставляет пользователю быстрые ссылки на справку, проверку лимита и другие действия.
    """
    builder = InlineKeyboardBuilder()

    # Кнопка "Справка"
    builder.add(InlineKeyboardButton(text="ℹ️ Справка", callback_data="help"))

    # Кнопка "Мой лимит"
    builder.add(InlineKeyboardButton(text="📊 Мой лимит", callback_data="my_limit"))

    # Кнопка "Поддержка"
    builder.add(InlineKeyboardButton(text="🤝 Поддержка", callback_data="support"))

    # Кнопка "Подписка"
    builder.add(InlineKeyboardButton(text="💳 Оформить подписку", callback_data="subscribe"))

    # Кнопка "О боте"
    builder.add(InlineKeyboardButton(text="🌌 О боте", callback_data="about"))

    # Располагаем кнопки в 2 столбца
    builder.adjust(2)

    return builder.as_markup()