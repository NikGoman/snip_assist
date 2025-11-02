import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    FREE_QUERIES_PER_DAY: int = int(os.getenv("FREE_QUERIES_PER_DAY", 3))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", 500))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/embeddings")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Phi-3-mini-4k-instruct")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./users.db")
    # --- Новое поле для ID админов ---
    ADMIN_USER_IDS: str = os.getenv("ADMIN_USER_IDS", "123456789")
    # --- /Новое поле ---

settings = Settings()
