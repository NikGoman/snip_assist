"""
Configuration for the Bot Service (app/).

This module defines the configuration settings for the Telegram bot service,
including API endpoints, limits, and database connection.
It uses environment variables loaded via python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # --- Telegram Bot Configuration ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")

    # --- Query Limits and Validation ---
    FREE_QUERIES_PER_DAY: int = int(os.getenv("FREE_QUERIES_PER_DAY", 300))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", 500))

    # --- Database Configuration ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./users.db")

    # --- API Service Configuration (New) ---
    # URL where the API service is accessible from the bot service.
    # In docker-compose, this will typically be the service name and port, e.g., "http://api:8000"
    API_SERVICE_URL: str = os.getenv("API_SERVICE_URL", "http://api:8000") # Default assumes docker-compose service name 'api'

    # --- Admin Configuration ---
    # Comma-separated string of admin user IDs
    ADMIN_USER_IDS: str = os.getenv("ADMIN_USER_IDS", "123456789")

    # --- Removed RAG-specific configurations ---
    # These are now handled by the 'rag' service's config (rag/config.py)
    # CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/embeddings")
    # MODEL_NAME: str = os.getenv("MODEL_NAME", "Phi-3-mini-4k-instruct")


settings = Settings()
