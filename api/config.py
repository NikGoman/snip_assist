"""
Configuration for the API Service.

This module defines the configuration settings for the API service,
including the URL of the RAG service, and service-specific host/port.
It uses Pydantic's BaseSettings for robust environment variable loading.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env.local (or .env) if running locally
# In a Docker container, these are typically passed via docker-compose.yml
load_dotenv(dotenv_path=os.getenv("ENV_FILE_PATH", ".env.local"), override=True)

class APIConfig(BaseSettings):
    """
    Configuration class for the API service.
    Loads settings from environment variables defined in .env.local.
    """
    # --- RAG Service Configuration ---
    # URL where the RAG service is accessible from the API service.
    # In docker-compose, this will typically be the service name and port, e.g., "http://rag:8001"
    RAG_SERVICE_URL: str = "http://rag:8001" # Default assumes docker-compose service name 'rag'

    # --- Service Configuration ---
    # Port the API service will run on inside the container (e.g., used by uvicorn in main.py)
    SERVICE_PORT: int = 8000

    # Host the API service will bind to inside the container
    SERVICE_HOST: str = "0.0.0.0" # Bind to all interfaces in the container

    # --- Advanced HTTP Client Settings for RAG (Optional) ---
    # Timeout for requests to the RAG service (in seconds)
    RAG_REQUEST_TIMEOUT: float = 30.0

    # Example for future use:
    # RAG_API_KEY: Optional[str] = None # If RAG service requires auth

    class Config:
        # By default, Pydantic BaseSettings looks for environment variables
        # This tells it to be case-insensitive when matching env vars
        case_sensitive = False
        # Allow extra fields if defined in .env but not in the model (optional, can be useful)
        # extra = "allow"

# Instantiate the configuration object
# This will load the settings from environment variables or defaults
api_config = APIConfig()

# Convenience function to get the config object
def get_api_config() -> APIConfig:
    """
    Returns the singleton instance of APIConfig.
    """
    return api_config

# Example usage within api modules:
# from api.config import get_api_config
# config = get_api_config()
# rag_url = config.RAG_SERVICE_URL
# port = config.SERVICE_PORT
