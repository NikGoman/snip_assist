"""
Configuration for the RAG Service.

This module defines the configuration settings for the RAG (Retrieval-Augmented Generation) service,
including paths for Chroma storage, LLM model, and embedding model.
It uses Pydantic's BaseSettings for robust environment variable loading with validation.
"""

import os
from typing import Optional
from pathlib import Path
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

# Load environment variables from .env.local (or .env) if running locally
# In a Docker container, these are typically passed via docker-compose.yml
env_file_path = os.getenv("ENV_FILE_PATH", ".env.local")
if os.path.exists(env_file_path):
    load_dotenv(dotenv_path=env_file_path, override=True)
else:
    # Only warn if .env.local is expected but not found
    if env_file_path == ".env.local":
        logging.warning(f"Environment file {env_file_path} not found, relying on system environment variables")

class RAGConfig(BaseSettings):
    """
    Configuration class for the RAG service.
    Loads settings from environment variables defined in .env.local.
    """
    # --- ChromaDB Configuration ---
    # Path where ChromaDB will store its data. In docker-compose, this will be mapped to a volume.
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # ChromaDB collection name for storing document embeddings
    CHROMA_COLLECTION_NAME: str = "construction_docs"

    # --- LLM Configuration for HuggingFace TransformersLLM ---
    # Model name or path for the HuggingFace LLM (e.g., Phi-3-mini)
    # This path should be accessible within the container or a HuggingFace hub model ID
    HF_LLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"  # Example default

    # Context window size for the LLM (optional, can be inferred from model, but good to set)
    LLM_CONTEXT_WINDOW: int = 4096

    # --- Embedding Model Configuration ---
    # Name or path for the embedding model (e.g., a sentence-transformers model)
    # This can be a Hugging Face model name or a local path
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2)
    # It's good practice to define this if it's static for your model choice
    EMBEDDING_DIM: int = 384

    # --- LlamaIndex Specific Configuration ---
    # Chunk size for splitting documents before indexing
    CHUNK_SIZE: int = 512

    # Chunk overlap for splitting documents
    CHUNK_OVERLAP: int = 50

    # --- Service Configuration ---
    # Port the RAG service will run on inside the container (e.g., used by uvicorn in main.py)
    SERVICE_PORT: int = 8001

    # Host the RAG service will bind to inside the container
    SERVICE_HOST: str = "0.0.0.0"  # Bind to all interfaces in the container

    # Default top_k value for retrieving relevant nodes from the index
    DEFAULT_TOP_K: int = 5

    # --- Advanced LLM Settings (Optional, can be added later) ---
    # Example:
    # LLM_TEMPERATURE: float = 0.1
    # LLM_MAX_NEW_TOKENS: int = 256

    # --- Validation methods ---
    @field_validator('CHROMA_PERSIST_DIR')
    @classmethod
    def validate_chroma_persist_dir(cls, v: str) -> str:
        """Validate that the Chroma persist directory path is safe and absolute when needed."""
        path = Path(v)

        # Prevent directory traversal attacks
        if ".." in v or path.is_absolute() and not str(path).startswith("/app"):
            raise ValueError(f"Unsafe path for CHROMA_PERSIST_DIR: {v}. Path should be relative or within app directory.")

        # Ensure directory exists or can be created
        full_path = Path.cwd() / path
        full_path.mkdir(parents=True, exist_ok=True)

        return v

    @field_validator('SERVICE_PORT')
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate that the service port is within valid range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port {v} is not within valid range (1-65535)")
        return v

    @field_validator('CHUNK_SIZE', 'CHUNK_OVERLAP', 'EMBEDDING_DIM', 'LLM_CONTEXT_WINDOW')
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that integer values are positive."""
        if v <= 0:
            raise ValueError(f"Value {v} must be positive")
        return v

    @field_validator('DEFAULT_TOP_K')
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate that top_k is reasonable."""
        if not (1 <= v <= 20):
            raise ValueError(f"DEFAULT_TOP_K {v} is not within reasonable range (1-20)")
        return v

    @model_validator(mode='after')
    def validate_model_compatibility(self):
        """Validate that model configurations are compatible."""
        # Check that chunk size is greater than overlap
        if self.CHUNK_SIZE <= self.CHUNK_OVERLAP:
            raise ValueError(f"CHUNK_SIZE ({self.CHUNK_SIZE}) must be greater than CHUNK_OVERLAP ({self.CHUNK_OVERLAP})")

        # Check that context window is reasonable for the model
        if self.LLM_CONTEXT_WINDOW <= 0:
            raise ValueError(f"LLM_CONTEXT_WINDOW ({self.LLM_CONTEXT_WINDOW}) must be positive")

        return self

    class Config:
        # By default, Pydantic BaseSettings looks for environment variables
        # This tells it to be case-insensitive when matching env vars
        case_sensitive = False
        # Allow extra fields if defined in .env but not in the model (optional, can be useful)
        # extra = "allow"


# Instantiate the configuration object
# This will load the settings from environment variables or defaults
rag_config = RAGConfig()

# Convenience function to get the config object
def get_rag_config() -> RAGConfig:
    """
    Returns the singleton instance of RAGConfig.

    Returns:
        RAGConfig: The configuration object with validated settings
    """
    return rag_config


# Example usage within rag modules:
# from rag.config import get_rag_config
# config = get_rag_config()
# persist_dir = config.CHROMA_PERSIST_DIR
# model_path = config.HF_LLM_MODEL_NAME
