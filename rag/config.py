"""
Configuration for the RAG Service.

This module defines the configuration settings for the RAG (Retrieval-Augmented Generation) service,
including paths for Chroma storage, LLM model, and embedding model.
It uses Pydantic's BaseSettings for robust environment variable loading.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env.local (or .env) if running locally
# In a Docker container, these are typically passed via docker-compose.yml
load_dotenv(dotenv_path=os.getenv("ENV_FILE_PATH", ".env.local"), override=True)

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
    HF_LLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct" # Example default

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
    SERVICE_HOST: str = "0.0.0.0" # Bind to all interfaces in the container

    # Default top_k value for retrieving relevant nodes from the index
    DEFAULT_TOP_K: int = 5

    # --- Advanced LLM Settings (Optional, can be added later) ---
    # Example:
    # LLM_TEMPERATURE: float = 0.1
    # LLM_MAX_NEW_TOKENS: int = 256

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
    """
    return rag_config

# Example usage within rag modules:
# from rag.config import get_rag_config
# config = get_rag_config()
# persist_dir = config.CHROMA_PERSIST_DIR
# model_path = config.HF_LLM_MODEL_NAME
