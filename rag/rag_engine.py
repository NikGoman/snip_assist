"""
RAG Engine Module (Updated for llama-index-core >= 0.13.0)

This module encapsulates the core logic for the Retrieval-Augmented Generation (RAG) system
using LlamaIndex (core + integrations) and Chroma. It handles document loading, indexing,
storage, and querying. This version is adapted to use HuggingFace models and torch,
and provides an asynchronous query method, aligning more closely with the original rag.py.
"""

import logging
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
# ИМПОРТ ИЗМЕНЕН: Теперь используем интеграционный пакет llama-index-llms-huggingface
from llama_index.llms.huggingface import HuggingFaceLLM as TransformersLLM
# ИМПОРТ ИЗМЕНЕН: В новых версиях llama-index интеграции с HuggingFace находятся в отдельных пакетах
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    # Альтернативный импорт для новых версий llama-index
    from llama_index.core.embeddings import HuggingFaceEmbedding
# ИМПОРТ ИЗМЕНЕН: Теперь используем интеграционный пакет llama-index-vector-stores-chroma
from llama_index.vector_stores.chroma import ChromaVectorStore

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import chromadb
from pydantic import BaseModel, Field

from rag.config import get_rag_config

logger = logging.getLogger(__name__)
config = get_rag_config()

# --- CRITICAL: Update rag/config.py ---
# Add the following to rag/config.py:
# # Model name for HuggingFace TransformersLLM
# HF_LLM_MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct" # Example default
try:
    HF_MODEL_NAME = config.HF_LLM_MODEL_NAME  # This key must exist in rag/config.py
except AttributeError:
    logger.error("Configuration key 'HF_LLM_MODEL_NAME' not found in rag/config.py. Please add it.")
    raise AttributeError("Configuration key 'HF_LLM_MODEL_NAME' is missing in rag/config.py. Please add it (e.g., HF_LLM_MODEL_NAME: str = 'microsoft/Phi-3-mini-4k-instruct').")

class QueryResponse(BaseModel):
    """Model representing the response from the RAG query."""
    response_text: str = Field(description="The generated response text.")
    source_nodes: List[dict] = Field(description="List of source nodes with metadata.")


class RAGEngine:
    """
    Encapsulates the RAG logic: loading/indexing documents and querying them.
    Uses LlamaIndex abstractions for modularity and flexibility.
    Updated to use TransformersLLM and provide async query.
    Compatible with llama-index-core >= 0.13.0 and integration packages.
    """

    def __init__(self):
        """
        Initializes the RAGEngine by setting up the LLM, embeddings, and vector store.
        Does not load the index yet; call load_index or rebuild_index.
        """
        self._index = None
        self._query_engine = None
        self._client = None
        self._vector_store = None

        # --- 1. Setup Embeddings (HuggingFace) ---
        try:
            self._embed_model = HuggingFaceEmbedding(
                model_name=config.EMBEDDING_MODEL_NAME,
                # trust_remote_code=True # Uncomment if using a custom or untrusted model
            )
            logger.info(f"Embedding model loaded: {config.EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {config.EMBEDDING_MODEL_NAME}: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")

        # --- 2. Setup LLM (Transformers with HuggingFace Model and Torch) ---
        try:
            logger.info(f"Loading HuggingFace model: {HF_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",  # Automatically assign layers to available devices (CPU/GPU)
                trust_remote_code=True,
                # pad_token_id=tokenizer.eos_token_id, # Often required for generation
            )

            # Wrap the HuggingFace model with LlamaIndex's TransformersLLM
            self._llm = TransformersLLM(
                model=model,
                tokenizer=tokenizer,
                context_window=config.LLM_CONTEXT_WINDOW,  # Use the context window from config
                max_new_tokens=512,  # Keep original value from rag.py
                generate_kwargs={"temperature": 0.1},  # Keep original value from rag.py
                device_map="auto"  # Pass device_map to LlamaIndex wrapper if needed, often handled by model
            )
            logger.info(f"LLM loaded from HuggingFace: {HF_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace LLM from {HF_MODEL_NAME}: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")

        # --- 3. Setup Vector Store (Chroma) ---
        try:
            self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            self._chroma_collection = self._client.get_or_create_collection(
                config.CHROMA_COLLECTION_NAME
            )
            # ИМПОРТ ПЕРЕМЕЩЕН: Импорт ChromaVectorStore теперь на уровне модуля, т.к. он из интеграционного пакета
            self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
            logger.info(f"Chroma vector store initialized at {config.CHROMA_PERSIST_DIR}, collection: {config.CHROMA_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}")

        # --- 4. Configure LlamaIndex Settings ---
        # This ensures that the LLM and Embed Model are used globally by LlamaIndex components
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model
        # Settings.node_parser = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        # ^ Uncomment if you want to set a global default parser for LlamaIndex operations.
        # For loading an existing index from Chroma, the global parser might not be relevant.
        # It's more relevant when creating an index from documents using from_documents().

    def _build_query_engine(self, top_k: int = 3, response_mode: str = "tree_summarize"):
        """
        Builds the query engine from the loaded index.
        This replicates the logic from the original rag.py's query method.
        Should be called after load_index or rebuild_index.
        """
        if self._index is None:
            raise ValueError("Index is not loaded. Call load_index or rebuild_index first.")

        # Use the `as_query_engine` method as in the original rag.py
        # This method encapsulates the retriever and response synthesis.
        self._query_engine = self._index.as_query_engine(
            llm=self._llm,
            similarity_top_k=top_k,  # Use provided top_k, defaulting to 3 as in original
            response_mode=response_mode,  # Use provided response_mode, defaulting as in original
            # Additional parameters can be passed here if needed
        )
        logger.info("Query engine built successfully using as_query_engine.")

    def load_index(self, data_dir: Optional[str] = None):
        """
        Loads an existing index from the Chroma vector store.
        Replicates the behavior of the original rag.py's __init__ -> load_index call.
        """
        logger.info("Loading index from Chroma vector store...")
        try:
            # Create the index object connected to the Chroma vector store
            # This is the key call from the original rag.py
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                show_progress=True  # Keep original behavior
            )

            # Build the query engine with default parameters from the original rag.py
            # similarity_top_k=3, response_mode="tree_summarize"
            self._build_query_engine(top_k=3, response_mode="tree_summarize")

            logger.info("Index loaded and query engine built successfully.")
        except Exception as e:
            logger.error(f"Failed to load index from vector store: {e}")
            # Replicate original behavior: set index to None on failure
            self._index = None
            # Depending on requirements, you might still raise or just log.
            # The original rag.py allows self.index = None and checks it in query.
            # For a service, raising might be preferred, but the caller (e.g., main.py) should handle it.
            # Let's raise to prevent a broken state, but the caller should handle it.
            raise RuntimeError(f"Index loading failed: {e}") from e

    def rebuild_index(self, data_dir: str):
        """
        Rebuilds the index from documents in the specified directory and persists it to Chroma.
        Also configures the node parser as part of the process.
        """
        logger.info(f"Rebuilding index from documents in: {data_dir}")
        try:
            # Configure the parser for document processing
            parser = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

            # 1. Load documents
            logger.info("Loading documents...")
            documents = SimpleDirectoryReader(
                input_dir=data_dir,
                recursive=True,
                required_exts=[".pdf", ".docx", ".txt", ".md"]
            ).load_data()
            logger.info(f"Loaded {len(documents)} documents.")

            # 2. Create the index from documents, connected to the Chroma vector store
            # This time, we *do* use the parser to chunk the documents
            logger.info("Building index and storing vectors in Chroma...")
            self._index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=None,  # Use default, which connects to vector_store
                vector_store=self._vector_store,
                embed_model=self._embed_model,
                show_progress=True,
                transformations=[parser]  # Apply the sentence splitter
            )

            # 3. Build the query engine for the newly created index
            # Use defaults as per original rag.py
            self._build_query_engine(top_k=3, response_mode="tree_summarize")
            logger.info("Index rebuilt and persisted to Chroma successfully.")

        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise RuntimeError(f"Index rebuilding failed: {e}") from e

    async def aquery(self, query_text: str, top_k: Optional[int] = 3) -> QueryResponse:
        """
        Asynchronously queries the RAG system with the given text.
        Replicates the async behavior and response formatting of the original rag.py's query method.

        Args:
            query_text (str): The user's query.
            top_k (Optional[int]): Number of top results to retrieve. Defaults to 3 as in original.

        Returns:
            QueryResponse: The generated response and source nodes.
                         The response_text will include formatted sources, matching original rag.py.
        """
        if self._query_engine is None or self._index is None:
            # Replicate original check
            error_msg = "Ошибка: база данных нормативов недоступна. Пожалуйста, свяжитесь с администратором."
            logger.error("Query engine or index is not loaded.")
            return QueryResponse(response_text=error_msg, source_nodes=[])

        if top_k is None:
            top_k = 3  # Default as per original

        logger.info(f"Processing query (top_k={top_k}): {query_text[:50]}...")

        try:
            # Perform the query using the LlamaIndex query engine (as_query_engine)
            # This call is synchronous internally within LlamaIndex, but we are in an async method
            response_obj = self._query_engine.query(query_text)

            # --- Replicate Source Formatting from Original rag.py ---
            source_nodes_info = []
            sources_lines = []
            for node_with_score in response_obj.source_nodes[:2]:  # Take top 2 as per original
                node = node_with_score.node
                metadata = node.metadata
                doc_title = metadata.get("file_name", "Документ")  # Use 'file_name' as in original
                page = metadata.get("page_label", "N/A")          # Use 'page_label' as in original
                sources_lines.append(f"- {doc_title} (стр. {page})")

                # Also store raw info for the Pydantic response
                source_nodes_info.append({
                    "id": node.node_id,
                    "text": node.text,
                    "metadata": metadata,
                    "score": node_with_score.score
                })

            # Construct the final response text as in the original
            # Original: f"{response.response}\n\nИсточники:\n" + "\n".join(sources)
            final_response_text = f"{response_obj.response}\n\nИсточники:\n" + "\n".join(sources_lines)

            logger.info("Query processed successfully.")
            return QueryResponse(response_text=final_response_text, source_nodes=source_nodes_info)

        except Exception as e:
            logger.error(f"Error during async query execution: {e}")
            # Consider returning an error response instead of raising
            # depending on how the calling API handles exceptions.
            error_msg = f"Произошла ошибка при обработке запроса: {str(e)}"
            return QueryResponse(response_text=error_msg, source_nodes=[])

    # Example method to check if index is loaded
    def is_loaded(self) -> bool:
        """Checks if the index and query engine are loaded."""
        return self._index is not None and self._query_engine is not None

# --- Example Usage (if running rag_engine.py directly) ---
# if __name__ == "__main__":
#     import os
#     DATA_DIR = "./data/documents_to_index" # Example path
#     ENGINE = RAGEngine()
#
#     if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
#         print("Rebuilding index from documents...")
#         ENGINE.rebuild_index(DATA_DIR)
#     else:
#         print("Loading existing index...")
#         ENGINE.load_index() # This will now correctly build the query engine with original params
#
#     if ENGINE.is_loaded():
#         print("Engine is loaded. Example async query:")
#         import asyncio
#         async def run_query():
#             res = await ENGINE.aquery("Какие требования к толщине бетонной плиты перекрытия?")
#             print(f"Response: {res.response_text}")
#             print(f"Sources: {len(res.source_nodes)} found.")
#         asyncio.run(run_query())
#     else:
#         print("Failed to load or build the engine.")
