# -*- coding: utf-8 -*-
"""
RAG Engine Module (Updated for llama-index-core >= 0.13.0)

This module encapsulates the core logic for the Retrieval-Augmented Generation (RAG) system
using LlamaIndex (core + integrations) and Chroma. It handles document loading, indexing,
storage, and querying. This version is adapted to use LlamaCPP for LLM, HuggingFace models
for embeddings, and torch. It includes caching logic to load existing indexes and uses an
improved prompt for the LLM.
"""

import logging
from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
# Используем LlamaCPP для LLM
from llama_index.llms.llama_cpp import LlamaCPP
# Используем интеграцию с HuggingFace для эмбеддингов
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    # Альтернативный импорт для новых версий llama-index
    from llama_index.core.embeddings import HuggingFaceEmbedding
# Используем интеграцию с Chroma
from llama_index.vector_stores.chroma import ChromaVectorStore
# Для улучшенного промпта
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
import chromadb
from pydantic import BaseModel, Field
from chromadb.errors import NotFoundError

from rag.config import get_rag_config

logger = logging.getLogger(__name__)
config = get_rag_config()  # Используем ленивую инициализацию из config.py


class QueryResponse(BaseModel):
    """Model representing the response from the RAG query."""
    response_text: str = Field(description="The generated response text.")
    source_nodes: List[dict] = Field(description="List of source nodes with metadata.")


class RAGEngine:
    """
    Encapsulates the RAG logic: loading/indexing documents and querying them.
    Uses LlamaIndex abstractions for modularity and flexibility.
    Updated to use LlamaCPP, caching, and improved prompt.
    Compatible with llama-index-core >= 0.13.0 and integration packages.
    """

    def __init__(self, force_rebuild: bool = False):
        """
        Initializes the RAGEngine by setting up the LLM, embeddings, and vector store.
        Does not load the index yet; call load_index or rebuild_index.

        Args:
            force_rebuild (bool): If True, deletes existing collection and creates a new one.
        """
        logger.info("RAGEngine.__init__ started.")
        self._index = None
        self._query_engine = None
        self._client = None
        self._vector_store = None
        self._llm = None  # Сохраняем LLM для использования в query_engine

        # --- 1. Setup Embeddings (HuggingFace) ---
        try:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
            self._embed_model = HuggingFaceEmbedding(
                model_name=config.EMBEDDING_MODEL_NAME,
                # trust_remote_code=True # Uncomment if using a custom or untrusted model
            )
            logger.info(f"Embedding model loaded: {config.EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {config.EMBEDDING_MODEL_NAME}: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")

        # --- 2. Setup LLM (LlamaCPP) ---
        try:
            logger.info(f"Loading LLM model from {config.LLM_MODEL_PATH}...")
            self._llm = LlamaCPP(
                model_path=config.LLM_MODEL_PATH,
                temperature=0.1,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                context_window=config.LLM_CONTEXT_WINDOW,
                verbose=False,  # Отключаем подробный лог Llama-CPP для чистоты
            )
            logger.info(f"LLM loaded from {config.LLM_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load LLM from {config.LLM_MODEL_PATH}: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")

        # --- 3. Setup Vector Store (Chroma) with Caching Logic ---
        # Проверяем, существует ли коллекция
        logger.info(f"Initializing Chroma client at {config.CHROMA_PERSIST_DIR}")
        self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        # Проверяем существование коллекции
        collection_exists = self._check_collection_exists()

        # Если force_rebuild=True, удаляем существующую коллекцию
        if force_rebuild:
            self._delete_collection_if_exists()
            collection_exists = False

        # Инициализируем ChromaVectorStore
        logger.info("Initializing ChromaVectorStore and embedding function...")
        embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME,
            device="cpu"  # Укажите "cuda", если используете GPU
        )

        if collection_exists:
            logger.info("Getting existing collection...")
            self._chroma_collection = self._client.get_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_fn
            )
        else:
            logger.info("Creating new collection...")
            self._chroma_collection = self._client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}  # Для семантики
            )

        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        logger.info(f"Chroma vector store initialized at {config.CHROMA_PERSIST_DIR}, collection: {config.CHROMA_COLLECTION_NAME}")

        # --- 4. Configure LlamaIndex Settings ---
        # This ensures that the LLM and Embed Model are used globally by LlamaIndex components
        logger.info("Setting up LlamaIndex Settings...")
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model
        logger.info("RAGEngine.__init__ completed.")

    def _check_collection_exists(self) -> bool:
        """Check if the Chroma collection exists."""
        try:
            self._client.get_collection(config.CHROMA_COLLECTION_NAME)
            logger.info(f"Collection '{config.CHROMA_COLLECTION_NAME}' already exists.")
            return True
        except NotFoundError:
            logger.info(f"Collection '{config.CHROMA_COLLECTION_NAME}' does not exist.")
            return False
        except ValueError:
            logger.info(f"(Fallback) Collection '{config.CHROMA_COLLECTION_NAME}' does not exist (ValueError).")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while checking for collection '{config.CHROMA_COLLECTION_NAME}': {e}")
            raise e

    def _delete_collection_if_exists(self):
        """Delete the collection if it exists."""
        try:
            self._client.delete_collection(config.CHROMA_COLLECTION_NAME)
            logger.info(f"Collection {config.CHROMA_COLLECTION_NAME} deleted (force_rebuild=True).")
        except NotFoundError:
            logger.info(f"Collection {config.CHROMA_COLLECTION_NAME} not found, will create new.")
        except ValueError:
            logger.info(f"(Fallback) Collection {config.CHROMA_COLLECTION_NAME} not found (ValueError) for deletion, will create new.")
        except Exception as e:
            logger.error(f"Error deleting collection {config.CHROMA_COLLECTION_NAME}: {e}")
            raise e

    def _build_query_engine(self, top_k: int = 5, response_mode: str = "tree_summarize"):
        """
        Builds the query engine from the loaded index.
        Includes improved prompt for LLM.
        Should be called after load_index or rebuild_index.
        """
        logger.info("_build_query_engine started.")
        if self._index is None:
            raise ValueError("Index is not loaded. Call load_index or rebuild_index first.")

        # --- NEW/UPDATED: Improved Prompt for Response Synthesis ---
        custom_prompt_tmpl = (
            "Ты являешься экспертом по строительным нормативам (ГОСТ, СНиП, СП). "
            "Твоя задача - дать точный и краткий ответ на вопрос пользователя, строго основываясь на предоставленном контексте.\n"
            "Контекст:\n{context_str}\n"
            "Вопрос: {query_str}\n"
            "Если в контексте нет информации для ответа на вопрос, скажи: 'Ответ не найден в документах.'\n"
            "Если информация найдена, сформулируй ответ, ссылаясь на конкретные нормативные акты (например, ГОСТ 26633-2015, СНиП 2.02.01-83 п. 2.41) и, если возможно, укажи числовые значения или номера пунктов.\n"
            "Ответ:"
        )

        # Создаём response_synthesizer с кастомным промптом
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.TREE_SUMMARIZE,  # Используем TREE_SUMMARIZE для кастомного промпта
            text_qa_template=custom_prompt_tmpl,
            llm=self._llm,  # Явно передаём LLM
        )

        # Build the query engine with the custom synthesizer
        self._query_engine = self._index.as_query_engine(
            similarity_top_k=top_k,
            response_synthesizer=response_synthesizer,  # Используем кастомный synthesizer
        )
        logger.info("Query engine built successfully with improved prompt.")

    def load_index(self, data_dir: Optional[str] = None):
        """
        Loads an existing index from the Chroma vector store.
        Checks for collection existence before loading.
        """
        logger.info("load_index started.")
        try:
            # Check if collection exists before attempting to load
            collection_exists = self._check_collection_exists()
            if not collection_exists:
                logger.error(f"Collection '{config.CHROMA_COLLECTION_NAME}' does not exist. Cannot load index.")
                self._index = None
                raise RuntimeError(f"Index collection '{config.CHROMA_COLLECTION_NAME}' not found.")

            # Create the index object connected to the Chroma vector store
            logger.info("Loading VectorStoreIndex from vector store...")
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                show_progress=True
            )
            logger.info("VectorStoreIndex loaded successfully.")

            # Build the query engine with default parameters (top_k=5 как в test_rag.py)
            logger.info("Building query engine...")
            self._build_query_engine(top_k=5, response_mode="tree_summarize")
            logger.info("Query engine built successfully after loading index.")

            logger.info("Index loaded and query engine built successfully.")
        except Exception as e:
            logger.error(f"Failed to load index from vector store: {e}")
            self._index = None
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

            # 2. Transform documents into nodes using the parser
            logger.info("Transforming documents into nodes...")
            nodes = parser(documents)
            logger.info(f"Created {len(nodes)} nodes from documents.")

            # --- NEW: Log content of each node before passing to index ---
            logger.info("Logging content of the first few nodes for debugging:")
            for i, node in enumerate(nodes[:5]): # Логируем только первые 5 для краткости
                logger.debug(f"Node {i} content preview (first 200 chars): {repr(node.text[:200])}")
                logger.debug(f"Node {i} content length: {len(node.text)}")
                logger.debug(f"Node {i} metadata: {node.metadata}")

            # If you suspect a specific node is problematic, you could add a loop like this:
            # for i, node in enumerate(nodes):
            #     logger.info(f"Processing node {i}/{len(nodes)}...")
            #     logger.debug(f"Node {i} content: {repr(node.text)}") # repr() покажет спецсимволы
            #     # Here you could potentially call embed_model.get_text_embedding_batch([node.text])
            #     # in a try-catch to isolate the failing chunk, but it's usually handled by from_documents.
            # --- End NEW Logging ---

            # Create the index from nodes, connected to the Chroma vector store
            logger.info("Building index and storing vectors in Chroma...")
            # Добавим логирование перед вызовом from_documents
            logger.info(f"Starting index creation with embed_model: {self._embed_model.model_name}")
            logger.info(f"Chunk size: {config.CHUNK_SIZE}, overlap: {config.CHUNK_OVERLAP}")
            logger.info(f"Total number of nodes to index: {len(nodes)}")

            self._index = VectorStoreIndex(
                nodes=nodes,
                storage_context=None,  # Use default, which connects to vector_store
                vector_store=self._vector_store,
                embed_model=self._embed_model,
                show_progress=True,
                # Note: transformations=[parser] is not needed here as nodes are already parsed
            )

            # 3. Build the query engine for the newly created index
            self._build_query_engine(top_k=5, response_mode="tree_summarize")
            logger.info("Index rebuilt and persisted to Chroma successfully.")

        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            # Попробуем захватить traceback для более подробной информации
            import traceback
            logger.error(f"Full traceback for index rebuild failure:\n{traceback.format_exc()}")
            raise RuntimeError(f"Index rebuilding failed: {e}") from e

    async def aquery(self, query_text: str, top_k: Optional[int] = 5) -> QueryResponse:
        """
        Asynchronously queries the RAG system with the given text.
        Replicates the async behavior and response formatting of the original rag.py's query method.

        Args:
            query_text (str): The user's query.
            top_k (Optional[int]): Number of top results to retrieve. Defaults to 5 (as in test_rag.py).

        Returns:
            QueryResponse: The generated response and source nodes.
                         The response_text will include formatted sources.
        """
        if self._query_engine is None or self._index is None:
            error_msg = "Ошибка: база данных нормативов недоступна. Пожалуйста, свяжитесь с администратором."
            logger.error("Query engine or index is not loaded.")
            return QueryResponse(response_text=error_msg, source_nodes=[])

        if top_k is None:
            top_k = 5  # Default as in test_rag.py

        logger.info(f"Processing query (top_k={top_k}): {query_text[:50]}...")

        try:
            # Perform the query using the LlamaIndex query engine
            response_obj = self._query_engine.query(query_text)

            # --- NEW: Debug logging for response and sources ---
            logger.info(f"Raw response from query engine: '{response_obj.response}'")
            logger.info(f"Number of source nodes returned: {len(response_obj.source_nodes)}")
            for i, node_with_score in enumerate(response_obj.source_nodes):
                logger.debug(f"Source Node {i}: Score={node_with_score.score}, Text='{node_with_score.node.text[:100]}...'") # repr() для спецсимволов
            # --- End Debug Logging ---

            # --- Replicate Source Formatting from Original rag.py ---
            source_nodes_info = []
            sources_lines = []
            for node_with_score in response_obj.source_nodes:  # Убрано ограничение на 2 как в test_rag.py
                node = node_with_score.node
                metadata = node.metadata
                doc_title = metadata.get("file_name", "Документ")
                page = metadata.get("page_label", "N/A")
                sources_lines.append(f"- {doc_title} (стр. {page})")

                # Also store raw info for the Pydantic response
                source_nodes_info.append({
                    "id": node.node_id,
                    "text": node.text,
                    "metadata": metadata,
                    "score": node_with_score.score
                })

            # Construct the final response text as in the original
            # NEW: Check if response_obj.response is empty and handle accordingly
            if not response_obj.response or response_obj.response.strip() == "":
                logger.warning("Query engine returned an empty response string.")
                final_response_text = f"Ответ не найден в документах.\n\nИсточники:\n" + "\n".join(sources_lines)
            else:
                final_response_text = f"{response_obj.response}\n\nИсточники:\n" + "\n".join(sources_lines)

            logger.info("Query processed successfully.")
            return QueryResponse(response_text=final_response_text, source_nodes=source_nodes_info)

        except Exception as e:
            logger.error(f"Error during async query execution: {e}")
            error_msg = f"Произошла ошибка при обработке запроса: {str(e)}"
            return QueryResponse(response_text=error_msg, source_nodes=[])

    # Example method to check if index is loaded
    def is_loaded(self) -> bool:
        """Checks if the index and query engine are loaded."""
        return self._index is not None and self._query_engine is not None

    def force_rebuild_collection(self):
        """Method to force rebuild the entire collection."""
        logger.info("Force rebuilding collection...")
        self._delete_collection_if_exists()
        # After deletion, the collection will be recreated on next index operation
        collection_exists = False
        embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME,
            device="cpu"
        )
        self._chroma_collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        logger.info("Collection force rebuilt successfully.")


# --- Example Usage (if running rag_engine.py directly) ---
# if __name__ == "__main__":
#     import os
#     DATA_DIR = "./data/documents_to_index" # Example path
#     ENGINE = RAGEngine(force_rebuild=True)
#
#     if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
#         print("Rebuilding index from documents...")
#         ENGINE.rebuild_index(DATA_DIR)
#     else:
#         print("Loading existing index...")
#         ENGINE.load_index() # This will now correctly build the query engine with new params
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
