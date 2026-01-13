# -*- coding: utf-8 -*-
"""
CLI script for testing the RAG pipeline in real conditions.

Initializes Chroma (persistent), indexes ./data/docs/text/,
takes a question from the user and returns an answer with sources.
Uses real embeddings and LLM.

This version:
- Does NOT perform automatic evaluation of faithfulness/value match.
- Saves generated answers and sources to a CSV file for manual review.
- Caches embeddings/index in Chroma to avoid re-generation on every run.
- Uses an improved prompt for the LLM.
"""

import logging
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
import chromadb
from chromadb.utils import embedding_functions
# ИМПОРТИРУЕМ конкретное исключение
from chromadb.errors import NotFoundError

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths and Configuration ---
DATA_DOCS_PATH = Path("./data/docs/text")
CHROMA_PERSIST_DIR = Path("./data/chroma")
CHROMA_COLLECTION_NAME = "construction_docs"

EMBEDDING_MODEL_NAME = "./models/embeddings/e5-base-en-ru"
LLM_MODEL_PATH = "./models/Phi-3-mini-4k-instruct-q4.gguf"
LLM_CONTEXT_WINDOW = 4096
LLM_MAX_NEW_TOKENS = 512

CHUNK_SIZE = 128
CHUNK_OVERLAP = 64

TEST_QUESTIONS_PATH = Path("./test_questions.yaml")
OUTPUT_REPORT_PATH = Path("./rag_manual_validation_report.csv")


# --- Utility Functions ---
def normalize_and_extract_value(val: str) -> Any:
    """
    Extracts numeric or alphanumeric value from string for comparison.
    Examples:
        "2,2 м"       -> 2.2
        "В22,5"       -> "В22.5"
        "F150"        -> "F150"
        "0,7–1,0 м"   -> [0.7, 1.0]
    """
    if not val:
        return val

    import re
    # Заменяем ',' → '.', убираем пробелы
    clean = re.sub(r'[^\d\w\.\-,]', '', val.replace(',', '.'))

    # Диапазон: "0.7-1.0"
    if '-' in clean:
        parts = clean.split('-')
        try:
            return [float(p) for p in parts]
        except ValueError:
            return clean

    # Число
    try:
        return float(clean)
    except ValueError:
        pass

    # Символ + число (В22.5, F150)
    match = re.match(r"^([A-Za-z]+)([\d.]+)$", clean)
    if match:
        return f"{match.group(1)}{float(match.group(2))}"

    return clean


def load_test_questions(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        logger.warning(f"Test questions file {file_path} not found. Running in interactive mode.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if not isinstance(data, list):
                logger.error(f"Expected list in {file_path}, got {type(data)}")
                return []
            # Возвращаем как есть, так как теперь не парсим в объекты
            return data
    except Exception as e:
        logger.error(f"Failed to load test questions: {e}")
        return []


# --- RAG Initialization (with caching check) ---
def initialize_rag_and_index(force_rebuild: bool = True): # Изменён default на True
    logger.info("Initializing RAG engine...")

    # 1. Chroma client
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # 2. Check if collection already exists
    collection_exists = False
    try:
        # Попытка получить коллекцию
        chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        collection_exists = True
        logger.info(f"Collection '{CHROMA_COLLECTION_NAME}' already exists. Loading...")
    except NotFoundError: # <-- ЛОВИМ ПРАВИЛЬНОЕ ИСКЛЮЧЕНИЕ
        logger.info(f"Collection '{CHROMA_COLLECTION_NAME}' does not exist. Will create new.")
        collection_exists = False
    except ValueError: # <-- На всякий случай, если Chroma вдруг поменяет тип исключения
        logger.info(f"(Fallback) Collection '{CHROMA_COLLECTION_NAME}' does not exist (ValueError). Will create new.")
        collection_exists = False
    except Exception as e:
        logger.error(f"Unexpected error while checking for collection '{CHROMA_COLLECTION_NAME}': {e}")
        # В случае неожиданной ошибки (например, проблемы с файлом), лучше упасть или попытаться пересоздать
        raise e

    if force_rebuild:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info(f"Collection {CHROMA_COLLECTION_NAME} deleted (force_rebuild=True).")
            collection_exists = False
        except NotFoundError: # <-- ЛОВИМ ПРАВИЛЬНОЕ ИСКЛЮЧЕНИЕ при удалении
            logger.info(f"Collection {CHROMA_COLLECTION_NAME} not found, will create new.")
        except ValueError: # <-- На всякий случай
            logger.info(f"(Fallback) Collection {CHROMA_COLLECTION_NAME} not found (ValueError) for deletion, will create new.")
        except Exception as e:
            logger.error(f"Error deleting collection {CHROMA_COLLECTION_NAME}: {e}")
            # Если не можем удалить, но коллекция существует, это проблема
            if collection_exists:
                 logger.error("Failed to delete existing collection, cannot proceed with force_rebuild.")
                 raise e
            # Если коллекции не существовало, то это нормально

    # 3. Embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device="cpu"
    )

    if collection_exists:
        # Загружаем существующую коллекцию
        chroma_collection = chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    else:
        # Создаём новую коллекцию
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 4. Models
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.embed_model = embed_model

    logger.info(f"Loading LLM from {LLM_MODEL_PATH}...")
    llm = LlamaCPP(
        model_path=LLM_MODEL_PATH,
        temperature=0.1,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
        verbose=False,
    )
    Settings.llm = llm

    # 5. Load or Create Index
    if collection_exists:
        logger.info("Loading existing VectorStoreIndex...")
        # Загружаем индекс из существующего vector_store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("Loaded existing index.")
    else:
        logger.info("Loading documents and creating new VectorStoreIndex...")
        # Загрузка документов и создание индекса (долгий процесс)
        reader = SimpleDirectoryReader(input_dir=DATA_DOCS_PATH, required_exts=[".txt"], recursive=True)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents.")

        node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
            vector_store=vector_store,
            show_progress=True,
        )
        logger.info("Created new index and embeddings.")

    # 6. --- NEW/UPDATED: Improved Prompt for Response Synthesis ---
    # Определяем улучшенный промпт
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
        response_mode=ResponseMode.TREE_SUMMARIZE,
        text_qa_template=custom_prompt_tmpl,
        llm=llm,
    )

    # 7. Query engine (using the custom response_synthesizer)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_synthesizer=response_synthesizer,
    )

    logger.info("RAG engine initialized.")
    return query_engine


# --- Manual Validation Runner ---
def run_manual_validation(query_engine, test_questions: List[Dict[str, Any]]):
    if not test_questions:
        logger.info("No test questions loaded. Skipping batch validation.")
        return

    logger.info(f"Running manual validation on {len(test_questions)} questions...")

    results = []

    for i, q in enumerate(test_questions, 1):
        question_text = q.get("question_text", "").strip() or q.get("question", "").strip() # Поддержка старого и нового формата
        question_id = q.get("question_id", f"Q{i:03d}")

        logger.info(f"Processing {question_id}: {question_text[:50]}...")

        try:
            response = query_engine.query(question_text)

            # Собираем информацию для отчёта
            retrieved_sources = [
                node.node.metadata.get("file_name", "Unknown file")
                for node in response.source_nodes
            ]

            result = {
                "question_id": question_id,
                "question_text": question_text,
                "retrieved_sources": "; ".join(retrieved_sources),
                "generated_answer": response.response,
                "expected_answer_snippet": q.get("expected_answer_snippet", ""),
                "expected_value": q.get("expected_value", ""),
                "expected_section": q.get("expected_section", ""),
                "expected_keywords_in_answer": q.get("expected_keywords_in_answer", []),
                "expected_keywords_in_sources": q.get("expected_keywords_in_sources", []),
                "gold_documents": q.get("gold_documents", []),
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {question_id}: {e}")
            results.append({
                "question_id": question_id,
                "question_text": question_text,
                "retrieved_sources": "",
                "generated_answer": f"ERROR: {e}",
                "expected_answer_snippet": "",
                "expected_value": "",
                "expected_section": "",
                "expected_keywords_in_answer": [],
                "expected_keywords_in_sources": [],
                "gold_documents": [],
            })

    # --- Generate Manual Validation CSV Report ---
    fieldnames = [
        "question_id", "question_text", "retrieved_sources", "generated_answer",
        "expected_answer_snippet", "expected_value", "expected_section",
        "expected_keywords_in_answer", "expected_keywords_in_sources", "gold_documents"
    ]

    with open(OUTPUT_REPORT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\n=== MANUAL VALIDATION REPORT ===")
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Report saved to: {OUTPUT_REPORT_PATH.absolute()}")
    logger.info("Please review 'generated_answer' against 'expected_*' fields manually.")


# --- Interactive Mode ---
def run_interactive_mode(query_engine):
    logger.info("RAG engine ready. Enter questions (type 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
            if not question or question.lower() in ["exit", "quit", "выйти"]:
                break

            response = query_engine.query(question)

            print("\n[ANSWER]")
            print(response.response if response.response else "No answer generated.")

            print("\n[SOURCES]")
            for node in response.source_nodes:
                meta = node.node.metadata
                fn = meta.get("file_name", "Unknown")
                pg = meta.get("page_label", "N/A")
                print(f"- {fn} (pg. {pg})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")

    logger.info("Interactive mode ended.")


# --- Main ---
def main():
    logger.info("Starting CLI RAG test (manual validation mode)...")

    # Load questions first
    test_questions = load_test_questions(TEST_QUESTIONS_PATH)

    # Initialize RAG (will load existing index if available)
    try:
        query_engine = initialize_rag_and_index(force_rebuild=True) # <-- force_rebuild=True по умолчанию и явно передано
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        return

    if test_questions:
        run_manual_validation(query_engine, test_questions)
    else:
        logger.info("No test questions found. Entering interactive mode.")
        run_interactive_mode(query_engine)


if __name__ == "__main__":
    main()
