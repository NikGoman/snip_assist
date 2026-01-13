# -*- coding: utf-8 -*-
"""
RAG Service - FastAPI Application

This module defines the FastAPI application for the RAG service.
It provides an endpoint to process user queries against a knowledge base using LlamaIndex.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import sys
import os

# Import the updated RAGEngine and its configuration
from rag.rag_engine import RAGEngine, QueryResponse as RAGQueryResponse
from rag.config import get_rag_config

config = get_rag_config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Consider adding a file handler later pointing to ./logs
        # logging.FileHandler("./logs/rag_service.log")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Service API",
    description="API for Retrieval-Augmented Generation service using LlamaIndex and Chroma.",
    version="0.1.0",
)

# --- Pydantic Models for API Endpoints ---
class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query_text: str
    # Use Optional[int] with a default that aligns with the engine's default (which is 5 in aquery)
    # If top_k is not provided, the engine will use its default (5).
    top_k: Optional[int] = 5

class RebuildIndexRequest(BaseModel):
    """Request model for the rebuild index endpoint."""
    data_dir: str
    force_rebuild: bool = False

class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    response_text: str
    source_nodes: List[dict]

# --- Global State for RAG Engine ---
# Using asyncio.Event for thread-safe signaling of readiness
_engine_ready: asyncio.Event = asyncio.Event()
rag_engine_instance: Optional['RAGEngine'] = None
_initialization_error: Optional[str] = None


async def _background_initialize():
    """
    Background task to initialize RAGEngine in a separate thread.
    Sets the _engine_ready event upon successful completion.
    """
    global rag_engine_instance, _initialization_error
    try:
        logger.info("Background init: Creating RAGEngine...")
        # Run RAGEngine() with force_rebuild=False by default in background
        rag_engine_instance = await asyncio.to_thread(RAGEngine, force_rebuild=False)
        logger.info("Background init: RAGEngine created, loading index...")
        # Run load_index in a thread pool
        await asyncio.to_thread(rag_engine_instance.load_index)
        logger.info("Background init: RAG Engine loaded successfully.")
    except Exception as e:
        logger.error(f"Background init: Failed to initialize RAG Engine: {e}")
        _initialization_error = str(e)
        # Do NOT set the event on error, keep it unset
        return
    # Only set the event if successful
    _engine_ready.set()
    logger.info("Background init: Event set, RAG Engine is ready.")


@app.on_event("startup")
async def startup_event():
    """Startup event that schedules RAGEngine initialization in the background."""
    logger.info("Starting up RAG Service... Scheduling background initialization.")
    # Schedule the background task
    asyncio.create_task(_background_initialize())
    logger.info("Background task scheduled.")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logger.info("Shutting down RAG Service...")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Reports whether the RAG engine is initialized.
    """
    is_ready = _engine_ready.is_set()
    status = "ok" if is_ready else "degraded (engine not initialized)"
    details = {
        "service": "rag",
        "engine_initialized": is_ready,  # Changed key name to match analysis
        "initialization_error": _initialization_error,
        "config": {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "llm_model_path": config.LLM_MODEL_PATH,
            "collection_name": config.CHROMA_COLLECTION_NAME,
            "persist_dir": config.CHROMA_PERSIST_DIR
        }
    }
    return {"status": status, "details": details}


@app.post("/initialize", status_code=200)
async def initialize_engine():
    """
    Explicit endpoint to wait for RAG engine initialization.
    Useful for readiness probes or explicit initialization calls from other services.
    """
    timeout = 120  # seconds
    try:
        await asyncio.wait_for(_engine_ready.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        # Check if there was an error during init
        if _initialization_error:
            raise HTTPException(status_code=500, detail=f"RAG Engine initialization failed: {_initialization_error}")
        else:
            raise HTTPException(status_code=500, detail="RAG Engine initialization timed out.")
    return {"message": "RAG Engine initialized successfully."}


@app.post("/rebuild-index", status_code=200)
async def rebuild_index_endpoint(request: RebuildIndexRequest):
    """
    Endpoint to rebuild the index from a specified directory.
    This will force delete the existing collection and create a new one if force_rebuild is True.
    """
    global rag_engine_instance, _engine_ready, _initialization_error

    logger.info(f"Rebuilding index from directory: {request.data_dir}, force_rebuild: {request.force_rebuild}")

    # Check if directory exists
    if not os.path.isdir(request.data_dir):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {request.data_dir}")

    try:
        # If engine is already initialized, we need to recreate it with force_rebuild
        if rag_engine_instance is not None:
            # Reset the event
            _engine_ready.clear()
            # Create new engine with the appropriate force_rebuild flag
            rag_engine_instance = await asyncio.to_thread(RAGEngine, force_rebuild=request.force_rebuild)
        else:
            # If engine is not yet initialized, create it with the appropriate flag
            rag_engine_instance = await asyncio.to_thread(RAGEngine, force_rebuild=request.force_rebuild)

        # Rebuild the index
        await asyncio.to_thread(rag_engine_instance.rebuild_index, request.data_dir)

        # Set the ready event
        _engine_ready.set()

        logger.info("Index rebuilt successfully.")
        return {"message": "Index rebuilt successfully", "directory": request.data_dir, "force_rebuild": request.force_rebuild}

    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        _initialization_error = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


@app.post("/query-rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: QueryRequest):
    """
    Endpoint to query the RAG system asynchronously.
    Waits for the engine to be ready before processing the query.
    """
    logger.info(f"Received query: {request.query_text[:50]}... (top_k={request.top_k})")

    # Wait for the engine to be ready (with a timeout)
    timeout = 30  # seconds for query might be appropriate
    try:
        await asyncio.wait_for(_engine_ready.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        # Check if there was an error during init
        if _initialization_error:
            raise HTTPException(status_code=500, detail=f"RAG Engine is not ready: {_initialization_error}")
        else:
            raise HTTPException(status_code=503, detail="RAG Engine is not ready (timed out waiting).")

    # Now rag_engine_instance is guaranteed to be loaded
    try:
        # Call the *async* query method from the updated rag_engine
        # Pass the top_k from the request, which defaults to 5 in the Pydantic model if not provided
        rag_response = await rag_engine_instance.aquery(
            query_text=request.query_text,
            top_k=request.top_k
        )
        logger.info("Query processed successfully by RAG engine.")
        logger.info(f"RAG returning response: {rag_response}") # Добавлено логирование ответа
        return rag_response
    except Exception as e:
        logger.error(f"Error processing query in RAG engine: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during query processing: {str(e)}")


@app.get("/status")
async def get_status():
    """
    Get detailed status of the RAG engine.
    """
    is_ready = _engine_ready.is_set()

    status_info = {
        "engine_ready": is_ready,
        "engine_loaded": rag_engine_instance is not None,
        "initialization_error": _initialization_error,
        "config": {
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "llm_model_path": config.LLM_MODEL_PATH,
            "collection_name": config.CHROMA_COLLECTION_NAME,
            "persist_dir": config.CHROMA_PERSIST_DIR,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "service_host": config.SERVICE_HOST,
            "service_port": config.SERVICE_PORT
        }
    }

    if rag_engine_instance:
        status_info["engine_status"] = {
            "is_loaded": rag_engine_instance.is_loaded()
        }

    return status_info


if __name__ == "__main__":
    import uvicorn
    # Use config values
    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT)
