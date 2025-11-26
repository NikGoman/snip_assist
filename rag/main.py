"""
RAG Service - FastAPI Application

This module defines the FastAPI application for the RAG (Retrieval-Augmented Generation) service.
It provides an endpoint to process user queries against a knowledge base using LlamaIndex.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import sys
import asyncio # Required for handling async calls if needed in the endpoint

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
# We can reuse the RAGQueryResponse from rag_engine.py or define specific API models.
# Reusing is fine if the internal model is suitable for the API response.
# If we need different fields for the API, we can create new models here.
# For now, let's assume RAGQueryResponse is suitable for the API too.
# QueryRequest is also potentially reusable if it fits the API contract well.
# Let's define it here for clarity and potential future API-specific changes,
# but make it compatible with the engine's expectations or adapt it.


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query_text: str
    # Use Optional[int] with a default that aligns with the engine's default (which is 3 in aquery)
    # If top_k is not provided, the engine will use its default (3).
    top_k: Optional[int] = 3


# --- Global RAG Engine Instance ---
# It's common to have a global instance in a service.
# Initialization happens in startup_event.
rag_engine_instance: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG engine on startup."""
    global rag_engine_instance
    logger.info("Starting up RAG Service...")
    try:
        rag_engine_instance = RAGEngine()

        # Attempt to load the existing index
        # The RAGEngine.load_index now correctly builds the query engine as per original rag.py
        rag_engine_instance.load_index()

        if rag_engine_instance.is_loaded():
            logger.info("RAG Engine loaded successfully.")
        else:
            logger.warning("RAG Engine initialized but index is not loaded (might be empty store).")

    except Exception as e:
        logger.critical(f"Failed to initialize RAG Engine on startup: {e}")
        # Depending on requirements, you might want to prevent the app from starting
        # raise RuntimeError("RAG Engine failed to start") from e
        logger.error("RAG Service will start but might not be fully functional.")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    status = "ok" if rag_engine_instance and rag_engine_instance.is_loaded() else "degraded (index not loaded)"
    return {"status": status, "service": "rag"}


@app.post("/query-rag", response_model=RAGQueryResponse) # Use the model from rag_engine
async def query_rag_endpoint(request: QueryRequest):
    """
    Endpoint to query the RAG system asynchronously.

    Args:
        request (QueryRequest): The query text and parameters.

    Returns:
        RAGQueryResponse: The generated response and source nodes, formatted as per rag_engine.
    """
    logger.info(f"Received query: {request.query_text[:50]}... (top_k={request.top_k})") # Log first 50 chars and top_k

    if rag_engine_instance is None or not rag_engine_instance.is_loaded():
        logger.error("RAG Engine instance is not initialized or index is not loaded.")
        raise HTTPException(status_code=500, detail="RAG Engine is not ready or index is not loaded. Check server logs.")

    try:
        # Call the *async* query method from the updated rag_engine
        # Pass the top_k from the request, which defaults to 3 in the Pydantic model if not provided
        rag_response = await rag_engine_instance.aquery(
            query_text=request.query_text,
            top_k=request.top_k
        )

        # The rag_response is already a RAGQueryResponse object from rag_engine
        # Return it directly as it matches the endpoint's response_model
        logger.info("Query processed successfully by RAG engine.")
        return rag_response

    except Exception as e:
        logger.error(f"Error processing query in RAG engine: {e}")
        # The rag_engine.aquery now catches internal errors and returns an error response text
        # If an unexpected exception bubbles up here, it's a server-side error.
        raise HTTPException(status_code=500, detail=f"Internal server error during query processing: {str(e)}")


# Example additional endpoint if needed, e.g., for indexing
# @app.post("/index-documents")
# async def index_documents_endpoint(data_dir: str = Body(embed=True)):
#     # Logic to re-index documents from a specified directory
#     if rag_engine_instance is None:
#         raise HTTPException(status_code=500, detail="RAG Engine is not initialized.")
#     try:
#         rag_engine_instance.rebuild_index(data_dir=data_dir)
#         return {"message": f"Indexing completed using documents from {data_dir}"}
#     except Exception as e:
#         logger.error(f"Error during indexing: {e}")
#         raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This allows running rag/main.py directly for testing
    # In docker-compose, uvicorn is typically called via CMD in Dockerfile
    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT) # Use config values
