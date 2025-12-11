"""
API Service - FastAPI Application

This module defines the FastAPI application for the API service.
It acts as an intermediary between the bot and the RAG service.
It receives queries from the bot, potentially checks limits (though this logic
might reside in the bot itself), forwards the query to the RAG service,
and returns the RAG's response back to the bot.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import logging
import sys
import httpx # Recommended for async HTTP requests
# import requests # Alternative, but less suitable for async FastAPI if rag is also async

# Assuming models are shared or redefined for API contract
# We can reuse QueryRequest and QueryResponse from rag_engine if they are in a shared location
# or define them here. For now, let's define them here for API clarity,
# but ensure they are compatible with rag's models.
# Let's assume rag_engine's models are the source of truth and import them.
# However, rag_engine.py is part of the 'rag' service, not a shared library.
# To avoid tight coupling and dependency issues in the 'api' service,
# it's often better to define *API-specific* models here that map to the RAG service's contract.
# For simplicity in this context, and given the rag_engine.py is already designed for this,
# we can define minimal compatible models here or import a shared definition if one exists later.
# For now, let's define compatible models here.

# --- Pydantic Models for API Contract ---
# These should ideally match the contract expected/returned by the RAG service.

class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query_text: str
    # Use Optional[int] with a default that aligns with the rag service's default (e.g., 3)
    top_k: Optional[int] = 3


class SourceNode(BaseModel):
    """Model representing a source node returned by the RAG service."""
    id: str
    text: str
    metadata: dict  # e.g., {'file_name': '...', 'page': ...}
    score: Optional[float] = None # Include score if rag service provides it


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    response_text: str
    source_nodes: list[SourceNode]


# --- Configuration and Logging ---
from api.config import get_api_config # Import config
config = get_api_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Consider adding a file handler later pointing to ./logs
        # logging.FileHandler("./logs/api_service.log")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Service for Snip Assist",
    description="Intermediary API service between the Telegram bot and the RAG service.",
    version="0.1.0",
)

# --- Global HTTP Client for RAG Service ---
# Using httpx.AsyncClient for async requests to the RAG service
# It's good practice to have a single, long-lived client instance
# initialized on startup and closed on shutdown.
async_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the HTTP client for RAG service on startup."""
    global async_client
    logger.info("Starting up API Service...")
    try:
        # Create the async client instance with the base URL of the RAG service
        # This assumes the RAG service is reachable at config.RAG_SERVICE_URL
        # e.g., http://rag:8001
        async_client = httpx.AsyncClient(base_url=config.RAG_SERVICE_URL, timeout=30.0) # 30s timeout
        logger.info(f"HTTP client for RAG service initialized with base URL: {config.RAG_SERVICE_URL}")
    except Exception as e:
        logger.critical(f"Failed to initialize HTTP client for RAG service: {e}")
        raise RuntimeError(f"API Service startup failed due to HTTP client initialization error: {e}") from e

@app.on_event("shutdown")
async def shutdown_event():
    """Close the HTTP client on shutdown."""
    global async_client
    if async_client:
        await async_client.aclose()
        logger.info("HTTP client for RAG service closed.")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    # Check if the async_client is initialized
    client_ok = async_client is not None
    # Optionally, perform a quick check to the RAG service's health endpoint
    rag_ok = False
    if client_ok:
        try:
            # Assuming rag service has a /health endpoint
            rag_health_response = await async_client.get("/health") # Relative to base_url
            rag_ok = rag_health_response.status_code == 200
        except httpx.RequestError as e:
            logger.warning(f"Could not reach RAG service for health check: {e}")
            rag_ok = False

    status = "ok" if client_ok and rag_ok else "degraded"
    details = {"service": "api", "client_initialized": client_ok, "rag_reachable": rag_ok}
    if not rag_ok:
        details["rag_status"] = "unreachable or unhealthy"
    return {"status": status, "details": details}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint to receive query from the bot and forward it to the RAG service.

    Args:
        request (QueryRequest): The query text and parameters from the bot.

    Returns:
        QueryResponse: The response from the RAG service.
    """
    logger.info(f"Received query from bot: {request.query_text[:50]}... (top_k={request.top_k})")

    if async_client is None:
        logger.error("HTTP client for RAG service is not initialized.")
        raise HTTPException(status_code=500, detail="API Service is misconfigured (RAG client not ready).")

    try:
        # Prepare the payload to send to the RAG service
        # This should match the expected format of rag's /query-rag endpoint
        rag_payload = {
            "query_text": request.query_text,
            "top_k": request.top_k
        }

        # Make an async POST request to the RAG service
        # The endpoint is relative to the base_url configured in async_client
        logger.debug(f"Forwarding request to RAG service at /query-rag")
        rag_response = await async_client.post("/query-rag", json=rag_payload)

        # Check if the RAG service responded successfully
        if rag_response.status_code != 200:
            logger.error(f"RAG service returned status {rag_response.status_code}: {rag_response.text}")
            raise HTTPException(status_code=rag_response.status_code, detail=f"RAG service error: {rag_response.text}")

        # Parse the JSON response from the RAG service
        rag_data = rag_response.json()
        logger.info("Successfully received response from RAG service.")

        # Validate the structure of the response (optional but good practice)
        # Pydantic will validate when constructing QueryResponse, raising if invalid
        # We could add more specific checks here if needed.

        # Construct and return the response object for the bot
        # This assumes rag_data structure matches QueryResponse model fields
        return QueryResponse(**rag_data)

    except httpx.RequestError as e:
        logger.error(f"Error making request to RAG service: {e}")
        # More specific error based on the type of RequestError could be added
        raise HTTPException(status_code=502, detail=f"Failed to communicate with RAG service: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"RAG service responded with an error status: {e}")
        # This handles cases where rag returns 4xx or 5xx, though we already checked status_code
        # It's a good catch-all if status check missed something or happens mid-request
        raise HTTPException(status_code=e.response.status_code, detail=f"RAG service responded with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in API query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error in API service: {str(e)}")

# Example additional endpoint if needed, e.g., for status or metrics
# @app.get("/status")
# async def get_status():
#     # Could aggregate status from bot, rag, db if applicable here
#     return {"status": "operational", "upstream_rag_ok": True, ...}

if __name__ == "__main__":
    import uvicorn
    # This allows running api/main.py directly for testing
    # In docker-compose, uvicorn is typically called via CMD in Dockerfile
    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT) # Use config values
