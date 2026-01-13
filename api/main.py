# -*- coding: utf-8 -*-
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
import asyncio # For async sleep
# import requests # Alternative, but less suitable for async FastAPI if rag is also async

# --- Pydantic Models for API Contract ---
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

# Global flag to track if RAG service is initialized
# NOTE: With async initialization in rag, this flag is less critical here.
# We rely on rag's /health endpoint state.
_is_rag_initialized: bool = False


async def initialize_rag_service_on_demand():
    """
    Calls the RAG service's /initialize endpoint and waits for it to complete.
    This function is kept for potential future use where explicit initialization
    might be triggered by the API, but currently it's not called from query_endpoint.
    """
    global async_client, _is_rag_initialized
    if async_client is None:
        logger.error("HTTP client for RAG service is not initialized in initialize_rag_service_on_demand.")
        raise RuntimeError("API Service is misconfigured (RAG client not ready) in initialize_rag_service_on_demand.")

    if _is_rag_initialized:
        logger.info("RAG service is already initialized (flag check).")
        return

    max_retries = 12 # 12 * 10s = 120s max wait
    retry_delay = 10 # seconds
    attempt = 0

    while attempt < max_retries:
        try:
            logger.info(f"Attempting to initialize RAG service via /initialize (attempt {attempt + 1}/{max_retries})...")
            # Call the initialize endpoint
            init_response = await async_client.post("/initialize", timeout=30.0) # Timeout for the init request itself

            if init_response.status_code == 200:
                logger.info("RAG service initialization started successfully via /initialize.")
                break # Exit the loop if init request was accepted
            else:
                logger.warning(f"RAG service /initialize returned status {init_response.status_code}: {init_response.text}")
                # Could be a 500 if RAG is still busy initializing on a previous call or just not ready
                # We continue to retry

        except httpx.RequestError as e:
            logger.warning(f"Request error during RAG service initialization call: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error during RAG service initialization call: {e}")

        attempt += 1
        if attempt < max_retries:
            logger.info(f"Retrying RAG initialization call in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
        else:
            logger.error("Max retries reached for RAG service initialization call.")
            raise RuntimeError("Failed to call RAG service /initialize endpoint after retries.")

    # Now, poll the /health endpoint until engine is ready
    attempt = 0
    while attempt < max_retries:
        try:
            health_response = await async_client.get("/health", timeout=10.0)
            if health_response.status_code == 200:
                health_data = health_response.json()
                logger.debug(f"RAG health check during explicit init: {health_data}")
                # NOTE: Changed from 'index_loaded' to 'engine_initialized'
                engine_ready = health_data.get("details", {}).get("engine_initialized", False)
                if engine_ready:
                    logger.info("RAG service is ready (engine loaded) after explicit init.")
                    _is_rag_initialized = True
                    return # Success, exit function
                else:
                    logger.info("RAG service is starting up (explicit init), engine not loaded yet. Waiting...")
            else:
                logger.warning(f"RAG health check returned status {health_response.status_code} during explicit init")

        except httpx.RequestError as e:
            logger.warning(f"Health check request error during explicit init: {e}")
        except Exception as e:
            logger.warning(f"Health check unexpected error during explicit init: {e}")

        attempt += 1
        if attempt < max_retries:
            logger.info(f"Retrying health check in {retry_delay} seconds during explicit init...")
            await asyncio.sleep(retry_delay)
        else:
            logger.error("Max retries reached for RAG service health check after explicit initialization call.")
            raise RuntimeError("RAG service did not become ready (engine loaded) after explicit initialization.")


@app.on_event("startup")
async def startup_event():
    """Initialize the HTTP client for RAG service on startup."""
    global async_client
    logger.info("Starting up API Service...")
    try:
        # Create the async client instance with the base URL of the RAG service
        # This assumes the RAG service is reachable at config.RAG_SERVICE_URL
        # e.g., http://rag:8001
        async_client = httpx.AsyncClient(base_url=config.RAG_SERVICE_URL, timeout=30.0) # Default timeout for other calls
        logger.info(f"HTTP client for RAG service initialized with base URL: {config.RAG_SERVICE_URL}")

        # DO NOT wait for RAG service readiness here.
        # Let the bot start, and RAG will be initialized in the background by rag service itself.
        logger.info("API Service startup complete. RAG readiness check deferred to first query or health check.")

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
    rag_details = {}
    if client_ok:
        try:
            # Assuming rag service has a /health endpoint
            rag_health_response = await async_client.get("/health") # Relative to base_url
            if rag_health_response.status_code == 200:
                rag_data = rag_health_response.json()
                # NOTE: Check for 'engine_initialized' instead of 'index_loaded'
                rag_ok = rag_data.get("details", {}).get("engine_initialized", False)
                rag_details = rag_data.get("details", {})
            else:
                logger.warning(f"RAG health check returned status {rag_health_response.status_code}")
        except httpx.RequestError as e:
            logger.warning(f"Could not reach RAG service for health check: {e}")
            rag_ok = False
            rag_details = {"error": str(e)}

    status = "ok" if client_ok and rag_ok else "degraded"
    details = {
        "service": "api",
        "client_initialized": client_ok,
        "rag_reachable_and_ready": rag_ok,
        "rag_details": rag_details # Include rag's health details for debugging
    }
    if not rag_ok:
        details["rag_status"] = "unreachable, unhealthy, or not ready (engine not loaded)"
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

    # NOTE: Removed await initialize_rag_service_on_demand() from here.
    # The RAG service initializes itself asynchronously on startup.
    # If it's not ready, the call to /query-rag below will fail appropriately
    # (likely with a timeout or 503 from rag if it's still initializing),
    # and that error will be propagated.

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
        # Use an increased timeout (e.g., 120s) for the query itself, accounting for potential LLM processing time
        # This timeout is specifically for waiting for the /query-rag response from rag service
        rag_response = await async_client.post("/query-rag", json=rag_payload, timeout=120.0)

        # Check if the RAG service responded successfully
        if rag_response.status_code != 200:
            logger.error(f"RAG service returned status {rag_response.status_code}: {rag_response.text}")
            # Propagate the error status from RAG service
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

    except httpx.TimeoutException as e:
        logger.error(f"Timeout error making request to RAG service: {e}")
        # Gateway Timeout if RAG takes too long to respond to our query
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: RAG service did not respond to the query in time (timeout=120s).")
    except httpx.RequestError as e:
        logger.error(f"Error making request to RAG service: {e}")
        # Connection errors, DNS failures, etc.
        raise HTTPException(status_code=502, detail=f"Failed to communicate with RAG service: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"RAG service responded with an error status: {e}")
        # Handles cases where rag returns 4xx or 5xx
        # Re-raise with the specific status code from RAG
        raise HTTPException(status_code=e.response.status_code, detail=f"RAG service responded with error: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error in API query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error in API service: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # This allows running api/main.py directly for testing
    # In docker-compose, uvicorn is typically called via CMD in Dockerfile
    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT) # Use config values
