import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.query_service import QueryService
from app.core.rag import RAGService

@pytest.fixture
def mock_rag_service():
    service = RAGService()
    service.query = AsyncMock(return_value="Mocked response")
    return service

@pytest.fixture
def query_service(mock_rag_service):
    service = QueryService()
    service.rag_service = mock_rag_service
    return service