import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.rag import RAGService
from llama_index.core import VectorStoreIndex
from llama_index.llms.transformers import TransformersLLM


@pytest.fixture
def mock_chroma_client():
    """Фикстура для мока ChromaDB клиента."""
    with patch("app.core.rag.chromadb") as mock_chroma:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chroma.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        yield mock_chroma


@pytest.fixture
def mock_llm():
    """Фикстура для мока LLM."""
    with patch("app.core.rag.AutoModelForCausalLM") as mock_model, \
         patch("app.core.rag.AutoTokenizer") as mock_tokenizer:
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        yield TransformersLLM(
            model=MagicMock(),
            tokenizer=MagicMock(),
            context_window=4096,
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.1},
            device_map="auto"
        )


@pytest.fixture
def rag_service(mock_chroma_client, mock_llm):
    """Фикстура для инициализации RAGService с моками."""
    with patch.object(RAGService, '_init_llm', return_value=mock_llm), \
         patch.object(RAGService, 'load_index'):
        service = RAGService()
        # Заменяем индекс на мок, чтобы не зависеть от реального индекса
        service.index = MagicMock(spec=VectorStoreIndex)
        return service


@pytest.mark.asyncio
async def test_rag_query_success(rag_service):
    """Тест успешного запроса к RAG."""
    mock_query_engine = AsyncMock()
    mock_response = MagicMock()
    mock_response.response = "Mocked RAG answer."
    mock_response.source_nodes = [
        MagicMock(metadata={"file_name": "gost_123.pdf", "page_label": "5"}),
        MagicMock(metadata={"file_name": "snip_456.docx", "page_label": "12"}),
    ]
    mock_query_engine.aquery.return_value = mock_response
    rag_service.index.as_query_engine.return_value = mock_query_engine

    query = "Какой ГОСТ регламентирует бетон?"
    result = await rag_service.query(query)

    expected_sources = "- gost_123.pdf (стр. 5)\n- snip_456.docx (стр. 12)"
    expected_result = f"Mocked RAG answer.\n\nИсточники:\n{expected_sources}"

    assert result == expected_result
    mock_query_engine.aquery.assert_called_once_with(query)


@pytest.mark.asyncio
async def test_rag_query_no_index(rag_service):
    """Тест запроса к RAG, когда индекс не загружен."""
    rag_service.index = None

    query = "Какой ГОСТ регламентирует бетон?"
    result = await rag_service.query(query)

    expected_result = "Ошибка: база данных нормативов недоступна. Пожалуйста, свяжитесь с администратором."
    assert result == expected_result


@pytest.mark.asyncio
async def test_rag_query_empty_sources(rag_service):
    """Тест запроса к RAG, когда источников нет."""
    mock_query_engine = AsyncMock()
    mock_response = MagicMock()
    mock_response.response = "Mocked RAG answer."
    mock_response.source_nodes = []  # Пустой список источников
    mock_query_engine.aquery.return_value = mock_response
    rag_service.index.as_query_engine.return_value = mock_query_engine

    query = "Какой ГОСТ регламентирует бетон?"
    result = await rag_service.query(query)

    expected_result = f"Mocked RAG answer.\n\nИсточники:\n"
    assert result == expected_result