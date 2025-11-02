from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.transformers import TransformersLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import chromadb
from app.core.config import settings
import logging

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection("construction_norms")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        self.load_index()
        self.llm = self._init_llm()

    def _init_llm(self):
        model_name = settings.MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        llm = TransformersLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=4096,
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.1},
            device_map="auto"
        )
        return llm

    def load_index(self):
        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                show_progress=True
            )
        except Exception as e:
            # --- Заменено print на logger.error ---
            logger.error(f"Ошибка при загрузке индекса: {e}")
            # --- /Заменено print на logger.error ---
            self.index = None

    async def query(self, user_query: str) -> str:
        if not self.index:
            return "Ошибка: база данных нормативов недоступна. Пожалуйста, свяжитесь с администратором."

        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=3,
            response_mode="tree_summarize"
        )
        response = await query_engine.aquery(user_query)

        # Извлечение метаданных для ссылок
        source_nodes = response.source_nodes
        sources = []
        for node in source_nodes[:2]:  # Берём топ-2 источника
            metadata = node.metadata
            doc_title = metadata.get("file_name", "Документ")
            page = metadata.get("page_label", "N/A")
            sources.append(f"- {doc_title} (стр. {page})")

        result = f"{response.response}\n\nИсточники:\n" + "\n".join(sources)
        return result
