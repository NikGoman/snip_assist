import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from app.core.config import settings
import chromadb

logger = logging.getLogger(__name__)

class RAGIndexer:
    """
    Сервис для индексации документов нормативной базы (ГОСТ/СНиП) в ChromaDB.
    Предназначен для однократного запуска при подготовке базы или её обновлении.
    """

    def __init__(self, documents_path: str):
        """
        :param documents_path: Путь к директории с документами (pdf, txt и т.д.)
        """
        self.documents_path = Path(documents_path)
        if not self.documents_path.exists():
            raise FileNotFoundError(f"Директория с документами не найдена: {self.documents_path}")

        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection("construction_norms")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

    def run_indexing(self):
        """
        Запускает процесс индексации.
        1. Читает документы.
        2. Разбивает на узлы.
        3. Создаёт индекс и сохраняет в Chroma.
        """
        logger.info(f"Начинаю индексацию из директории: {self.documents_path}")
        try:
            # 1. Загрузка документов
            documents = SimpleDirectoryReader(
                input_dir=str(self.documents_path),
                required_exts=[".txt", ".pdf", ".docx"]  # Укажите нужные расширения
            ).load_data()

            logger.info(f"Загружено {len(documents)} документов.")

            # 2. Разбиение на узлы
            parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            nodes = parser.get_nodes_from_documents(documents)

            logger.info(f"Создано {len(nodes)} узлов для индексации.")

            # 3. Создание и сохранение индекса
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=None, # Будет создан автоматически векторным хранилищем
                vector_store=self.vector_store,
                show_progress=True
            )

            logger.info("Индексация завершена успешно.")
            return index

        except Exception as e:
            logger.error(f"Ошибка при индексации: {e}")
            raise e

if __name__ == "__main__":
    # Пример использования. Запускать отдельно, например, `python -m app.core.rag_indexer`
    import sys
    if len(sys.argv) != 2:
        print("Использование: python -m app.core.rag_indexer <путь_к_документам>")
        sys.exit(1)

    docs_path = sys.argv[1]
    indexer = RAGIndexer(documents_path=docs_path)
    indexer.run_indexing()