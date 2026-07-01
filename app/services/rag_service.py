"""RAG service layer.

This is where all business logic lives. The service is responsible for:
- Loading the vector store, retriever, and chain at startup
- Executing retrieval + generation for incoming questions
- Returning answers with source citations

API routes must remain thin and delegate everything here.
"""

from __future__ import annotations

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from app.chains.rag_chain import build_rag_chain
from app.config.settings import Settings
from app.embeddings.embedding_model import get_embedding_model
from app.llm.groq_client import get_groq_llm
from app.retrieval.retriever import extract_sources, get_retriever, retrieve_documents
from app.schemas.response import AskResponse, SourceItem
from app.utils.logger import get_logger
from app.vectorstore.chroma_store import load_vectorstore

logger = get_logger(__name__)


class RAGService:
    """Encapsulates the loaded RAG components and answers questions.

    A single instance is created once at FastAPI startup and stored in
    app.state, so the embedding model, vector store, retriever, and chain
    are never recreated per request.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever: VectorStoreRetriever | None = None
        self.chain: Runnable | None = None
        self._loaded = False

    def load(self) -> None:
        """Load the embedding model, vector store, retriever, and chain.

        Called once during FastAPI startup. Never recreated per request.
        """
        logger.info("Loading RAG service components...")

        embeddings = get_embedding_model(self.settings.embedding_model)

        vectordb = load_vectorstore(
            embeddings=embeddings,
            collection_name=self.settings.collection_name,
            persist_directory=self.settings.chroma_db_dir,
        )

        self.retriever = get_retriever(
            vectordb=vectordb,
            top_k=self.settings.top_k,
            fetch_k=self.settings.fetch_k,
        )

        llm = get_groq_llm(
            api_key=self.settings.groq_api_key,
            model=self.settings.groq_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

        self.chain = build_rag_chain(retriever=self.retriever, llm=llm)
        self._loaded = True

        logger.info("RAG service components loaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Whether the service has completed loading."""
        return self._loaded

    def ask(self, question: str) -> AskResponse:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: The user's natural-language question.

        Returns:
            An AskResponse containing the question, generated answer,
            and source citations.

        Raises:
            RuntimeError: If the service has not been loaded yet.
        """
        if not self._loaded or self.retriever is None or self.chain is None:
            raise RuntimeError("RAGService.load() must be called before ask().")

        logger.info("Question received: %s", question)

        docs = retrieve_documents(self.retriever, question)
        answer = self.chain.invoke(question)
        sources = extract_sources(docs)

        logger.info("Answer generated")

        return AskResponse(
            question=question,
            answer=answer,
            sources=[SourceItem(**s) for s in sources],
        )
