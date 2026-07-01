"""RAG chain assembly: retriever + prompt + LLM.

Adapted to expose the retrieved documents (for source
citations) alongside the generated answer.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from app.prompts.rag_prompt import get_rag_prompt
from app.retrieval.retriever import format_docs
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_rag_chain(retriever: VectorStoreRetriever, llm: ChatGroq) -> Runnable:
    """Build the RAG chain: retrieve -> format -> prompt -> LLM -> parse.

    Args:
        retriever: A configured VectorStoreRetriever.
        llm: A configured ChatGroq instance.

    Returns:
        A LangChain Runnable that takes a question string and returns
        the generated answer string.
    """
    prompt = get_rag_prompt()

    def retrieve(question: str) -> dict[str, str]:
        docs = retriever.invoke(question)
        return {"question": question, "context": format_docs(docs)}

    chain = RunnableLambda(retrieve) | prompt | llm | StrOutputParser()
    logger.info("RAG chain assembled")
    return chain
