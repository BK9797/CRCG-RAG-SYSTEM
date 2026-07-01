"""RAG prompt templates.

The prompt text is preserved verbatim from the original Kaggle script
to keep answer style and grounding behavior identical.
"""

from __future__ import annotations

from langchain.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """You are a precise assistant for the CRCG knowledge base.
Answer ONLY from the provided context.
If the answer is not explicitly supported by the context, say: "I could not find that in the knowledge base."
Do not invent facts.
When a question asks for a number, policy, date, location, or contact detail, quote the exact value from context.
If multiple passages mention the same topic, prefer the most specific and most recent-looking passage.
Keep the answer concise, factual, and easy to verify.
"""

RAG_USER_PROMPT = """Question:
{question}

Context:
{context}

Return:
1) A direct answer.
2) A short bullet list of the supporting facts you used.
3) If relevant, mention the page number(s).
"""


def get_rag_prompt() -> ChatPromptTemplate:
    """Build the chat prompt template used by the RAG chain.

    Returns:
        A ChatPromptTemplate with the system and human messages configured.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ]
    )
