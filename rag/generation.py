
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

MODEL = "qwen/qwen3.8-27b"

_CLIENT = None


def get_client() -> Groq:
    global _CLIENT

    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        _CLIENT = Groq(api_key=api_key)

    return _CLIENT


def generate_answer(query: str, documents: list) -> str:
    if not documents:
        return (
            "The answer could not be found because "
            "no relevant document context was provided."
        )

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')} "
        f"| Page: {doc.metadata.get('page', 'Unknown')}]\n"
        f"{doc.page_content.strip()}"
        for doc in documents
    )

    prompt = f"""Answer the user's question using ONLY the provided context snippets below.

If the context does not contain enough information to answer,
state that the answer could not be found in the provided documents.

Do not invent or extrapolate information.

Cite the source document and page number when answering.

Context:
{context}

Question:
{query}

Answer:"""

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise, grounded "
                        "document-based RAG assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=600,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Generation error: {e}")
        return "Unable to generate an answer at this time."

