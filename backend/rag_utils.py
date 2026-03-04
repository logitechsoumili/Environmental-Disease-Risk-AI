import json
import os
import pickle
from typing import Dict, List

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import ollama
except ImportError:
    ollama = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "..", "rag")
INDEX_PATH = os.path.join(RAG_DIR, "faiss.index")
DOCS_PATH = os.path.join(RAG_DIR, "documents.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
TOP_K = 3


_faiss_index = None
_documents = []
_embedder = None


CLASS_QUERY_HINTS = {
    "stagnant_water": "stagnant water, mosquito breeding, dengue, malaria, water-borne disease prevention",
    "garbage_dirty": "solid waste, garbage exposure, sanitation, vector control, diarrhea prevention",
    "air_pollution": "air quality, respiratory disease, asthma prevention, particulate matter, public health guidelines",
    "hygienic_environment": "healthy environment maintenance, sanitation best practices, preventive public health",
}


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_document(doc) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ["text", "content", "page_content", "body", "chunk"]:
            if key in doc and _safe_text(doc[key]):
                return _safe_text(doc[key])
        return _safe_text(doc)
    return _safe_text(doc)


def load_rag_resources() -> None:
    global _faiss_index, _documents, _embedder

    if _documents and _faiss_index is not None and _embedder is not None:
        return

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"RAG documents not found: {DOCS_PATH}")
    with open(DOCS_PATH, "rb") as file:
        raw_docs = pickle.load(file)
    _documents = [_normalize_document(doc) for doc in raw_docs]

    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu.")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    _faiss_index = faiss.read_index(INDEX_PATH)

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    _embedder = SentenceTransformer(EMBED_MODEL_NAME)


def _retrieve_context(environment_class: str, question: str = "", top_k: int = TOP_K) -> List[str]:
    load_rag_resources()

    hint = CLASS_QUERY_HINTS.get(environment_class, environment_class.replace("_", " "))
    query = f"{environment_class} environmental health risks and prevention. {hint}. {question}".strip()

    embedding = _embedder.encode([query], convert_to_numpy=True)
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)

    distances, indices = _faiss_index.search(embedding, top_k)
    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(_documents):
            chunk = _safe_text(_documents[idx])
            if chunk:
                context_chunks.append(chunk)
    return context_chunks


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    if ollama is None:
        raise RuntimeError("ollama python package is not installed.")

    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return _safe_text(response.get("message", {}).get("content"))


def _extract_json_block(text: str) -> Dict:
    text = _safe_text(text)
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _fallback_advisory(environment_class: str) -> Dict[str, List[str]]:
    label = environment_class.replace("_", " ").title()
    return {
        "diseases": [f"General environment-related health risks linked to {label}"],
        "preventive_measures": [
            "Maintain clean surroundings and remove contamination sources",
            "Use protective equipment when exposure risk is high",
            "Seek local public health guidance for current outbreaks",
        ],
        "health_guidelines": [
            "Monitor symptoms such as fever, cough, breathing issues, or skin irritation",
            "Consult a qualified healthcare professional for diagnosis",
            "Use official local health department advisories for final decisions",
        ],
        "rag_answer": "RAG advisory fallback generated because AI context services are unavailable.",
    }


def generate_health_advisory(environment_class: str) -> Dict:
    try:
        context_chunks = _retrieve_context(environment_class=environment_class, question="")
    except Exception:
        return _fallback_advisory(environment_class)

    context = "\n\n".join(context_chunks[:TOP_K])

    # Task instructions based on detected class

    if environment_class == "stagnant_water":
        task = """
    List ONLY 3 diseases caused by stagnant water.
    Must include dengue and malaria.
    Then list exactly 3 preventive measures.
    Then list exactly 2 health guidelines.
    Each item must be under 10 words.
    """

    elif environment_class == "air_pollution":
        task = """
    List ONLY 3 diseases caused by air pollution.
    Then list exactly 3 preventive measures.
    Then list exactly 2 health guidelines.
    Each item must be under 10 words.
    """

    elif environment_class == "garbage_dirty":
        task = """
    List ONLY 3 common diseases caused by garbage accumulation.
    Focus only on realistic infectious diseases.
    Do not include cancers or unrelated illnesses.
    Then list exactly 3 preventive measures.
    Then list exactly 2 health guidelines.
    Each item must be under 10 words.
    """

    elif environment_class == "hygienic_environment":
        task = """
    Display that "No disease risk detected" in disease list.
    Then list exactly 3 preventive practices.
    Then list exactly 2 health promotion guidelines.
    Each item must be under 10 words.
    """

    else:
        task = "Explain the health impact concisely."


    system_prompt = """
    You are an environmental health expert providing evidence-based public health advice.
    Return structured information only and avoid unnecessary commentary.
    """

    user_prompt = f"""
    Detected environmental condition: {environment_class}

    Retrieved context:
    {context}

    Task instructions:
    {task}

    Return ONLY valid JSON in this format:

    {{
    "diseases": ["item1", "item2", "item3"],
    "preventive_measures": ["item1", "item2", "item3"],
    "health_guidelines": ["item1", "item2"]
    }}

    Rules:
    - Do not include any extra text outside JSON.
    - Do not add explanations outside the lists.
    - Follow the exact number of items requested.
    - Each item must be concise.
    """.strip()

    try:
        raw = _call_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
        parsed = _extract_json_block(raw)
        if not parsed:
            return _fallback_advisory(environment_class)

        diseases = parsed.get("diseases", [])
        prevention = parsed.get("preventive_measures", [])
        guidelines = parsed.get("health_guidelines", [])
        rag_answer = _safe_text(parsed.get("rag_answer"))

        if not isinstance(diseases, list):
            diseases = []
        if not isinstance(prevention, list):
            prevention = []
        if not isinstance(guidelines, list):
            guidelines = []

        return {
            "diseases": [str(x) for x in diseases][:6],
            "preventive_measures": [str(x) for x in prevention][:8],
            "health_guidelines": [str(x) for x in guidelines][:8],
            "rag_answer": rag_answer or "Advisory generated from retrieved public health context.",
        }
    except Exception:
        return _fallback_advisory(environment_class)


def answer_followup_question(environment_class: str, question: str) -> str:
    question = _safe_text(question)
    if not question:
        return "Please enter a valid question."

    try:
        context_chunks = _retrieve_context(environment_class=environment_class, question=question)
    except Exception:
        return "Follow-up advisory is unavailable right now. Please try again after enabling RAG dependencies."

    context = "\n\n".join(context_chunks[:TOP_K])

    system_prompt = (
    "You are an environmental health expert answering follow-up questions."
    "Use the provided context and stay focused on the detected condition."
    )

    user_prompt = f"""
    Detected environmental condition: {environment_class}

    User question:
    {question}

    Retrieved context:
    {context}

    Instructions:
    - Answer clearly and concisely
    - Stay relevant to the detected condition
    - 3-5 sentences maximum
    - Avoid mentioning unrelated environmental hazards
    """.strip()

    try:
        return _call_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return "Could not generate the answer from the local LLM. Please check Ollama service and model availability."
