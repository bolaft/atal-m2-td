#!/usr/bin/env python3

from typing import Iterator
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from run_embedder import embedder

from src.llm import OpenAIGenerator, OllamaGenerator
from src.gradio import create_gradio_interface

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#        Configuration       #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# Nom du modèle Ollama à utiliser
OLLAMA_MODEL_NAME = "qwen3:0.6b"

# Optionnel: définir une clé API OpenAI
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_API_KEY = None

# Définir un prompt système pour guider le comportement du modèle
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Read the provided sources carefully and answer faithfully based on them. "
    "Always include the sources you used in your response. "
    "If you don't know the answer, just say you don't know. "
    "Do not make up answers."
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#             LLM            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

if OPENAI_API_KEY:
    llm = OpenAIGenerator(
        model_name=OPENAI_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        system_prompt=SYSTEM_PROMPT,
    )
else:
    llm = OllamaGenerator(model_name=OLLAMA_MODEL_NAME, system_prompt=SYSTEM_PROMPT)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           REQUÊTE          #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def retrieve_chunks(query: str) -> list[dict]:
    # Charger le modèle TF-IDF
    with open("./out/tfidf_model.pkl", "rb") as f:
        vectorizer: TfidfVectorizer = pickle.load(f)

    # Charger les chunks
    chunks = [json.loads(l) for l in open("./out/chunks_with_embeddings.jsonl", "r", encoding="utf-8") if l.strip()]
    texts = [chunk["text"] for chunk in chunks]

    # Récupération TF-IDF
    query_vec = vectorizer.transform([query])
    text_vecs = vectorizer.transform(texts)
    tfidf_similarities = np.dot(text_vecs, query_vec.T).toarray().flatten()
    tfidf_sorted_indices = np.argsort(tfidf_similarities)[::-1]
    tfidf_top_chunks = [chunks[i] for i in tfidf_sorted_indices[:3]]  # Top 

    # Récupération embedding
    query_embedding = embedder.encode([query])[0] 

    # Filtre en cas d'erreur embedding
    valid_chunks = [chunk for chunk in chunks if chunk.get("embedding")]
    chunk_embeddings = np.array([chunk["embedding"] for chunk in valid_chunks])

    if chunk_embeddings.size == 0:
        print("Warning: No valid embeddings found in chunks.")
        return tfidf_top_chunks

    embedding_similarities = np.dot(chunk_embeddings, query_embedding)
    embedding_sorted_indices = np.argsort(embedding_similarities)[::-1]
    embedding_top_chunks = [valid_chunks[i] for i in embedding_sorted_indices[:3]]  # Top 3

    # Combine and return unique chunks
    combined_chunks = {chunk["text"]: chunk for chunk in (tfidf_top_chunks + embedding_top_chunks)}
    return list(combined_chunks.values())

def build_context(chunks: list[dict]) -> str:
    context_lines = []
    for chunk in chunks:
        text = chunk["text"]
        source = chunk.get("file", "Unknown source")  # Include the document source
        context_lines.append(f"Source: {source}\n{text}")
    return "\n\n".join(context_lines)

def build_prompt(query: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nQUERY:\n{query}"

# Cette fonction est exposée via une interface Gradio.
# L'interface Gradio est générée dynamiquement à partir des annotations de type
# de la fonction. Ajouter ou modifier des paramètres à la fonction `rag` mettra
# automatiquement à jour l'interface web.
def rag(query: str) -> Iterator[str]:
    # Step 1: Récupérer les chunks pertinents
    chunks = retrieve_chunks(query)

    # Step 2: Construire un context
    context = build_context(chunks)

    # Step 3: Construire le prompt final
    prompt = build_prompt(query, context)

    print("PROMPT:\n")
    print(prompt)

    # Step 4: Generate response using the LLM
    for token in llm.answer(prompt, stream=True):
        yield token

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           GRADIO           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

if __name__ == "__main__":
    # Créé et lance l'interface Gradio
    create_gradio_interface(rag)
