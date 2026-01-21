#!/usr/bin/env python3

from typing import Iterator

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
    "Provide concise and accurate answers based on the user's questions. "
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
    """
    TODO: Implémenter la récupération de chunks pertinents en fonction de la requête.
    """
    ...

def build_context(chunks: list[dict]) -> str:
    """
    TODO: Construire un contexte textuel à partir des chunks récupérés.
    """
    ...

def build_prompt(query: str, context: str) -> str:
    """
    TODO: Construire le prompt final à envoyer au LLM en combinant la requête et le contexte.
    """
    ...


# Cette fonction est exposée via une interface Gradio.
# L'interface Gradio est générée dynamiquement à partir des annotations de type
# de la fonction. Ajouter ou modifier des paramètres à la fonction `rag` mettra
# automatiquement à jour l'interface web.
def rag(query: str) -> Iterator[str]:

    # TODO:
    # 1) Retrieval: identifier les documents pertinents.
    # 2) Contexte: créer un contexte à partir des documents.
    # 3) RAG: utiliser le LLM pour générer une réponse basée sur le contexte.
    # 4) Historique: gérer le fil de conversation.

    for token in llm.answer(query, stream=True):
        yield token

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           GRADIO           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

if __name__ == "__main__":
    # Créé et lance l'interface Gradio
    create_gradio_interface(rag)
