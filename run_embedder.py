#!/usr/bin/env python3

import json

from sentence_transformers import SentenceTransformer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           CONFIG           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

INPUT_FILE = "./out/chunks.jsonl"
OUTPUT_FILE = "./out/chunks_with_embeddings.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           MODÈLE           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

embedder = SentenceTransformer(MODEL_NAME, device="cpu")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#         EMBEDDINGS         #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def embed(texts: list[str]) -> list[list[float]]:
    """
    Calcule les embeddings pour une liste de textes.
    Utilise le modèle SentenceTransformer chargé précédemment.
    """
    return embedder.encode(texts, show_progress_bar=True)


def add_embeddings_to_chunk_file(input_path: str, output_path: str) -> None:
    """
    Ajoute les embeddings aux chunks et écrit dans un nouveau fichier JSONL.
    Chaque chunk dans le fichier de sortie contiendra un champ "embedding".
    """
    # Lecture du fichier JSONL
    chunks = [json.loads(l) for l in open(input_path, "r", encoding="utf-8") if l.strip()]

    # Extraction des textes des chunks
    texts = [str(c.get("text", "") or "").strip() for c in chunks]

    # Calcul des embeddings
    embeddings = embed(texts)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk, emb, text in zip(chunks, embeddings, texts, strict=False):
            chunk["embedding"] = [] if not text else emb.tolist()
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    add_embeddings_to_chunk_file(INPUT_FILE, OUTPUT_FILE)
