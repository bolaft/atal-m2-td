## TD RAG - objectif

Ce TD vous fait construire une pipeline RAG minimale en 3 étapes :

1) **Chunking** : transformer des documents en morceaux de texte (chunks)
2) **Embeddings** : calculer un vecteur par chunk
3) **RAG** : récupérer les meilleurs chunks, construire un contexte, interroger un LLM

## Installation

### Option A - venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Option B - conda

```bash
conda create -n tp-rag
conda activate tp-rag
```

### Dépendances Python

CPU :

```bash
pip install -r requirements-cpu.txt
```

GPU (si avez déjà CUDA installé) :

```bash
pip install -r requirements.txt
```

### Ollama

Nécessaire seulement si vous ne comptez pas utiliser OpenAI.

```bash
./install_ollama.sh
ollama serve &
ollama pull qwen2:0.5b
```

## Données

Le repo contient déjà un PDF dans `doc/` pour tester.
Vous pouvez ajouter vos propres fichiers PDF dans `doc/`.
Source recommandée : [https://www.insee.fr/fr/statistiques](https://www.insee.fr/fr/statistiques).

## Utilisation (les 3 scripts du TD)

### Étape 1 - Chunking (PDF → chunks)

```bash
./run_chunker.py
```

Sortie : `./out/chunks.jsonl` (1 ligne = 1 chunk) au format :

```json
{"text": "...", "file": "..."}
```

Le chunker est très basique et ne prend en charge que les PDF. À vous de l'améliorer.

Vous pouvez modifier `CHUNK_SIZE` dans `run_chunker.py`.

### Étape 2 - Embeddings (chunks → chunks + embedding)

```bash
./run_embedder.py
```

Sortie : `./out/chunks_with_embeddings.jsonl` avec un champ en plus :

```json
{"text": "...", "file": "...", "embedding": [0.1, -0.2, ...]}
```

Utilise par défaut le modèle `sentence-transformers/all-MiniLM-L6-v2`.

### Étape 3 - RAG (à compléter)

Le client RAG (interface Gradio) est dans `run_rag_client.py`.

Pour le moment la pipeline RAG n'est pas implémentée, l'interface ne permet que d'envoyer des questions au LLM sans contexte.

Il se lance via :

```bash
./run_rag_client.py
```

À vous d'implémenter:
- La recherche des meilleurs chunks (retrieval)
- La construction du contexte (context creation)
- La génération avec citations (generation)
- L'abstention si preuves insuffisantes, etc.

L'objectif est d'avoir un pipeline RAG minimal fonctionnel, capable de répondre à différentes questions en s'appuyant sur les documents chunkés.

Vous n'avez pas de contrainte de méthode, par exemple vous pouvez implémenter un système d'agents via LangGraph comme vous l'avez vu dans un précédent TP.

### LLM : Ollama vs OpenAI

Le choix du backend se fait dans `run_rag_client.py` via :
- `OPENAI_API_KEY` : si `None`, le backend Ollama est utilisé
- `OLLAMA_MODEL_NAME` : nom du modèle Ollama (ex: `qwen3:4b`)
- `OPENAI_MODEL_NAME` : nom du modèle OpenAI (ex: `gpt-4o`)

### Gradio

Le point d'entrée est déjà prêt pour le RAG :
- `create_gradio_interface(rag)` dans `run_rag_client.py`

Gradio utilise la signature de la fonction `rag` et ses annotations de type pour générer automatiquement l'interface utilisateur (champs, options, etc.).

Il n'y a donc rien à développer côté interface.

### Utilisation du LLM

L'implémentation du LLM est dans [lib/llm.py](lib/llm.py), pas la peine d'y toucher normalement. Voici son API :

#### Texte

```python
llm.answer("Quelle est la capitale de la France ?")
# La capitale de la France est Paris.
```

#### Streaming

```python
for token in llm.answer(
    "Quelle est la capitale de la France ?",
    stream=True
):
    print(token, end="", flush=True)
# 'La', 'capitale', 'de', 'la', 'France', 'est', 'Paris.'
```

#### JSON

```python
llm.parse("Jules a 25 ans.", format={
    "name": str,
    "age": int
})
# {"name": "Jules", "age": 25}
```

#### Tool Calling

```python
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

llm.answer("Quelle heure est-il ?", tools=[get_current_time])
# Il est 14:35:20.
```

## Évaluation

Lancez la pipeline une fois (chunks + embeddings), puis démarrez l'interface :

```bash
./run_chunker.py
./run_embedder.py
./run_rag_client.py
```

Pour chaque question ci-dessous, vérifiez :
- La réponse s'appuie sur le document et inclut au moins une citation (page ou identifiant de chunk).
- Les citations pointent vers des passages qui soutiennent bien la réponse.
- Si l'information n'est pas dans le document, le système s'abstient ("je ne sais pas") sans inventer.

### Tests "lookup" (réponse courte + citation précise)

- "Quel est le PIB par habitant des Pays de la Loire en 2016 ?"
- "Quel est le taux de chômage mentionné dans le document (valeur + période) ?"
- "Quel indicateur est utilisé pour mesurer l'effort de R&D régional, et comment est-il défini ?"

### Tests "définition / méthode" (retrieval sur encadrés)

- "Définis solde naturel et solde migratoire tels qu'utilisés dans le dossier."
- "Comment le dossier définit-il l'artificialisation (métrique / source) ?"

### Tests "multi-preuves" (2+ citations attendues)

- "Explique le lien entre croissance démographique, logement et artificialisation (2 passages distincts)."
- "Que dit le dossier sur les impacts de la crise Covid-19 (activité économique, mobilité/air, emploi) ?"

### Tests "robustesse" (reformulation)

Posez deux fois la même question avec une reformulation et comparez :
- La réponse doit rester cohérente.
- Les citations doivent rester pertinentes.

Exemple :
- "Quels sont les grands enjeux mis en avant en introduction ?"
- "En deux phrases, quels enjeux l'avant-propos met-il en avant ?"

### Tests "abstention" (doit répondre "je ne sais pas")

- "Donne le numéro SIRET de l'Insee cité dans ce document."
- "Quel est le salaire moyen des enseignants en Pays de la Loire en 2020, selon ce dossier ?"
- "Quelle est l'adresse e-mail personnelle de l'auteur du dossier ?"
