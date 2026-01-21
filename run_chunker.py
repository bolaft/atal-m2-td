#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           CONFIG           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

INPUT_DIR = "./doc"
OUTPUT_FILE = "./out/chunks.jsonl"
CHUNK_SIZE = 512

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           PARSER           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def read_file(path: Path) -> str:
	"""
	Parser basique qui ne lit que les PDFs.
	Extrait le texte brut du PDF et le retourne.
	"""
	reader = PdfReader(str(path))
	text = ""

	for page in reader.pages:
		text += page.extract_text() or ""

	return text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           CHUNKER          #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def chunk_file(text: str) -> list[str]:
	"""
	Simple chunker qui divise le texte en morceaux de taille fixe.
	"""

	words = text.split()
	chunks = []
	current_chunk = []

	for word in words:
		current_chunk.append(word)
		if len(" ".join(current_chunk)) >= CHUNK_SIZE:
			chunks.append(" ".join(current_chunk))
			current_chunk = []

	if current_chunk:
		chunks.append(" ".join(current_chunk))

	return chunks

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#             RUN            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

if __name__ == "__main__":
	out = Path(OUTPUT_FILE)
	out.parent.mkdir(parents=True, exist_ok=True)

	pdfs = sorted(Path(INPUT_DIR).rglob("*.pdf"))
	with out.open("w", encoding="utf-8") as f:
		for pdf in tqdm(pdfs, desc="Chunking PDFs", unit="file"):
			text = read_file(pdf)
			for chunk in chunk_file(text):
				f.write(
					json.dumps({"text": chunk, "file": str(Path(pdf).name)}, ensure_ascii=False)
					+ "\n"
				)
