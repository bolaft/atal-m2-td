#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#           CONFIG           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━ #

INPUT_DIR = "./doc"
OUTPUT_FILE = "./out/chunks.jsonl"
CHUNK_SIZE = 512

nltk.download("punkt")

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

def chunk_sentences(text: str, n: int = 3) -> list[str]:
	sentences = sent_tokenize(text)
	chunks = [" ".join(sentences[i : i + n]) for i in range(0, len(sentences), n)]
	return chunks

def chunk_file(text: str) -> list[str]:
	return chunk_sentences(text, n=3)

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
