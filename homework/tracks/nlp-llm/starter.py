"""Starter for RAG NLP Homework."""

import marimo
import torch
import numpy as np

__generated_with = "0.10.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    return (mo,)

@app.cell
def __():
    class Chunker:
        def __init__(self, chunk_size: int, overlap: int):
            self.chunk_size = chunk_size
            self.overlap = overlap
            
        def split(self, text: str):
            # TODO: Реализовать нарезку текста
            pass
    return (Chunker,)

@app.cell
def __():
    class SimpleEmbedder:
        def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
            # TODO: Загрузить модель
            pass
            
        def encode(self, texts: list[str]):
            # TODO: Сделать forward pass и вернуть эмбеддинги
            pass
    return (SimpleEmbedder,)

@app.cell
def __():
    class VectorStore:
        def __init__(self):
            # В памяти (in-memory) или FAISS
            self.vectors = []
            self.texts = []
            
        def add(self, texts: list[str], embeddings):
            # TODO: Сохранить
            pass
            
        def search(self, query_emb, top_k: int):
            # TODO: Найти top-k ближайших (cosine similarity)
            pass
    return (VectorStore,)

@app.cell
def __(Chunker, SimpleEmbedder, VectorStore):
    class RAGPipeline:
        def __init__(self):
            # TODO: Инициализация компонентов
            pass
            
        def index(self, documents: list[str]):
            # TODO: chunk -> encode -> store
            pass
            
        def answer(self, query: str) -> str:
            # TODO: encode query -> search -> generate answer
            pass
    return (RAGPipeline,)

if __name__ == "__main__":
    app.run()
