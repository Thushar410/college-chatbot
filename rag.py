import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_chunks(base_folder="college_data", chunk_size=300):
    all_text = ""
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n\n"

    words = all_text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_texts(texts):
    return np.array(embedder.encode(texts), dtype="float32")

def build_index():
    chunks = load_chunks()
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve(query, index, chunks, k=3):
    q_emb = embed_texts([query])
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]