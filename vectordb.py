import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dim: int, index_path="vector.index", meta_path="metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.load()

    def add(self, vectors: list, texts: list):
        np_vectors = np.array(vectors).astype("float32")
        print(f"Vector shape: {np_vectors.shape}")
        self.index.add(np_vectors)
        self.metadata.extend(texts)
        self.save()

    def search(self, query_vector: list, top_k=3):
        query = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query, top_k)
        return [self.metadata[i] for i in indices[0]]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)