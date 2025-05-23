import faiss
import numpy as np


class Authenticator:
    def __init__(self, embedding_dim=512, threshold=0.8):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.threshold = threshold
        self.ids = []

    def load_embeddings(self, embeddings, ids):
        self.index.reset()
        self.ids.clear()

        if embeddings:
            emb_array = np.stack(embeddings).astype(np.float32)
            self.index.add(emb_array)
            self.ids = ids

    def authenticate(self, embedding_tensor):
        if embedding_tensor is None:
            return None, None 
            
        if self.index.ntotal == 0:
            return None, None

        embedding = (
            embedding_tensor.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        )

        D, I = self.index.search(embedding, 1)

        if D[0][0] <= self.threshold:
            return self.ids[I[0][0]], D[0][0]
        return None, D[0][0]
