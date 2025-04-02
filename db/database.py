import sqlite3

import numpy as np
import torch


class Database:
    def __init__(self, db_path="db/faces.db", embedding_dim=512):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    user_id TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)
            conn.commit()

    def save_embedding(self, user_id: str, embedding_tensor: torch.Tensor):
        embedding = embedding_tensor.detach().cpu().numpy().astype(np.float32)
        blob = embedding.tobytes()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO face_embeddings (user_id, embedding)
                VALUES (?, ?)
            """,
                (user_id, blob),
            )
            conn.commit()

    def load_all_embeddings(self):
        embeddings = []
        user_ids = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, embedding FROM face_embeddings")
            for user_id, blob in cursor.fetchall():
                embedding = np.frombuffer(blob, dtype=np.float32)
                if embedding.shape[0] == self.embedding_dim:
                    embeddings.append(embedding)
                    user_ids.append(user_id)

        return embeddings, user_ids
