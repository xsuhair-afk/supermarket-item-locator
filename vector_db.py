import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

class VectorDB:
    def __init__(self):
        # In-memory storage for simplicity
        # Format: list of dicts {"id": str, "metadata": dict, "vector": np.ndarray}
        self.database = []

    def insert(self, id: str, metadata: dict, vector: np.ndarray):
        self.database.append({
            "id": id,
            "metadata": metadata,
            "vector": vector
        })

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        if not self.database:
            return []

        results = []
        for item in self.database:
            db_vector = item["vector"]
            
            # Calculate 3 metrics
            # 1. Cosine similarity (1 - cosine distance)
            cos_dist = cosine(query_vector, db_vector)
            cos_sim = 1 - cos_dist
            
            # 2. Euclidean Distance (L2)
            l2_dist = euclidean(query_vector, db_vector)
            
            # 3. Manhattan Distance (L1)
            l1_dist = cityblock(query_vector, db_vector)

            results.append({
                "id": item["id"],
                "metadata": item["metadata"],
                "metrics": {
                    "cosine_similarity": float(cos_sim),
                    "euclidean_l2": float(l2_dist),
                    "manhattan_l1": float(l1_dist)
                }
            })

        # Dynamically rank based on Cosine Similarity (best for VGG high-dim normalized embeddings)
        # We can implement a more complex comparative logic if needed.
        results.sort(key=lambda x: x["metrics"]["cosine_similarity"], reverse=True)
        return results[:top_k]
