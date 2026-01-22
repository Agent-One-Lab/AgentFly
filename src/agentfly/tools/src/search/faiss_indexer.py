import faiss
import numpy as np


class Indexer:
    def __init__(
        self,
        embeddings=None,
        vector_size=None,
        ids=None,
        similarity="cosine",
        index_file=None,
    ):
        assert embeddings is not None or index_file is not None, (
            "Either embeddings or index_file must be provided"
        )
        self.similarity = similarity

        if embeddings is not None:
            self.index = faiss.IndexFlatIP(vector_size)
            if similarity == "cosine":
                embeddings /= np.linalg.norm(embeddings, axis=1)[:, None]
            self.index.add(embeddings)
        elif index_file is not None:
            # Just load the index as-is, no conversion
            self.index = faiss.read_index(index_file)
            print(
                f"Loaded FAISS index: {self.index.ntotal} vectors, {self.index.d} dimensions"
            )

        if ids is None:
            if embeddings is not None:
                self.ids = list(range(embeddings.shape[0]))
            else:
                self.ids = None
        else:
            self.ids = ids

    def add(self, embeddings, ids=None):
        if self.similarity == "cosine":
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, None]
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        if ids is None:
            self.ids.extend(
                list(range(self.ids[-1] + 1, self.ids[-1] + 1 + embeddings.shape[0]))
            )
        else:
            self.ids.extend(ids)

    def search(self, queries: np.array, top_n: int):
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)

        # Check dimension compatibility
        if queries.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension {queries.shape[1]} doesn't match index dimension {self.index.d}"
            )

        if self.similarity == "cosine":
            queries /= np.linalg.norm(queries, axis=1)[:, None]
        scores, indexes = self.index.search(queries, top_n)

        scores_ids = [
            [(s, self.ids[i]) for s, i in zip(top_n_score, top_n_idx)]
            for top_n_score, top_n_idx in zip(scores, indexes)
        ]

        return scores_ids
