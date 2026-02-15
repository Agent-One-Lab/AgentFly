import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(
        self,
        embeddings=None,
        vector_size=None,
        ids=None,
        similarity="cosine",
        index_file=None,
        use_gpu=False,
        num_gpus=1
    ):
        self.similarity = similarity
        self.use_gpu = use_gpu
        self.num_gpus = num_gpus

        # 1. Load or Create the Base Index (CPU)
        if embeddings is not None:
            self.index = faiss.IndexFlatIP(vector_size)
            if similarity == "cosine":
                embeddings /= np.linalg.norm(embeddings, axis=1)[:, None]
            self.index.add(embeddings)
        elif index_file is not None:
            self.index = faiss.read_index(index_file)
            print(f"Loaded FAISS index: {self.index.ntotal} vectors, {self.index.d} dimensions")

        # 2. Move to GPU if requested
        if self.use_gpu:
            self.index = self._convert_to_gpu(self.index, num_gpus)

        # 3. Handle IDs
        if ids is None:
            self.ids = list(range(self.index.ntotal))
        else:
            self.ids = ids

    def _get_gpu_count(self):
        """Get number of available GPUs. faiss.get_num_gpus() can return 0 on some
        conda builds (e.g. conda-forge faiss-gpu 1.8+); fall back to torch if available.
        """
        res_count = faiss.get_num_gpus()
        if res_count > 0:
            return res_count
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 0

    def _convert_to_gpu(self, cpu_index, num_gpus):
        """Robustly shards the index across multiple GPUs using GpuMultipleClonerOptions."""
        res_count = self._get_gpu_count()
        if res_count == 0:
            print("No GPUs found, using CPU index")
            return cpu_index

        actual_gpus = min(num_gpus, res_count)
        print(f"Sharding index across {actual_gpus} GPUs...")

        # Multiple versions compatibility
        try:
            # Use GpuMultipleClonerOptions for multi-GPU sharding
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Enable sharding (splits data)
            co.useFloat16 = True  # Critical for 21M vectors to fit VRAM

            gpu_index = faiss.index_cpu_to_all_gpus(
                cpu_index,
                co=co,
                ngpu=actual_gpus
            )
            return gpu_index
        except Exception as e:
            # Fallback: try single-GPU with StandardGpuResources if available
            # (some conda faiss-gpu builds don't expose StandardGpuResources)
            if (actual_gpus >= 1 and hasattr(faiss, "StandardGpuResources")
                    and hasattr(faiss, "index_cpu_to_gpu")):
                try:
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    print(f"Using single GPU (multi-GPU failed: {e})")
                    return gpu_index
                except Exception as e2:
                    logger.warning("GPU index creation failed (%s), using CPU index", e2)
            else:
                logger.warning("GPU index creation failed (%s), using CPU index", e)
            return cpu_index
    
    def add(self, embeddings, ids=None):
        if self.similarity == "cosine":
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, None]
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        self.index.add(embeddings)
        
        # Update IDs
        new_ids = ids if ids is not None else list(range(self.ids[-1] + 1, self.ids[-1] + 1 + embeddings.shape[0]))
        if self.ids is not None:
            self.ids.extend(new_ids)
        else:
            self.ids = new_ids

    def search(self, queries: np.array, top_n: int):
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self.index.d:
            raise ValueError(f"Query dimension {queries.shape[1]} doesn't match {self.index.d}")

        if self.similarity == "cosine":
            queries /= np.linalg.norm(queries, axis=1)[:, None]

        # FAISS search (releases GIL automatically)
        scores, indexes = self.index.search(queries.astype('float32'), top_n)

        scores_ids = [
            [(float(s), self.ids[i]) for s, i in zip(top_n_score, top_n_idx) if i != -1]
            for top_n_score, top_n_idx in zip(scores, indexes)
        ]
        return scores_ids