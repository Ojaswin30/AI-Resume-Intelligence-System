import numpy as np
from typing import List
from app.core.config import settings

class EmbeddingEngine:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[EmbeddingEngine] SentenceTransformers not initialized ({e}), using lexical similarity fallback.")
            self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384))
        
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings
        else:
            # Fallback simple bag-of-words / character n-gram representation
            return self._fallback_embed(texts)

    def compute_similarity(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        
        query_vec = self.embed_texts([query])[0]
        cand_vecs = self.embed_texts(candidates)

        # Dot product of normalized vectors = Cosine similarity
        similarities = np.dot(cand_vecs, query_vec)
        # Clip to [0.0, 1.0] range
        clipped = [max(0.0, min(1.0, float(score))) for score in similarities]
        return clipped

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        # Simple vocabulary vectorizer fallback
        vocab = set()
        tokenized = []
        for t in texts:
            tokens = [w.lower() for w in t.split() if len(w) > 2]
            tokenized.append(tokens)
            vocab.update(tokens)
        
        vocab_list = sorted(list(vocab))
        vocab_map = {word: idx for idx, word in enumerate(vocab_list)}
        dim = max(len(vocab_list), 1)

        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            for tok in tokens:
                if tok in vocab_map:
                    matrix[i, vocab_map[tok]] += 1.0
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] = matrix[i] / norm
        return matrix
