from __future__ import annotations
import os
import re
from typing import List, Tuple

#  Config 
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM_OAI = 1536
EMBED_DIM_SBERT = int(os.getenv("SBERT_DIM", "1024"))  # mixedbread 1024 by default
MAX_CHARS = 8000
CTRL_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]")
WS_MULTI   = re.compile(r"\s+")

def _sanitize(s: str) -> str:
    s = "" if s is None else str(s)
    s = CTRL_CHARS.sub(" ", s)
    s = WS_MULTI.sub(" ", s).strip()
    return s[:MAX_CHARS]

#  Backends 
def _try_openai_embed(texts: List[str]) -> Tuple[List[List[float]], int]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")

    from openai import OpenAI  
    client = OpenAI(api_key=key)

    # Preflight with a known good input to catch config issues early
    _ = client.embeddings.create(model=EMBED_MODEL, input=["ping"])

    # Real call in manageable batches
    vecs: List[List[float]] = []
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
    return vecs, EMBED_DIM_OAI

def _try_sbert_embed(texts: List[str]) -> Tuple[List[List[float]], int]:
    from sentence_transformers import SentenceTransformer
    # Prefer a strong free model; fall back further if needed
    model_name = os.getenv("SBERT_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
    m = SentenceTransformer(model_name)
    v = m.encode(texts, normalize_embeddings=True)
    return v.tolist(), v.shape[1] if hasattr(v, "shape") else EMBED_DIM_SBERT

def _fallback_hash_embed(texts: List[str], dim: int = 384) -> Tuple[List[List[float]], int]:
    import hashlib
    vecs: List[List[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        # repeat/cut to length
        bytes_needed = dim * 4  # we’ll convert 4 bytes -> float-ish chunks
        buf = (h * ((bytes_needed // len(h)) + 1))[:bytes_needed]
        # convert groups of 4 bytes into pseudo-floats in [0,1)
        vals = [ (int.from_bytes(buf[j:j+4], "little") % 1000000) / 1000000.0 for j in range(0, len(buf), 4) ]
        vecs.append(vals[:dim])
    return vecs, dim

#  Public helper 
def embed_texts_robust(raw_texts: List[str]) -> Tuple[List[List[float]], int, str]:
    """
    Sanitize, then try OpenAI → SBERT → hash.
    Returns (vectors, dim, backend)
    """
    texts = [_sanitize(s) for s in raw_texts if isinstance(s, (str, bytes)) and _sanitize(s)]
    if not texts:
        return [], 0, "empty"

    # Try OpenAI
    try:
        v, d = _try_openai_embed(texts)
        return v, d, "openai"
    except Exception:
        pass

    # Try SBERT
    try:
        v, d = _try_sbert_embed(texts)
        return v, d, "sbert"
    except Exception:
        pass

    # Last resort
    v, d = _fallback_hash_embed(texts)
    return v, d, "hash"
