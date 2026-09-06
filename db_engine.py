"""ChromaDB persistence and versioned visual/text embedding helpers."""
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import chromadb
import torch
from PIL import Image

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_VERSION, ensure_directories
from models.runtime import runtime

LOGGER = logging.getLogger(__name__)
_vector_collection: Any = None
_COLLECTION_NAME = f"cctv_embeddings_{EMBEDDING_MODEL_NAME}_{EMBEDDING_VERSION}"


def _device() -> str:
    """Return the fastest PyTorch accelerator available on this machine."""
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _load_visual_model() -> tuple[Any, Any, str]:
    """Return the active visual model via the shared lazy runtime."""
    loaded = runtime.visual()
    return loaded.model, loaded.processor, loaded.device


def _collection():
    global _vector_collection
    if _vector_collection is not None:
        return _vector_collection
    ensure_directories()
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    _vector_collection = client.get_or_create_collection(
        name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return _vector_collection


def _feature_tensor(outputs: object, embedding_name: str) -> torch.Tensor:
    """Return a 2-D embedding tensor from CLIP/SigLIP-family outputs."""
    features = getattr(outputs, embedding_name, None)
    if not isinstance(features, torch.Tensor):
        # BaseModelOutputWithPooling exposes this directly. Prefer it over
        # ``outputs[0]``, which is the full token sequence and not one vector.
        features = getattr(outputs, "pooler_output", None)
    if not isinstance(features, torch.Tensor):
        try:
            features = outputs[0]  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            features = outputs
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Visual model did not return a tensor for {embedding_name}")
    # ``outputs[0]`` from an encoder-only fallback is [batch, tokens, hidden].
    if features.ndim == 3:
        features = features[:, 0, :]
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2:
        raise TypeError(f"Visual model returned an invalid {features.ndim}-D embedding tensor")
    return features


def _normalise_features(features: torch.Tensor) -> torch.Tensor:
    """L2-normalise embeddings without ever calling ``norm`` on an output wrapper."""
    return features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def get_image_embedding(
    image: Image.Image | list[Image.Image], batch_size: int = 32
) -> list[float] | list[list[float]]:
    """Embed one image or a list of images, batching list inputs for throughput."""
    if isinstance(image, list):
        return get_image_embeddings(image, batch_size=batch_size)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image or a list of PIL images")
    return get_image_embeddings([image], batch_size=1)[0]


def get_image_embeddings(images: list[Image.Image], batch_size: int = 32) -> list[list[float]]:
    """Return normalised active-model vectors in GPU/CPU-friendly batches."""
    if not images:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not all(isinstance(image, Image.Image) for image in images):
        raise TypeError("images must contain only PIL.Image.Image values")
    model, processor, device = _load_visual_model()
    vectors: list[list[float]] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=[image.convert("RGB") for image in batch], return_tensors="pt").to(device)
        if device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            tensor = _feature_tensor(model.get_image_features(**inputs), "image_embeds")
            tensor = _normalise_features(tensor)
        vectors.extend(tensor.cpu().detach().numpy().tolist())
    return vectors


def _generate_prompt_variations(query: str) -> list[str]:
    query_lower = query.lower().strip()
    variations = [query]
    
    # Semantic mappings for common CCTV queries
    mappings = [
        (["phone", "cellphone", "mobile"], 
         ["speaking on a phone", "holding a mobile phone to ear", "talking on a cell phone"]),
        (["backpack", "bag", "pack"], 
         ["carrying a backpack", "wearing a backpack", "person with a bag"]),
        (["red jacket", "red shirt", "red coat", "red hoodie"], 
         ["person wearing red jacket", "subject dressed in red top", "individual in red clothing"]),
        (["white car", "white sedan", "white vehicle"], 
         ["a white car", "a white vehicle on the road", "a white sedan automobile"]),
        (["black car", "black sedan", "black vehicle"], 
         ["a black car", "a black vehicle on the road", "a black sedan automobile"]),
    ]
    
    for keywords, prompts in mappings:
        if any(keyword in query_lower for keyword in keywords):
            subject = "person"
            if "woman" in query_lower or "female" in query_lower or "girl" in query_lower:
                subject = "woman"
            elif "man" in query_lower or "male" in query_lower or "boy" in query_lower:
                subject = "man"
            elif "car" in query_lower or "vehicle" in query_lower or "truck" in query_lower or "bus" in query_lower:
                subject = "vehicle"
                
            for p in prompts:
                p_custom = p
                if "person" in p and subject != "person":
                    p_custom = p.replace("person", subject)
                if p_custom not in variations:
                    variations.append(p_custom)
            break
    else:
        if "wearing" in query_lower:
            variations.append(query_lower.replace("wearing", "dressed in"))
            variations.append(f"a person {query_lower}")
        elif "in " in query_lower:
            variations.append(query_lower.replace("in ", "wearing "))
            variations.append(f"a subject {query_lower}")
        else:
            variations.append(f"a photo of {query_lower}")
            variations.append(f"a clear view of {query_lower}")
            
    return list(dict.fromkeys(variations))[:3]


def get_text_embedding(text: str) -> list[float]:
    """Return an L2-normalised vector from the active text/vision model."""
    if not text or not text.strip():
        raise ValueError("Search text cannot be empty")
    model, processor, device = _load_visual_model()
    prompts = _generate_prompt_variations(text)
    LOGGER.info("Generated prompt variations for query '%s': %s", text, prompts)
    
    embeddings = []
    for prompt in prompts:
        # SigLIP2 was trained with fixed-length padding; CLIP accepts this too.
        inputs = processor(text=[prompt], return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            tensor = _feature_tensor(model.get_text_features(**inputs), "text_embeds")
            tensor = _normalise_features(tensor)
            embeddings.append(tensor)
            
    # Average the vectors
    stacked = torch.stack(embeddings, dim=0)
    mean_tensor = torch.mean(stacked, dim=0)
    mean_tensor = mean_tensor / mean_tensor.norm(p=2, dim=-1, keepdim=True)
    
    return mean_tensor.squeeze(0).cpu().detach().numpy().tolist()


def add_frame_data(
    camera_id: str,
    video_source: str,
    timestamp: float,
    frame_idx: int,
    embedding: list[float],
    bounding_box: list[float] | tuple[float, float, float, float],
    primary_plate: str = "",
    plate_candidates: list[str] | None = None,
    confidence: float = 0.0,
    record_id: str = "",
) -> str:
    """Persist a detected subject vector plus serialisable CCTV metadata."""
    if not record_id:
        video_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in camera_id)[:80]
        record_id = f"{video_id}_unknown_{frame_idx}_{uuid.uuid4().hex[:8]}"
    candidates = plate_candidates or []
    metadata = {
        "camera_id": str(camera_id), "video_source": str(video_source),
        "timestamp": float(timestamp), "frame_idx": int(frame_idx),
        "bbox": ",".join(f"{float(value):.1f}" for value in bounding_box),
        "primary_plate": str(primary_plate),
        "plate_candidates": ",".join(candidates),
        "confidence": str(confidence),
    }
    document = " ".join(candidates)
    _collection().upsert(ids=[record_id], embeddings=[embedding], metadatas=[metadata], documents=[document])
    return record_id


def add_frames_data(records: list[dict[str, Any]]) -> list[str]:
    """Persist a group of detections in one Chroma transaction."""
    if not records:
        return []
    ids, embeddings, metadatas, documents = [], [], [], []
    seen_ids = set()
    for record in records:
        camera_id, video_source = str(record["camera_id"]), str(record["video_source"])
        frame_idx, bounding_box = int(record["frame_idx"]), record["bounding_box"]
        
        record_id = record.get("id") or record.get("record_id")
        if not record_id:
            video_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in camera_id)[:80]
            track_id = record.get("track_id") or "unknown"
            record_id = f"{video_id}_{track_id}_{frame_idx}_{uuid.uuid4().hex[:8]}"
        
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)
        
        ids.append(record_id)
        embeddings.append(record["embedding"])
        
        candidates = record.get("plate_candidates", [])
        metadatas.append({
            "camera_id": camera_id, "video_source": video_source,
            "timestamp": float(record["timestamp"]), "frame_idx": frame_idx,
            "bbox": ",".join(f"{float(value):.1f}" for value in bounding_box),
            "primary_plate": str(record.get("primary_plate", "")),
            "plate_candidates": ",".join(candidates),
            "confidence": str(record.get("confidence", 0.0)),
        })
        documents.append(" ".join(candidates))

    # Filter final ids list in a seen_ids set before collection.upsert() to prevent duplicate key crashes
    final_ids, final_embeddings, final_metadatas, final_documents = [], [], [], []
    final_seen = set()
    for item_id, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
        if item_id not in final_seen:
            final_seen.add(item_id)
            final_ids.append(item_id)
            final_embeddings.append(emb)
            final_metadatas.append(meta)
            final_documents.append(doc)

    if final_ids:
        _collection().upsert(ids=final_ids, embeddings=final_embeddings, metadatas=final_metadatas, documents=final_documents)
    return final_ids


def query_vectors(
    query_embedding: list[float],
    top_k: int = 10,
    camera_filter: Optional[str] = None,
    query_text: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return flat, score-sorted vector matches, optionally from one camera."""
    if query_text:
        clean_query = "".join(c for c in query_text if c.isalnum()).upper()
        if len(clean_query) >= 3:
            where_filter = {"camera_id": str(camera_filter)} if camera_filter else None
            response = _collection().query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k)),
                where=where_filter,
                where_document={"$contains": clean_query},
                include=["metadatas", "distances"],
            )
            ids = response.get("ids", [[]])[0]
            if ids:
                metadata = response.get("metadatas", [[]])[0]
                distances = response.get("distances", [[]])[0]
                return [
                    {"id": item_id, "metadata": item_metadata, "distance": float(distance),
                     "similarity": round(1.0 - float(distance), 4)}
                    for item_id, item_metadata, distance in zip(ids, metadata, distances)
                ]
                
    where = {"camera_id": str(camera_filter)} if camera_filter else None
    response = _collection().query(
        query_embeddings=[query_embedding], n_results=max(1, int(top_k)),
        where=where, include=["metadatas", "distances"],
    )
    ids = response.get("ids", [[]])[0]
    metadata = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    return [
        {"id": item_id, "metadata": item_metadata, "distance": float(distance),
         "similarity": round(1.0 - float(distance), 4)}
        for item_id, item_metadata, distance in zip(ids, metadata, distances)
    ]
