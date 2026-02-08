import logging
import numpy as np
from typing import List, Dict, Any, Optional
import pickle
import hashlib
import os
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingCacheEntry:
    """Entry in the embedding cache"""
    text: str
    embedding: np.ndarray
    timestamp: datetime
    model_name: str
    dimension: int

class CybersecurityEmbeddingManager:
    """Manager for cybersecurity-focused embeddings with caching"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "data/embeddings_cache",
        device: str = None
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize cache
        self.cache: Dict[str, EmbeddingCacheEntry] = {}
        self.cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
        self._load_cache()
        
        # Cybersecurity-specific embedding configurations
        self.security_context_weights = {
            "log_analysis": 1.2,  # Weight log entries higher
            "vulnerability": 1.3,  # Weight vulnerability info higher
            "incident_response": 1.2,
            "threat_intelligence": 1.3,
            "policy": 1.1,
            "general": 1.0
        }
        
        # Special cybersecurity terms to boost
        self.security_keywords = {
            "terms": [
                "cve", "exploit", "vulnerability", "malware", "ransomware",
                "phishing", "breach", "intrusion", "firewall", "ids", "ips",
                "siem", "soc", "incident", "response", "forensics",
                "authentication", "authorization", "encryption", "tls", "ssl",
                "zero-day", "patch", "update", "mitigation", "remediation",
                "compliance", "gdpr", "hipaa", "pci", "iso27001", "nist",
                "log", "audit", "monitoring", "detection", "prevention"
            ],
            "boost_factor": 1.5
        }
        
        logger.info(f"Embedding manager initialized with model: {model_name}")
    
    def embed_text(self, text: str, security_context: str = "general") -> np.ndarray:
        """
        Generate embedding for text with cybersecurity context awareness
        """
        # Check cache first
        cache_key = self._generate_cache_key(text, security_context)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for: {text[:50]}...")
            return self.cache[cache_key].embedding
        
        # Generate embedding
        try:
            # Basic embedding
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Apply security context weighting
            weight = self.security_context_weights.get(security_context, 1.0)
            embedding = embedding * weight
            
            # Normalize after weighting
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache the result
            cache_entry = EmbeddingCacheEntry(
                text=text,
                embedding=embedding,
                timestamp=datetime.now(),
                model_name=self.model_name,
                dimension=len(embedding)
            )
            self.cache[cache_key] = cache_entry
            
            # Periodic cache save
            if len(self.cache) % 100 == 0:
                self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def embed_batch(
        self, 
        texts: List[str], 
        security_contexts: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        """
        if security_contexts is None:
            security_contexts = ["general"] * len(texts)
        
        # Separate cached and non-cached texts
        cached_embeddings = []
        non_cached_texts = []
        non_cached_contexts = []
        non_cached_indices = []
        
        for i, (text, context) in enumerate(zip(texts, security_contexts)):
            cache_key = self._generate_cache_key(text, context)
            if cache_key in self.cache:
                cached_embeddings.append(self.cache[cache_key].embedding)
            else:
                non_cached_texts.append(text)
                non_cached_contexts.append(context)
                non_cached_indices.append(i)
        
        # Generate embeddings for non-cached texts
        if non_cached_texts:
            logger.info(f"Generating embeddings for {len(non_cached_texts)} new texts")
            
            # Batch encode
            new_embeddings = self.model.encode(
                non_cached_texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Apply context weighting and cache
            for idx, (text, context, embedding) in enumerate(zip(
                non_cached_texts, non_cached_contexts, new_embeddings
            )):
                weight = self.security_context_weights.get(context, 1.0)
                embedding = embedding * weight
                embedding = embedding / np.linalg.norm(embedding)
                
                # Cache
                cache_key = self._generate_cache_key(text, context)
                cache_entry = EmbeddingCacheEntry(
                    text=text,
                    embedding=embedding,
                    timestamp=datetime.now(),
                    model_name=self.model_name,
                    dimension=len(embedding)
                )
                self.cache[cache_key] = cache_entry
                
                cached_embeddings.append(embedding)
        
        # Combine all embeddings in original order
        all_embeddings = [None] * len(texts)
        cache_idx = 0
        
        for i in range(len(texts)):
            if i in non_cached_indices:
                # Find the index in non_cached results
                nc_idx = non_cached_indices.index(i)
                # Already added to cached_embeddings in order
                all_embeddings[i] = cached_embeddings[cache_idx]
                cache_idx += 1
            else:
                all_embeddings[i] = cached_embeddings[cache_idx]
                cache_idx += 1
        
        # Save cache after batch operation
        self._save_cache()
        
        return np.array(all_embeddings)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of chunks with their metadata
        """
        texts = [chunk.get('text', '') for chunk in chunks]
        contexts = [chunk.get('security_context', 'general') for chunk in chunks]
        
        embeddings = self.embed_batch(texts, contexts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            chunk['embedding_dimension'] = len(embeddings[i])
            chunk['embedding_model'] = self.model_name
        
        return chunks
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except:
            return 0.0
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: List[np.ndarray],
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find similar chunks to query embedding
        """
        if not chunk_embeddings or not chunks:
            return []
        
        # Compute similarities
        similarities = []
        for embedding in chunk_embeddings:
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append(similarity)
        
        # Get top-k indices
        if len(similarities) <= top_k:
            indices = list(range(len(similarities)))
        else:
            indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by threshold and prepare results
        results = []
        for idx in indices:
            if similarities[idx] >= threshold:
                result = chunks[idx].copy()
                result['similarity_score'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def _generate_cache_key(self, text: str, context: str) -> str:
        """Generate cache key for text and context"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model_name}:{context}:{text_hash}"
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                
                # Filter out old entries (older than 30 days)
                cutoff_date = datetime.now().timestamp() - (30 * 24 * 60 * 60)
                self.cache = {
                    k: v for k, v in loaded_cache.items() 
                    if v.timestamp.timestamp() > cutoff_date
                }
                
                logger.info(f"Loaded {len(self.cache)} embeddings from cache")
            else:
                self.cache = {}
                logger.info("No existing cache found")
                
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = len(self.cache)
        
        # Count by context
        context_counts = {}
        for entry in self.cache.values():
            # Extract context from cache key
            parts = entry.text.split(":") if ":" in entry.text else ["unknown"]
            context = parts[1] if len(parts) > 1 else "general"
            context_counts[context] = context_counts.get(context, 0) + 1
        
        # Estimate memory usage
        avg_dimension = self.model.get_sentence_embedding_dimension()
        estimated_memory = total_size * avg_dimension * 4  # 4 bytes per float32
        
        return {
            "total_entries": total_size,
            "context_distribution": context_counts,
            "estimated_memory_bytes": estimated_memory,
            "cache_file_size": os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        }
    
    def clear_cache(self, older_than_days: int = None):
        """Clear embedding cache"""
        if older_than_days:
            cutoff = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
            initial_count = len(self.cache)
            self.cache = {
                k: v for k, v in self.cache.items() 
                if v.timestamp.timestamp() > cutoff
            }
            logger.info(f"Cleared {initial_count - len(self.cache)} old cache entries")
        else:
            self.cache.clear()
            logger.info("Cleared all cache entries")
        
        self._save_cache()