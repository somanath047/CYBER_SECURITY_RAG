import logging
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logger = logging.getLogger(__name__)

class CybersecurityRetriever:
    """Advanced retriever for cybersecurity documents with hybrid search"""
    
    def __init__(
        self,
        vector_store_path: str = "data/vector_store",
        embedding_dim: int = 384,
        similarity_top_k: int = 5,
        similarity_threshold: float = 0.7,
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True
    ):
        self.vector_store_path = vector_store_path
        self.embedding_dim = embedding_dim
        self.similarity_top_k = similarity_top_k
        self.similarity_threshold = similarity_threshold
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        self.chunk_embeddings = []
        
        # TF-IDF for hybrid search
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Metadata store
        self.metadata_store = {}
        
        # Load existing index if available
        self._load_index()
        
        # Cybersecurity-specific query expansion terms
        self.security_query_expansion = {
            "attack": ["exploit", "intrusion", "breach", "compromise", "incursion"],
            "vulnerability": ["weakness", "flaw", "bug", "cve", "exposure"],
            "malware": ["ransomware", "trojan", "virus", "worm", "spyware"],
            "log": ["audit", "event", "record", "trail", "entry"],
            "firewall": ["filter", "gateway", "barrier", "shield"],
            "encryption": ["cipher", "cryptography", "encode", "secure"],
            "authentication": ["auth", "login", "credentials", "verify"],
            "network": ["traffic", "packet", "protocol", "connection"],
            "threat": ["risk", "danger", "hazard", "menace"],
            "incident": ["event", "occurrence", "situation", "case"]
        }
        
        # Security severity boost terms
        self.severity_boost_terms = {
            "critical": 2.0,
            "high": 1.5,
            "medium": 1.2,
            "low": 1.0,
            "emergency": 2.5,
            "urgent": 2.0,
            "important": 1.5
        }
        
        logger.info("Cybersecurity retriever initialized")
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """
        Add chunks and their embeddings to the retriever
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return
        
        # Convert embeddings to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and embeddings
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        self.chunk_embeddings.extend(embeddings)
        
        # Update metadata store
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('chunk_id', f'chunk_{start_idx + i}')
            self.metadata_store[chunk_id] = {
                'index': start_idx + i,
                'metadata': chunk.get('metadata', {}),
                'security_context': chunk.get('security_context', 'general'),
                'text_preview': chunk.get('text', '')[:100]
            }
        
        # Update TF-IDF for hybrid search
        if self.enable_hybrid_search:
            self._update_tfidf(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to retriever. Total chunks: {len(self.chunks)}")
        
        # Save index
        self._save_index()
    
    def retrieve(
        self, 
        query: str, 
        query_embedding: np.ndarray,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a cybersecurity query
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("No chunks indexed for retrieval")
            return []
        
        # Apply query expansion for cybersecurity context
        expanded_queries = self._expand_security_query(query)
        
        # Perform vector search
        vector_results = self._vector_search(query_embedding, filters)
        
        # Perform hybrid search if enabled
        if self.enable_hybrid_search and self.tfidf_matrix is not None:
            keyword_results = self._keyword_search(query, expanded_queries, filters)
            combined_results = self._combine_results(
                vector_results, keyword_results, query_embedding
            )
        else:
            combined_results = vector_results
        
        # Apply re-ranking if enabled
        if self.enable_reranking:
            reranked_results = self._rerank_results(combined_results, query, query_embedding)
        else:
            reranked_results = combined_results
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in reranked_results 
            if result.get('similarity_score', 0) >= self.similarity_threshold
        ]
        
        # Apply severity boosting for cybersecurity
        boosted_results = self._apply_severity_boosting(filtered_results, query)
        
        # Return top-k results
        return boosted_results[:self.similarity_top_k]
    
    def _vector_search(
        self, 
        query_embedding: np.ndarray, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        # Convert to float32 for FAISS
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search FAISS index
        k = min(self.similarity_top_k * 3, len(self.chunks))  # Get more for filtering
        distances, indices = self.index.search(query_embedding, k)
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Apply filters if provided
            if filters and not self._passes_filters(idx, filters):
                continue
            
            chunk = self.chunks[idx]
            similarity = float(distance)  # FAISS returns inner product for IndexFlatIP
            
            results.append({
                **chunk,
                'similarity_score': similarity,
                'retrieval_method': 'vector',
                'rank': i + 1
            })
        
        return results
    
    def _keyword_search(
        self, 
        query: str, 
        expanded_queries: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search using TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        # Combine original and expanded queries
        all_queries = [query] + expanded_queries
        
        # Transform queries to TF-IDF vectors
        query_vectors = self.tfidf_vectorizer.transform(all_queries)
        
        # Compute similarities
        similarities = np.dot(query_vectors, self.tfidf_matrix.T).toarray()
        
        # Get max similarity across all queries for each document
        max_similarities = np.max(similarities, axis=0)
        
        # Get top indices
        k = min(self.similarity_top_k * 3, len(self.chunks))
        top_indices = np.argsort(max_similarities)[-k:][::-1]
        
        # Process results
        results = []
        for rank, idx in enumerate(top_indices):
            if max_similarities[idx] <= 0:
                continue
            
            # Apply filters
            if filters and not self._passes_filters(idx, filters):
                continue
            
            chunk = self.chunks[idx]
            
            results.append({
                **chunk,
                'similarity_score': float(max_similarities[idx]),
                'retrieval_method': 'keyword',
                'rank': rank + 1
            })
        
        return results
    
    def _combine_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]],
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results"""
        # Create a map of chunk_id to result
        combined_map = {}
        
        # Add vector results with higher initial weight
        for result in vector_results:
            chunk_id = result.get('chunk_id')
            combined_map[chunk_id] = {
                'result': result,
                'vector_score': result.get('similarity_score', 0),
                'keyword_score': 0
            }
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result.get('chunk_id')
            if chunk_id in combined_map:
                combined_map[chunk_id]['keyword_score'] = result.get('similarity_score', 0)
            else:
                combined_map[chunk_id] = {
                    'result': result,
                    'vector_score': 0,
                    'keyword_score': result.get('similarity_score', 0)
                }
        
        # Calculate combined scores
        combined_results = []
        for chunk_id, scores in combined_map.items():
            result = scores['result']
            
            # Hybrid scoring: 70% vector, 30% keyword
            vector_weight = 0.7
            keyword_weight = 0.3
            
            # Normalize scores (assuming cosine similarity range)
            vector_score = max(0, scores['vector_score'])
            keyword_score = max(0, scores['keyword_score'])
            
            # Combine scores
            combined_score = (vector_weight * vector_score + 
                            keyword_weight * keyword_score)
            
            # Context relevance boost
            context_boost = self._calculate_context_relevance(
                result, query_embedding
            )
            combined_score *= context_boost
            
            result['similarity_score'] = combined_score
            result['retrieval_method'] = 'hybrid'
            result['vector_score'] = vector_score
            result['keyword_score'] = keyword_score
            
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return combined_results
    
    def _rerank_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Re-rank results using cybersecurity-specific heuristics"""
        if not results:
            return results
        
        reranked = []
        
        for result in results:
            score = result.get('similarity_score', 0)
            
            # 1. Recency boost (for incident reports, logs)
            recency_boost = self._calculate_recency_boost(result)
            score *= recency_boost
            
            # 2. Security context relevance
            context = result.get('security_context', 'general')
            context_relevance = self._calculate_context_relevance(result, query_embedding)
            score *= context_relevance
            
            # 3. Query term density
            term_density = self._calculate_term_density(result['text'], query)
            score *= (1 + term_density * 0.5)  # Up to 50% boost
            
            # 4. Source authority (prioritize official sources)
            authority_boost = self._calculate_authority_boost(result)
            score *= authority_boost
            
            # 5. Chunk coherence (longer, well-structured chunks)
            coherence_boost = self._calculate_coherence_boost(result)
            score *= coherence_boost
            
            result['similarity_score'] = score
            result['reranking_factors'] = {
                'recency_boost': recency_boost,
                'context_relevance': context_relevance,
                'term_density': term_density,
                'authority_boost': authority_boost,
                'coherence_boost': coherence_boost
            }
            
            reranked.append(result)
        
        # Re-sort by new scores
        reranked.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return reranked
    
    def _expand_security_query(self, query: str) -> List[str]:
        """Expand query with cybersecurity synonyms and related terms"""
        expanded = []
        query_lower = query.lower()
        
        # Add original query
        expanded.append(query)
        
        # Add synonyms for security terms
        for term, synonyms in self.security_query_expansion.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded.append(expanded_query)
        
        # Add CVE pattern expansion
        if re.search(r'cve[-\s]?\d{4}[-\s]?\d+', query_lower, re.IGNORECASE):
            # If query mentions CVE, add vulnerability-related terms
            expanded.append(query + " vulnerability")
            expanded.append(query + " exploit")
            expanded.append(query + " patch")
        
        # Add log-related expansions
        if any(log_term in query_lower for log_term in ['log', 'audit', 'event']):
            expanded.append(query + " analysis")
            expanded.append(query + " forensic")
            expanded.append("security " + query)
        
        # Remove duplicates and return
        return list(dict.fromkeys(expanded))[:5]  # Limit to 5 expansions
    
    def _apply_severity_boosting(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Boost results containing severity indicators"""
        query_lower = query.lower()
        
        for result in results:
            text_lower = result['text'].lower()
            score = result.get('similarity_score', 0)
            
            # Check for severity terms in query
            for severity_term, boost_factor in self.severity_boost_terms.items():
                if severity_term in query_lower and severity_term in text_lower:
                    score *= boost_factor
                    break
            
            # Check for emergency/urgency indicators
            urgency_indicators = ['immediately', 'asap', 'urgent', 'emergency', 'critical']
            if any(indicator in query_lower for indicator in urgency_indicators):
                # Boost chunks with recent timestamps or incident context
                if result.get('security_context') == 'incident_response':
                    score *= 1.3
                if 'timestamp' in text_lower or '202' in text_lower:  # Recent year
                    score *= 1.2
            
            result['similarity_score'] = score
        
        # Re-sort after boosting
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def _calculate_recency_boost(self, result: Dict[str, Any]) -> float:
        """Boost recent documents (especially for incidents and logs)"""
        metadata = result.get('metadata', {})
        context = result.get('security_context', '')
        
        # Check for timestamps in text
        text = result.get('text', '')
        
        # Look for recent dates (2020-2024)
        recent_year_pattern = r'(202[0-4])'
        years = re.findall(recent_year_pattern, text)
        
        if years:
            # More recent = higher boost
            max_year = max(map(int, years))
            recency_factor = 1.0 + (max_year - 2020) * 0.1  # 10% per year
            return min(recency_factor, 1.5)  # Cap at 50% boost
        
        # For incident reports and logs, assume recent if no date
        if context in ['incident_response', 'log_analysis']:
            return 1.2
        
        return 1.0
    
    def _calculate_context_relevance(
        self, 
        result: Dict[str, Any], 
        query_embedding: np.ndarray
    ) -> float:
        """Calculate how relevant the security context is to the query"""
        context = result.get('security_context', 'general')
        
        # Context-specific embeddings (simplified approach)
        context_embeddings = {
            'log_analysis': np.array([0.1, 0.8, 0.1]),  # Example
            'vulnerability': np.array([0.8, 0.1, 0.1]),
            'incident_response': np.array([0.6, 0.3, 0.1]),
            'threat_intelligence': np.array([0.7, 0.2, 0.1]),
            'policy': np.array([0.2, 0.1, 0.7]),
            'general': np.array([0.3, 0.3, 0.4])
        }
        
        # In production, you'd use actual context embeddings
        # This is a simplified heuristic
        
        # Boost based on context match with query terms
        text_lower = result['text'].lower()
        context_keywords = {
            'log_analysis': ['log', 'timestamp', 'error', 'failed', 'access'],
            'vulnerability': ['cve', 'vulnerability', 'exploit', 'patch', 'fix'],
            'incident_response': ['incident', 'response', 'contain', 'eradicate'],
            'threat_intelligence': ['threat', 'ioc', 'ttp', 'malware', 'actor'],
            'policy': ['policy', 'procedure', 'compliance', 'standard']
        }
        
        if context in context_keywords:
            keywords = context_keywords[context]
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count > 0:
                return 1.0 + (match_count * 0.1)  # 10% per keyword match
        
        return 1.0
    
    def _calculate_term_density(self, text: str, query: str) -> float:
        """Calculate how densely query terms appear in text"""
        if not query or not text:
            return 0.0
        
        query_terms = set(query.lower().split())
        text_lower = text.lower()
        
        if not query_terms:
            return 0.0
        
        # Count term occurrences
        total_occurrences = 0
        for term in query_terms:
            if len(term) > 2:  # Ignore very short terms
                total_occurrences += text_lower.count(term)
        
        # Calculate density (occurrences per 1000 chars)
        text_length = len(text_lower)
        if text_length == 0:
            return 0.0
        
        density = (total_occurrences / text_length) * 1000
        return min(density, 10.0) / 10.0  # Normalize to 0-1
    
    def _calculate_authority_boost(self, result: Dict[str, Any]) -> float:
        """Boost results from authoritative sources"""
        metadata = result.get('metadata', {})
        source = metadata.get('file_name', '').lower()
        
        # Official sources get boost
        official_indicators = [
            'cisa', 'nist', 'mitre', 'sans', 'cis',
            'iso', 'pci', 'gdpr', 'hipaa',
            'microsoft', 'cisco', 'paloalto', 'fortinet'
        ]
        
        for indicator in official_indicators:
            if indicator in source:
                return 1.3  # 30% boost for official sources
        
        # .gov or .mil domains
        if source.endswith('.gov') or source.endswith('.mil'):
            return 1.4
        
        return 1.0
    
    def _calculate_coherence_boost(self, result: Dict[str, Any]) -> float:
        """Boost well-structured, coherent chunks"""
        text = result.get('text', '')
        
        # Check for proper sentence structure
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.8  # Penalize single-sentence chunks
        
        # Check average sentence length
        avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
        if 20 <= avg_sentence_len <= 100:
            return 1.2  # Good sentence length
        
        # Check for paragraph structure
        if '\n\n' in text:
            return 1.1
        
        return 1.0
    
    def _passes_filters(self, chunk_idx: int, filters: Dict[str, Any]) -> bool:
        """Check if chunk passes all filters"""
        chunk = self.chunks[chunk_idx]
        metadata = chunk.get('metadata', {})
        
        for key, value in filters.items():
            if key == 'security_context':
                if chunk.get('security_context') != value:
                    return False
            elif key == 'document_type':
                if metadata.get('document_type') != value:
                    return False
            elif key == 'security_level':
                if metadata.get('security_level') != value:
                    return False
            elif key in metadata:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _update_tfidf(self, new_chunks: List[Dict[str, Any]]):
        """Update TF-IDF matrix with new chunks"""
        texts = [chunk.get('text', '') for chunk in new_chunks]
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)  # Include bigrams for security terms
            )
            # Initialize with existing chunks if any
            if self.chunks:
                existing_texts = [chunk.get('text', '') for chunk in self.chunks]
                texts = existing_texts + texts
        
        # Fit or update TF-IDF
        all_texts = [chunk.get('text', '') for chunk in self.chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Save FAISS index
            faiss_index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
            faiss.write_index(self.index, faiss_index_path)
            
            # Save chunks and metadata
            chunks_path = os.path.join(self.vector_store_path, "chunks.pkl")
            with open(chunks_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'chunk_embeddings': self.chunk_embeddings,
                    'metadata_store': self.metadata_store
                }, f)
            
            # Save TF-IDF if exists
            if self.tfidf_vectorizer is not None:
                tfidf_path = os.path.join(self.vector_store_path, "tfidf.pkl")
                with open(tfidf_path, 'wb') as f:
                    pickle.dump({
                        'vectorizer': self.tfidf_vectorizer,
                        'matrix': self.tfidf_matrix
                    }, f)
            
            logger.info(f"Saved index with {len(self.chunks)} chunks to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            faiss_index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
            chunks_path = os.path.join(self.vector_store_path, "chunks.pkl")
            
            if os.path.exists(faiss_index_path) and os.path.exists(chunks_path):
                # Load FAISS index
                self.index = faiss.read_index(faiss_index_path)
                
                # Load chunks and metadata
                with open(chunks_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.chunk_embeddings = data['chunk_embeddings']
                    self.metadata_store = data['metadata_store']
                
                # Load TF-IDF if exists
                tfidf_path = os.path.join(self.vector_store_path, "tfidf.pkl")
                if os.path.exists(tfidf_path):
                    with open(tfidf_path, 'rb') as f:
                        tfidf_data = pickle.load(f)
                        self.tfidf_vectorizer = tfidf_data['vectorizer']
                        self.tfidf_matrix = tfidf_data['matrix']
                
                logger.info(f"Loaded index with {len(self.chunks)} chunks from {self.vector_store_path}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dim,
            "index_type": "FAISS FlatIP" if self.index else "None",
            "hybrid_search_enabled": self.enable_hybrid_search,
            "reranking_enabled": self.enable_reranking,
            "chunks_by_context": self._count_chunks_by_context(),
            "average_chunk_size": self._calculate_average_chunk_size()
        }
    
    def _count_chunks_by_context(self) -> Dict[str, int]:
        """Count chunks by security context"""
        counts = {}
        for chunk in self.chunks:
            context = chunk.get('security_context', 'general')
            counts[context] = counts.get(context, 0) + 1
        return counts
    
    def _calculate_average_chunk_size(self) -> Dict[str, float]:
        """Calculate average chunk sizes"""
        if not self.chunks:
            return {"chars": 0, "words": 0}
        
        total_chars = 0
        total_words = 0
        
        for chunk in self.chunks:
            text = chunk.get('text', '')
            total_chars += len(text)
            total_words += len(text.split())
        
        return {
            "chars": total_chars / len(self.chunks),
            "words": total_words / len(self.chunks)
        }