import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class Chunk:
    """Chunk of text with cybersecurity context"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    embedding: np.ndarray = None
    security_context: str = "general"
    
    def to_dict(self):
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "security_context": self.security_context
        }

class CybersecurityChunker:
    """Advanced chunker optimized for cybersecurity documents"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        security_contexts: Dict[str, List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for cybersecurity documents
        self.separators = separators or [
            "\n\n## ",  # Markdown headers
            "\n\n# ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            " "
        ]
        
        # Cybersecurity-specific context patterns
        self.security_contexts = security_contexts or {
            "log_analysis": [
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # Timestamps
                r"ERROR|WARNING|ALERT|CRITICAL|FAILED",  # Log levels
                r"src=|dst=|proto=|sport=|dport=",  # Network info
            ],
            "vulnerability": [
                r"CVE-\d{4}-\d+",  # CVE IDs
                r"CVSS:\d+\.\d+",  # CVSS scores
                r"severity:\s*(critical|high|medium|low)",
                r"affected versions?:",
                r"mitigation:",
                r"workaround:",
            ],
            "incident_response": [
                r"incident id:",  # Incident IDs
                r"severity level:",
                r"affected systems:",
                r"timeline:",
                r"containment actions:",
                r"eradication steps:",
                r"lessons learned:",
            ],
            "policy": [
                r"policy number:",
                r"effective date:",
                r"review date:",
                r"scope:",
                r"purpose:",
                r"responsibilities:",
                r"compliance:",
            ],
            "threat_intel": [
                r"ttp:\s*",  # TTPs
                r"ioc:\s*",  # IOCs
                r"malware family:",
                r"attack vector:",
                r"attribution:",
            ]
        }
        
        # Semantic chunking model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a cybersecurity document with context-aware splitting
        """
        document_type = metadata.get('document_type', 'general_document')
        
        if document_type == "security_log":
            return self._chunk_log_file(content, metadata)
        elif document_type == "vulnerability_report":
            return self._chunk_vulnerability_report(content, metadata)
        elif document_type == "incident_report":
            return self._chunk_incident_report(content, metadata)
        elif document_type == "security_policy":
            return self._chunk_policy_document(content, metadata)
        else:
            return self._chunk_general_document(content, metadata)
    
    def _chunk_general_document(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk general documents using recursive splitting"""
        chunks = []
        current_chunk = ""
        sentences = sent_tokenize(content)
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # Save current chunk
                if current_chunk:
                    chunk = self._create_chunk(current_chunk, metadata, len(chunks))
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                if len(chunks) > 0:
                    # Take last few sentences from previous chunk for overlap
                    prev_sentences = sent_tokenize(chunks[-1].text)
                    overlap_sentences = prev_sentences[-3:]  # Last 3 sentences
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunk = self._create_chunk(current_chunk, metadata, len(chunks))
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {metadata.get('file_name', 'unknown')}")
        return chunks
    
    def _chunk_log_file(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk security log files preserving log entry boundaries"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_length = len(line)
            
            # Check if this line starts a new logical entry (timestamp pattern)
            is_new_entry = bool(re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', line[:10]))
            
            if is_new_entry and current_length + line_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunk = self._create_chunk(chunk_text, metadata, len(chunks), "log_analysis")
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [line]
                current_length = line_length
            else:
                if current_length + line_length <= self.chunk_size:
                    current_chunk.append(line)
                    current_length += line_length
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        chunk = self._create_chunk(chunk_text, metadata, len(chunks), "log_analysis")
                        chunks.append(chunk)
                    
                    current_chunk = [line]
                    current_length = line_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk = self._create_chunk(chunk_text, metadata, len(chunks), "log_analysis")
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_vulnerability_report(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk vulnerability reports by sections"""
        chunks = []
        
        # Split by common vulnerability report sections
        sections = re.split(r'\n#{1,3}\s+', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Further split large sections
            section_chunks = self._split_by_semantic_boundaries(section, metadata)
            for chunk_text in section_chunks:
                chunk = self._create_chunk(chunk_text, metadata, len(chunks), "vulnerability")
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_incident_report(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk incident reports preserving timeline and actions"""
        chunks = []
        
        # Try to identify structured sections
        section_patterns = {
            "timeline": r"(?i)timeline[:]?\s*\n",
            "containment": r"(?i)containment[:]?\s*\n",
            "eradication": r"(?i)eradication[:]?\s*\n",
            "lessons": r"(?i)lessons learned[:]?\s*\n",
        }
        
        current_section = "overview"
        section_text = ""
        
        lines = content.split('\n')
        for line in lines:
            # Check if line starts a new section
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line):
                    # Save previous section
                    if section_text:
                        section_chunks = self._split_by_semantic_boundaries(
                            section_text, metadata, max_chunk_size=self.chunk_size//2
                        )
                        for chunk_text in section_chunks:
                            chunk = self._create_chunk(
                                chunk_text, metadata, len(chunks), 
                                f"incident_{current_section}"
                            )
                            chunks.append(chunk)
                    
                    # Start new section
                    current_section = section_name
                    section_text = line + "\n"
                    section_found = True
                    break
            
            if not section_found:
                section_text += line + "\n"
        
        # Add final section
        if section_text:
            section_chunks = self._split_by_semantic_boundaries(
                section_text, metadata, max_chunk_size=self.chunk_size//2
            )
            for chunk_text in section_chunks:
                chunk = self._create_chunk(
                    chunk_text, metadata, len(chunks), 
                    f"incident_{current_section}"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_policy_document(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk policy documents by clauses and sections"""
        chunks = []
        
        # Split by numbered clauses (e.g., "1.1", "2.3.1")
        clauses = re.split(r'\n\d+\.\d+(?:\.\d+)*\s+', content)
        
        for i, clause in enumerate(clauses):
            if not clause.strip():
                continue
            
            # Add clause number back
            if i > 0:
                clause_number = content.split('\n')[-1].split()[0] if i == 1 else f"{i}. "
                clause = clause_number + clause
            
            clause_chunks = self._split_by_semantic_boundaries(clause, metadata)
            for chunk_text in clause_chunks:
                chunk = self._create_chunk(chunk_text, metadata, len(chunks), "policy")
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_semantic_boundaries(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        max_chunk_size: int = None
    ) -> List[str]:
        """Split text by semantic boundaries using sentence embeddings"""
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size
        
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        # Get sentence embeddings
        sentence_embeddings = self.semantic_model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_length = 0
        current_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
                # Update running average embedding
                if current_embedding is None:
                    current_embedding = embedding
                else:
                    current_embedding = (current_embedding * len(current_chunk) + embedding) / (len(current_chunk) + 1)
            else:
                # Check semantic similarity with next sentences
                if i + 1 < len(sentences):
                    next_embedding = sentence_embeddings[i + 1]
                    similarity = np.dot(current_embedding, next_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(next_embedding)
                    ) if current_embedding is not None else 0
                    
                    # If highly similar, include in current chunk
                    if similarity > 0.8:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                        continue
                
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
                current_embedding = embedding
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _create_chunk(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        chunk_index: int,
        security_context: str = "general"
    ) -> Chunk:
        """Create a chunk with proper metadata"""
        chunk_metadata = {
            **metadata,
            "chunk_index": chunk_index,
            "chunk_length": len(text),
            "char_count": len(text),
            "word_count": len(text.split())
        }
        
        chunk_id = f"{metadata.get('file_name', 'doc')}_chunk_{chunk_index}"
        
        return Chunk(
            text=text.strip(),
            metadata=chunk_metadata,
            chunk_id=chunk_id,
            security_context=security_context
        )
    
    def detect_security_context(self, text: str) -> str:
        """Detect the security context of a text chunk"""
        text_lower = text.lower()
        
        for context, patterns in self.security_contexts.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return context
        
        return "general"