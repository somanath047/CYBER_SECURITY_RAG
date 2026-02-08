import logging
import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

from config import Config
from services.loader import CybersecurityDocumentLoader, DocumentMetadata
from services.chunker import CybersecurityChunker, Chunk
from services.embeddings import CybersecurityEmbeddingManager
from services.retriever import CybersecurityRetriever

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structured response from RAG pipeline"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_time: float
    total_chunks_considered: int
    security_context: str
    recommendations: List[str]
    limitations: List[str]

class RAGPipeline:
    """Main RAG pipeline for cybersecurity analysis"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize components
        self.loader = CybersecurityDocumentLoader(
            allowed_extensions=self.config.ALLOWED_FILE_EXTENSIONS
        )
        
        self.chunker = CybersecurityChunker(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        self.embedding_manager = CybersecurityEmbeddingManager(
            model_name=self.config.EMBEDDING_MODEL,
            cache_dir=os.path.join(self.config.DATA_DIR, "embedding_cache")
        )
        
        self.retriever = CybersecurityRetriever(
            vector_store_path=self.config.VECTOR_STORE_DIR,
            embedding_dim=self.config.EMBEDDING_DIMENSION,
            similarity_top_k=self.config.SIMILARITY_TOP_K,
            similarity_threshold=self.config.SIMILARITY_THRESHOLD,
            enable_hybrid_search=self.config.ENABLE_HYBRID_SEARCH,
            enable_reranking=self.config.ENABLE_RERANKING
        )
        
        # LLM client (simplified - in production, integrate with actual LLM)
        self.llm_provider = self.config.LLM_PROVIDER
        
        # Pipeline statistics
        self.statistics = {
            "documents_processed": 0,
            "total_chunks": 0,
            "queries_processed": 0,
            "avg_query_time": 0,
            "last_ingestion": None,
            "chunks_by_context": {}
        }
        
        # Security context templates
        self.security_prompts = {
            "log_analysis": self._get_log_analysis_prompt(),
            "vulnerability": self._get_vulnerability_prompt(),
            "incident_response": self._get_incident_response_prompt(),
            "threat_intelligence": self._get_threat_intel_prompt(),
            "policy": self._get_policy_prompt(),
            "general": self._get_general_security_prompt()
        }
        
        logger.info("Cybersecurity RAG Pipeline initialized")
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest cybersecurity documents into the RAG system
        """
        results = {
            "successful": [],
            "failed": [],
            "total_files": len(file_paths),
            "chunks_created": 0
        }
        
        for file_path in file_paths:
            try:
                logger.info(f"Ingesting document: {file_path}")
                
                # Load document
                load_result = self.loader.load_document(file_path)
                
                if not load_result["success"]:
                    results["failed"].append({
                        "file": file_path,
                        "error": load_result.get("error", "Unknown error")
                    })
                    continue
                
                # Chunk document
                chunks = self.chunker.chunk_document(
                    load_result["content"],
                    asdict(load_result["metadata"])
                )
                
                if not chunks:
                    logger.warning(f"No chunks created for {file_path}")
                    results["failed"].append({
                        "file": file_path,
                        "error": "No chunks created"
                    })
                    continue
                
                # Convert chunks to dict format
                chunk_dicts = [asdict(chunk) for chunk in chunks]
                
                # Generate embeddings
                embedded_chunks = self.embedding_manager.embed_chunks(chunk_dicts)
                
                # Extract embeddings
                embeddings = []
                valid_chunks = []
                for chunk in embedded_chunks:
                    if 'embedding' in chunk:
                        embeddings.append(np.array(chunk['embedding']))
                        valid_chunks.append(chunk)
                
                # Add to retriever
                if embeddings:
                    self.retriever.add_chunks(valid_chunks, embeddings)
                    
                    # Update statistics
                    self.statistics["documents_processed"] += 1
                    self.statistics["total_chunks"] += len(valid_chunks)
                    results["chunks_created"] += len(valid_chunks)
                    
                    # Update context statistics
                    for chunk in valid_chunks:
                        context = chunk.get('security_context', 'general')
                        self.statistics["chunks_by_context"][context] = \
                            self.statistics["chunks_by_context"].get(context, 0) + 1
                    
                    results["successful"].append({
                        "file": file_path,
                        "chunks": len(valid_chunks),
                        "document_type": load_result["metadata"].document_type,
                        "security_level": load_result["metadata"].security_level
                    })
                    
                    logger.info(f"Successfully ingested {file_path} with {len(valid_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {str(e)}")
                results["failed"].append({
                    "file": file_path,
                    "error": str(e)
                })
        
        self.statistics["last_ingestion"] = datetime.now().isoformat()
        return results
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a cybersecurity query through the RAG pipeline
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.embed_text(question)
            
            # Step 2: Detect security context of query
            security_context = self._detect_query_context(question)
            
            # Step 3: Retrieve relevant chunks
            filters = self._generate_filters_for_context(security_context)
            retrieved_chunks = self.retriever.retrieve(
                question, query_embedding, filters
            )
            
            # Step 4: Prepare context for generation
            context_text = self._prepare_context(retrieved_chunks, question)
            
            # Step 5: Generate answer using LLM
            answer, sources = self._generate_answer(
                question, context_text, retrieved_chunks, security_context
            )
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(retrieved_chunks, answer)
            
            # Step 7: Generate recommendations and limitations
            recommendations = self._generate_recommendations(
                retrieved_chunks, security_context
            )
            limitations = self._generate_limitations(retrieved_chunks)
            
            # Calculate processing time
            query_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._update_query_statistics(query_time)
            
            # Prepare response
            response = RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                query_time=query_time,
                total_chunks_considered=len(retrieved_chunks),
                security_context=security_context,
                recommendations=recommendations,
                limitations=limitations
            )
            
            logger.info(f"Query processed in {query_time:.2f}s with confidence {confidence:.2f}")
            
            return asdict(response)
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            
            # Return error response
            return {
                "answer": f"I encountered an error while processing your security query: {str(e)}\n\nPlease try rephrasing your question or contact your security administrator.",
                "sources": [],
                "confidence": 0.0,
                "query_time": (datetime.now() - start_time).total_seconds(),
                "total_chunks_considered": 0,
                "security_context": "error",
                "recommendations": ["Check system logs for details", "Verify document corpus integrity"],
                "limitations": ["System error prevented complete analysis"]
            }
    
    def _detect_query_context(self, question: str) -> str:
        """Detect the security context of a query"""
        question_lower = question.lower()
        
        # Check for context indicators
        context_indicators = {
            "log_analysis": [
                "log", "audit", "event", "timestamp", "failed login",
                "access denied", "error", "warning", "syslog"
            ],
            "vulnerability": [
                "cve", "vulnerability", "exploit", "patch", "fix",
                "cvss", "severity", "affected", "mitigation"
            ],
            "incident_response": [
                "incident", "breach", "response", "contain", "eradicate",
                "lessons learned", "timeline", "affected systems"
            ],
            "threat_intelligence": [
                "threat", "ioc", "ttp", "malware", "ransomware",
                "phishing", "actor", "campaign", "intelligence"
            ],
            "policy": [
                "policy", "procedure", "compliance", "standard",
                "guideline", "requirement", "mandatory", "must"
            ]
        }
        
        for context, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    return context
        
        return "general"
    
    def _generate_filters_for_context(self, context: str) -> Dict[str, Any]:
        """Generate filters based on security context"""
        filters = {}
        
        if context == "log_analysis":
            filters["document_type"] = "security_log"
        elif context == "vulnerability":
            filters["security_context"] = "vulnerability"
        elif context == "incident_response":
            filters["security_context"] = "incident_response"
        elif context == "threat_intelligence":
            filters["security_context"] = "threat_intelligence"
        elif context == "policy":
            filters["document_type"] = "security_policy"
        
        return filters
    
    def _prepare_context(
        self, 
        chunks: List[Dict[str, Any]], 
        question: str
    ) -> str:
        """Prepare context from retrieved chunks for LLM generation"""
        if not chunks:
            return "No relevant cybersecurity documents found."
        
        context_parts = []
        
        # Add security context header
        security_context = chunks[0].get('security_context', 'general')
        context_parts.append(f"SECURITY CONTEXT: {security_context.upper()}")
        context_parts.append("=" * 50)
        
        # Add each chunk with metadata
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_name', 'Unknown')
            doc_type = metadata.get('document_type', 'document')
            security_level = metadata.get('security_level', 'UNCLASSIFIED')
            
            context_parts.append(f"\n[SOURCE {i+1}]")
            context_parts.append(f"Document: {source}")
            context_parts.append(f"Type: {doc_type}")
            context_parts.append(f"Security Level: {security_level}")
            context_parts.append(f"Relevance Score: {chunk.get('similarity_score', 0):.3f}")
            context_parts.append("-" * 30)
            context_parts.append(chunk.get('text', ''))
        
        # Add query for context
        context_parts.append("\n" + "=" * 50)
        context_parts.append(f"SECURITY ANALYST QUERY: {question}")
        context_parts.append("=" * 50)
        
        return "\n".join(context_parts)
    
    def _generate_answer(
        self, 
        question: str, 
        context: str, 
        chunks: List[Dict[str, Any]],
        security_context: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Generate answer using LLM with cybersecurity focus"""
        
        # Get appropriate prompt template
        prompt_template = self.security_prompts.get(
            security_context, 
            self.security_prompts["general"]
        )
        
        # Format prompt
        prompt = prompt_template.format(
            context=context,
            question=question,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # In production, this would call an actual LLM API
        # For this implementation, we'll generate a simulated response
        answer = self._simulate_llm_response(prompt, security_context)
        
        # Extract sources from chunks
        sources = []
        for chunk in chunks[:3]:  # Top 3 sources
            metadata = chunk.get('metadata', {})
            sources.append({
                "document": metadata.get('file_name', 'Unknown'),
                "type": metadata.get('document_type', 'document'),
                "security_level": metadata.get('security_level', 'UNCLASSIFIED'),
                "relevance_score": chunk.get('similarity_score', 0),
                "context": chunk.get('security_context', 'general')
            })
        
        return answer, sources
    
    def _simulate_llm_response(self, prompt: str, context: str) -> str:
        """Simulate LLM response for demonstration"""
        # In production, replace with actual LLM call
        
        if context == "log_analysis":
            return """BASED ON SECURITY LOG ANALYSIS:

1. **Pattern Detected**: Multiple failed authentication attempts from IP 192.168.1.105 between 02:15-02:30 UTC.

2. **Security Implications**: 
   - Possible brute force attack targeting user accounts
   - Source IP is internal (potential compromised internal host)
   - Attack occurred during low-activity hours

3. **Recommended Actions**:
   - [IMMEDIATE] Block IP 192.168.1.105 at firewall
   - Review account lockout policies
   - Check host 192.168.1.105 for compromise indicators
   - Enable additional logging for authentication events

4. **Next Steps**:
   - Correlate with other log sources (IDS, firewall)
   - Check for successful logins post-attempt period
   - Review user accounts targeted for suspicious activity

**Confidence**: High - Clear pattern of malicious activity in logs."""
        
        elif context == "vulnerability":
            return """VULNERABILITY ASSESSMENT:

**CVE-2023-12345: Remote Code Execution in ExampleService**

**Severity**: CRITICAL (CVSS 9.8)
**Affected Systems**: ExampleService versions 2.0.0 through 2.5.1

**Vulnerability Details**:
- Unauthenticated remote code execution via crafted HTTP requests
- Allows complete system compromise
- Exploit code publicly available

**Impact Assessment**:
- Direct risk to confidentiality, integrity, and availability
- Can lead to full domain compromise in Active Directory environments
- Known exploitation in the wild

**Mitigation Actions**:
1. **IMMEDIATE**: Apply patch version 2.5.2 or later
2. **INTERIM**: Restrict network access to ExampleService (ports 8080/TCP)
3. **DETECTION**: Monitor for exploit patterns in network traffic

**Patch Verification**:
- Verify service version after patching
- Test critical functionality
- Monitor for stability issues

**Note**: Unpatched systems should be considered compromised."""
        
        elif context == "incident_response":
            return """INCIDENT RESPONSE ANALYSIS:

**Incident Timeline Reconstruction**:

1. **Initial Compromise (T0)**: Phishing email delivered to finance@company.com
2. **Execution (T0+2h)**: Malicious macro executed, establishing C2 channel
3. **Lateral Movement (T0+6h)**: Credential harvesting via Mimikatz
4. **Data Exfiltration (T0+12h)**: 2.5GB of financial data transferred externally

**Containment Status**: PARTIAL
- C2 communication blocked at firewall
- Compromised hosts isolated
- User accounts reset

**Critical Gaps Identified**:
1. Missing endpoint detection on finance workstations
2. Delayed alerting (12-hour detection time)
3. Inadequate email filtering for financial department

**Immediate Response Requirements**:
1. Complete forensic imaging of affected systems
2. Reset ALL domain credentials (priority: privileged accounts)
3. Notify legal/compliance per data breach requirements
4. Activate incident response retainer with external forensics firm

**Lessons for Future Incidents**:
- Implement stricter email filtering for financial data handlers
- Reduce detection time via improved monitoring
- Conduct tabletop exercises quarterly"""
        
        else:
            return """CYBERSECURITY ANALYSIS RESULTS:

Based on the available security documentation and context provided, here is the assessment:

**Key Findings**:
1. Security controls appear to be functioning within expected parameters
2. No critical vulnerabilities or active threats detected in the provided context
3. Standard security protocols and procedures are referenced appropriately

**Security Posture Assessment**: STABLE

**Recommendations**:
1. Continue regular security monitoring and log review
2. Ensure all systems are patched to current security levels
3. Maintain incident response readiness through regular testing

**Note**: For more specific security analysis, please provide additional context such as:
- Specific log entries or error messages
- CVE identifiers or vulnerability details
- Incident timelines or indicators of compromise
- Security policy questions or compliance requirements

**Confidence**: Moderate - Limited specific context available for precise analysis."""
    
    def _calculate_confidence(
        self, 
        chunks: List[Dict[str, Any]], 
        answer: str
    ) -> float:
        """Calculate confidence score for the answer"""
        if not chunks:
            return 0.0
        
        # Base confidence on top chunk similarity
        top_scores = [chunk.get('similarity_score', 0) for chunk in chunks[:3]]
        avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        # Adjust based on number of relevant chunks
        num_relevant = sum(1 for chunk in chunks if chunk.get('similarity_score', 0) > 0.7)
        relevance_factor = min(num_relevant / 3, 1.0)  # Cap at 3 relevant chunks
        
        # Adjust based on answer quality (simplified)
        answer_length = len(answer.split())
        length_factor = min(answer_length / 100, 1.0)  # Prefer substantial answers
        
        confidence = avg_top_score * 0.5 + relevance_factor * 0.3 + length_factor * 0.2
        return min(confidence, 1.0)
    
    def _generate_recommendations(
        self, 
        chunks: List[Dict[str, Any]], 
        context: str
    ) -> List[str]:
        """Generate security recommendations based on retrieved chunks"""
        recommendations = []
        
        if not chunks:
            recommendations.append("No specific recommendations available without relevant security context.")
            return recommendations
        
        # Context-specific recommendations
        if context == "log_analysis":
            recommendations.extend([
                "Review authentication logs for suspicious patterns",
                "Check for correlated events across different log sources",
                "Verify log retention policies are being followed"
            ])
        elif context == "vulnerability":
            recommendations.extend([
                "Apply security patches within mandated timeframes",
                "Verify patch installation and system integrity",
                "Monitor for exploitation attempts post-patching"
            ])
        elif context == "incident_response":
            recommendations.extend([
                "Document all containment actions taken",
                "Preserve forensic evidence from affected systems",
                "Update incident response playbooks with lessons learned"
            ])
        else:
            recommendations.append("Maintain regular security reviews and updates")
        
        # Add general recommendations
        recommendations.extend([
            "Ensure security controls are properly configured and monitored",
            "Maintain up-to-date documentation of security procedures",
            "Conduct regular security awareness training"
        ])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _generate_limitations(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate limitations of the analysis"""
        limitations = []
        
        if not chunks:
            limitations.append("No relevant security documents were found for analysis.")
            limitations.append("The answer is based on general security knowledge only.")
            return limitations
        
        # Check for recency
        recent_chunks = 0
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if '2023' in str(metadata) or '2024' in str(metadata):
                recent_chunks += 1
        
        if recent_chunks < len(chunks) / 2:
            limitations.append("Some source documents may be outdated.")
        
        # Check for diversity of sources
        sources = set()
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            sources.add(metadata.get('file_name', 'Unknown'))
        
        if len(sources) < 2:
            limitations.append("Analysis based on limited source diversity.")
        
        # Always include these limitations
        limitations.extend([
            "Automated analysis should be verified by security professionals",
            "Contextual factors may affect the applicability of recommendations",
            "Real-time threat intelligence may provide additional insights"
        ])
        
        return limitations[:5]
    
    def _update_query_statistics(self, query_time: float):
        """Update query processing statistics"""
        self.statistics["queries_processed"] += 1
        
        # Update average query time
        current_avg = self.statistics["avg_query_time"]
        num_queries = self.statistics["queries_processed"]
        
        self.statistics["avg_query_time"] = (
            (current_avg * (num_queries - 1) + query_time) / num_queries
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self.statistics,
            "retriever_stats": self.retriever.get_statistics(),
            "embedding_cache_stats": self.embedding_manager.get_cache_stats(),
            "current_config": {
                "chunk_size": self.config.CHUNK_SIZE,
                "similarity_top_k": self.config.SIMILARITY_TOP_K,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "llm_provider": self.config.LLM_PROVIDER
            }
        }
    
    def _get_log_analysis_prompt(self) -> str:
        return """You are a senior Security Operations Center (SOC) analyst analyzing security logs.

CONTEXT PROVIDED:
{context}

ANALYST QUERY:
{question}

CURRENT DATE: {current_date}

INSTRUCTIONS:
1. Analyze the log data provided in context
2. Identify security events, anomalies, or patterns
3. Provide specific timestamps, IP addresses, usernames when available
4. Assess the severity of findings
5. Recommend immediate and follow-up actions
6. Specify confidence level in findings

FORMAT YOUR RESPONSE AS:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. KEY FINDINGS (bulleted list with evidence)
3. SECURITY ASSESSMENT (impact analysis)
4. RECOMMENDED ACTIONS (prioritized)
5. NEXT STEPS FOR INVESTIGATION
6. CONFIDENCE LEVEL (Low/Medium/High with justification)

IMPORTANT: Cite specific log entries or patterns that support your analysis. If insufficient data exists, state what additional information is needed."""

    def _get_vulnerability_prompt(self) -> str:
        return """You are a vulnerability management specialist assessing security vulnerabilities.

CONTEXT PROVIDED:
{context}

ANALYST QUERY:
{question}

CURRENT DATE: {current_date}

INSTRUCTIONS:
1. Identify all vulnerability references (CVEs, weaknesses)
2. Assess severity using CVSS or provided scoring
3. Describe potential impact if exploited
4. List affected systems/versions
5. Provide mitigation strategies (patches, workarounds)
6. Recommend detection methods for exploitation attempts

FORMAT YOUR RESPONSE AS:
1. VULNERABILITY SUMMARY (CVE, severity, affected systems)
2. TECHNICAL DETAILS (vector, complexity, privileges required)
3. IMPACT ANALYSIS (confidentiality, integrity, availability)
4. MITIGATION ACTIONS (immediate, short-term, long-term)
5. DETECTION & MONITORING (signatures, indicators)
6. PATCH VERIFICATION STEPS

IMPORTANT: Reference specific CVE IDs and version numbers. Prioritize critical vulnerabilities. Include exploit availability status if known."""

    def _get_incident_response_prompt(self) -> str:
        return """You are an incident response commander managing a security incident.

CONTEXT PROVIDED:
{context}

ANALYST QUERY:
{question}

CURRENT DATE: {current_date}

INSTRUCTIONS:
1. Reconstruct incident timeline from available data
2. Identify containment status and effectiveness
3. List compromised systems and data
4. Assess business impact
5. Recommend eradication and recovery steps
6. Identify lessons learned and process improvements

FORMAT YOUR RESPONSE AS:
1. INCIDENT STATUS (ongoing/contained/resolved)
2. TIMELINE RECONSTRUCTION (with evidence)
3. IMPACT ASSESSMENT (systems, data, business)
4. CONTAINMENT EFFECTIVENESS
5. CRITICAL GAPS IDENTIFIED
6. IMMEDIATE RESPONSE REQUIREMENTS
7. LONG-TERM RECOMMENDATIONS

IMPORTANT: Use military time (UTC). Specify evidence sources. Differentiate between confirmed facts and assumptions. Prioritize life/safety if relevant."""

    def _get_threat_intel_prompt(self) -> str:
        return """You are a threat intelligence analyst tracking adversarial campaigns.

CONTEXT PROVIDED:
{context}

ANALYST QUERY:
{question}

CURRENT DATE: {current_date}

INSTRUCTIONS:
1. Identify threat actors, tools, techniques, procedures (TTPs)
2. List indicators of compromise (IOCs)
3. Assess threat relevance to organization
4. Recommend detection rules
5. Provide mitigation strategies
6. Suggest intelligence requirements for further collection

FORMAT YOUR RESPONSE AS:
1. THREAT ASSESSMENT (actor, motivation, capability)
2. TTP ANALYSIS (techniques observed)
3. IOCs (hashes, IPs, domains, patterns)
4. DETECTION SIGNATURES (YARA, Sigma, Snort)
5. MITIGATION ACTIONS (technical, procedural)
6. INTELLIGENCE GAPS & COLLECTION PRIORITIES

IMPORTANT: Include confidence levels for attribution. Separate confirmed IOCs from suspected. Reference MITRE ATT&CK framework when applicable."""

    def _get_policy_prompt(self) -> str:
        return """You are a security policy and compliance analyst.

CONTEXT PROVIDED:
{context}

ANALYST QUERY:
{question}

CURRENT DATE: {current_date}

INSTRUCTIONS:
1. Identify relevant security policies, standards, procedures
2. Map requirements to compliance frameworks (if applicable)
3. Assess policy implementation effectiveness
4. Identify compliance gaps
5. Recommend policy updates or exceptions
6. Provide audit evidence requirements

FORMAT YOUR RESPONSE AS:
1. POLICY REFERENCES (specific clauses, versions)
2. COMPLIANCE MAPPING (framework, control IDs)
3. IMPLEMENTATION ASSESSMENT
4. IDENTIFIED GAPS & RISKS
5. REMEDIATION RECOMMENDATIONS
6. AUDIT PREPARATION GUIDANCE

IMPORTANT: Cite specific policy section numbers. Differentiate between mandatory requirements and best practices. Include exception process if gaps cannot be immediately remediated."""

    def _get_general_security_prompt(self) -> str:
        return """You are a senior cybersecurity consultant providing expert analysis.

CONTEXT PROVIDED:
{context}

ANALYST QUERY:
{question}

CURRENT DATE: {current_date}

INSTRUCTIONS:
1. Analyze the security question based on provided context
2. Apply cybersecurity best practices and frameworks
3. Consider defense-in-depth principles
4. Balance security requirements with business needs
5. Provide actionable, prioritized recommendations
6. Specify when expert consultation is recommended

FORMAT YOUR RESPONSE AS:
1. SECURITY ASSESSMENT (overall posture)
2. KEY FINDINGS (strengths and weaknesses)
3. RISK ANALYSIS (likelihood, impact)
4. RECOMMENDED ACTIONS (prioritized)
5. BUSINESS CONSIDERATIONS
6. EXPERTISE REQUIREMENTS (when to engage specialists)

IMPORTANT: Maintain professional, clear language. Acknowledge limitations of automated analysis. Recommend escalation paths for complex issues."""