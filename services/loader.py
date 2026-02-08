import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import PyPDF2
from docx import Document
import json
import yaml
import markdown
from bs4 import BeautifulSoup
from pathlib import Path
import csv
import mimetypes

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for cybersecurity documents"""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    extension: str
    num_pages: Optional[int] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    security_level: str = "UNCLASSIFIED"
    document_type: str = "unknown"  # log, report, cve, policy, etc.

class CybersecurityDocumentLoader:
    """Loader for cybersecurity documents with security context awareness"""
    
    def __init__(self, allowed_extensions: set = None):
        self.allowed_extensions = allowed_extensions or {
            '.pdf', '.txt', '.md', '.json', '.csv', '.log', 
            '.xml', '.yaml', '.yml', '.docx', '.html', '.htm'
        }
        # Initialize mimetypes
        mimetypes.init()
        
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a single cybersecurity document with metadata
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file extension
            if file_path.suffix.lower() not in self.allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Get file metadata
            file_size = file_path.stat().st_size
            file_type, _ = mimetypes.guess_type(str(file_path))
            if not file_type:
                file_type = "application/octet-stream"
            
            # Detect document type from content and name
            document_type = self._detect_document_type(file_path)
            
            # Extract security metadata
            metadata = DocumentMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_size,
                file_type=file_type,
                extension=file_path.suffix.lower(),
                security_level=self._detect_security_level(file_path),
                document_type=document_type
            )
            
            # Load content based on file type
            content = self._extract_content(file_path, metadata)
            
            logger.info(f"Loaded document: {file_path.name} | Type: {document_type} | Size: {len(content)} chars")
            
            return {
                "content": content,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            return {
                "content": "",
                "metadata": None,
                "success": False,
                "error": str(e)
            }
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple cybersecurity documents
        """
        results = []
        for file_path in file_paths:
            result = self.load_document(file_path)
            results.append(result)
        return results
    
    def _extract_content(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Extract content based on file type"""
        extension = metadata.extension
        
        if extension == '.pdf':
            return self._extract_pdf_content(file_path, metadata)
        elif extension == '.docx':
            return self._extract_docx_content(file_path, metadata)
        elif extension == '.txt' or extension == '.log':
            return self._extract_text_content(file_path)
        elif extension == '.md':
            return self._extract_markdown_content(file_path)
        elif extension == '.json':
            return self._extract_json_content(file_path)
        elif extension == '.csv':
            return self._extract_csv_content(file_path)
        elif extension in ['.yaml', '.yml']:
            return self._extract_yaml_content(file_path)
        elif extension in ['.xml', '.html', '.htm']:
            return self._extract_xml_content(file_path)
        else:
            # Try fallback
            return self._extract_text_content(file_path)
    
    def _extract_pdf_content(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Extract text from PDF with cybersecurity context"""
        content = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata.num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            # Add page context for citations
                            text = f"[Page {page_num}] {text}"
                            content.append(text)
                    except Exception as page_error:
                        logger.warning(f"Error extracting page {page_num}: {page_error}")
                
                # Try to extract metadata
                if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                    if '/Author' in pdf_reader.metadata:
                        metadata.author = str(pdf_reader.metadata['/Author'])
                    if '/CreationDate' in pdf_reader.metadata:
                        metadata.created_date = str(pdf_reader.metadata['/CreationDate'])
                
            return "\n\n".join(content) if content else ""
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def _extract_docx_content(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Extract text from DOCX"""
        try:
            doc = Document(file_path)
            content = []
            for para in doc.paragraphs:
                if para.text and para.text.strip():
                    content.append(para.text)
            return "\n".join(content) if content else ""
        except Exception as e:
            logger.error(f"DOCX extraction error: {str(e)}")
            return ""
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                return content if content else ""
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                        return file.read()
                except:
                    continue
            return ""
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
            return ""
    
    def _extract_markdown_content(self, file_path: Path) -> str:
        """Extract text from markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                if not md_content:
                    return ""
                # Convert to plain text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"Markdown extraction error: {str(e)}")
            return self._extract_text_content(file_path)
    
    def _extract_json_content(self, file_path: Path) -> str:
        """Extract text from JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to readable text
                return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {file_path}")
            return self._extract_text_content(file_path)
        except Exception as e:
            logger.error(f"JSON extraction error: {str(e)}")
            return self._extract_text_content(file_path)
    
    def _extract_csv_content(self, file_path: Path) -> str:
        """Extract text from CSV"""
        try:
            content = []
            with open(file_path, 'r', encoding='utf-8', newline='') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if row:  # Skip empty rows
                        content.append(', '.join(row))
            return '\n'.join(content) if content else ""
        except Exception as e:
            logger.error(f"CSV extraction error: {str(e)}")
            return self._extract_text_content(file_path)
    
    def _extract_yaml_content(self, file_path: Path) -> str:
        """Extract text from YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                if data is None:
                    return ""
                return yaml.dump(data, default_flow_style=False)
        except yaml.YAMLError:
            logger.error(f"Invalid YAML in {file_path}")
            return self._extract_text_content(file_path)
        except Exception as e:
            logger.error(f"YAML extraction error: {str(e)}")
            return self._extract_text_content(file_path)
    
    def _extract_xml_content(self, file_path: Path) -> str:
        """Extract text from XML/HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content:
                    return ""
                soup = BeautifulSoup(content, 'lxml')
                return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"XML extraction error: {str(e)}")
            return self._extract_text_content(file_path)
    
    def _detect_document_type(self, file_path: Path) -> str:
        """Detect cybersecurity document type"""
        filename = file_path.name.lower()
        
        # Security log files
        log_indicators = ['access', 'error', 'audit', 'security', 'syslog', 'firewall', 'auth', 'login']
        if any(log_ext in filename for log_ext in log_indicators):
            return "security_log"
        elif filename.endswith('.log'):
            return "log_file"
        
        # CVE and vulnerability reports
        if 'cve' in filename or 'vuln' in filename or 'exploit' in filename:
            return "vulnerability_report"
        
        # Incident reports
        if 'incident' in filename or 'breach' in filename or 'response' in filename:
            return "incident_report"
        
        # Policy documents
        policy_indicators = ['policy', 'procedure', 'standard', 'guideline', 'sop', 'control']
        if any(policy_word in filename for policy_word in policy_indicators):
            return "security_policy"
        
        # Threat intelligence
        threat_indicators = ['threat', 'intel', 'ioc', 'ttp', 'malware', 'ransomware', 'phishing']
        if any(ti_word in filename for ti_word in threat_indicators):
            return "threat_intelligence"
        
        # Compliance
        compliance_indicators = ['iso', 'nist', 'gdpr', 'hipaa', 'pci', 'compliance', 'soc2', 'fedramp']
        if any(comp_word in filename for comp_word in compliance_indicators):
            return "compliance_document"
        
        # Network scans
        scan_indicators = ['nmap', 'scan', 'nessus', 'qualys', 'openvas', 'vulnerability', 'pentest']
        if any(scan_word in filename for scan_word in scan_indicators):
            return "scan_report"
        
        # Based on extension
        if file_path.suffix == '.pdf':
            return "report_pdf"
        elif file_path.suffix == '.md':
            return "markdown_doc"
        elif file_path.suffix == '.json':
            return "json_data"
        elif file_path.suffix == '.csv':
            return "csv_data"
        elif file_path.suffix == '.xml':
            return "xml_data"
        
        return "general_document"
    
    def _detect_security_level(self, file_path: Path) -> str:
        """Detect document security level"""
        filename = file_path.name.lower()
        
        # First check filename
        classification_indicators = {
            "TOP SECRET": ["top secret", "ts//", "ts/si", "ts_si"],
            "SECRET": ["secret", "s//", "s/si", "s_si"],
            "CONFIDENTIAL": ["confidential", "c//", "c/si", "c_si"],
            "FOUO": ["fouo", "for official use only"],
            "LIMITED DISTRIBUTION": ["limited distribution", "limited dist", "ld"],
            "PROPRIETARY": ["proprietary", "company confidential", "confidential proprietary"],
            "INTERNAL ONLY": ["internal only", "internal use", "company internal"]
        }
        
        for level, indicators in classification_indicators.items():
            for indicator in indicators:
                if indicator in filename:
                    return level
        
        # Then check first 1000 characters of content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content_preview = file.read(1000).lower()
                
                for level, indicators in classification_indicators.items():
                    for indicator in indicators:
                        if indicator in content_preview:
                            return level
        except:
            pass
        
        # Check for sensitive keywords
        sensitive_keywords = [
            'password', 'credential', 'ssh key', 'api key', 
            'token', 'secret key', 'private key', 'encryption key'
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content_preview = file.read(500).lower()
                if any(keyword in content_preview for keyword in sensitive_keywords):
                    return "RESTRICTED"
        except:
            pass
        
        return "UNCLASSIFIED"