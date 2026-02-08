from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Try to import services, but handle gracefully if they fail
try:
    from services.rag_pipeline import RAGPipeline
    rag_pipeline = RAGPipeline()
    services_loaded = True
    logger.info("RAG pipeline loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to load RAG pipeline: {e}")
    logger.warning("Running in minimal mode")
    services_loaded = False
    rag_pipeline = None
except Exception as e:
    logger.error(f"Error initializing RAG pipeline: {e}")
    services_loaded = False
    rag_pipeline = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy" if services_loaded else "degraded",
        "service": "cybersecurity-rag",
        "version": "1.0.0",
        "mode": "full" if services_loaded else "minimal",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """Ingest cybersecurity documents"""
    if not services_loaded:
        return jsonify({
            "error": "Document ingestion service not available in minimal mode",
            "status": "degraded"
        }), 503
    
    try:
        data = request.json
        file_paths = data.get('file_paths', [])
        
        if not file_paths:
            return jsonify({
                "error": "No file paths provided"
            }), 400
        
        results = rag_pipeline.ingest_documents(file_paths)
        
        logger.info(f"Successfully ingested {len(results['successful'])} documents")
        
        return jsonify({
            "status": "success",
            "message": f"Processed {len(results['successful'])} documents",
            "details": results
        })
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        return jsonify({
            "error": f"Document ingestion failed: {str(e)}"
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Main endpoint for cybersecurity queries"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "error": "No question provided"
            }), 400
        
        logger.info(f"Processing cybersecurity question: {question[:100]}...")
        
        if services_loaded and rag_pipeline:
            # Process through RAG pipeline
            result = rag_pipeline.process_query(question)
            logger.info(f"Generated answer with {len(result['sources'])} sources")
        else:
            # Fallback to minimal response
            result = {
                "answer": f"Security Analysis for: {question}\n\n"
                         f"⚠️ **SYSTEM STATUS**: Running in minimal mode\n"
                         f"📊 **ANALYSIS**: This query would normally retrieve relevant security documents, "
                         f"analyze patterns, and provide actionable intelligence.\n\n"
                         f"🔧 **RECOMMENDATION**: Install all dependencies for full functionality:\n"
                         f"```bash\npip install -r requirements-fixed.txt\n```",
                "sources": [
                    {
                        "document": "system_status.log",
                        "type": "system_log",
                        "security_level": "UNCLASSIFIED",
                        "relevance_score": 0.9,
                        "context": "system"
                    }
                ],
                "confidence": 0.7,
                "query_time": 0.1,
                "total_chunks_considered": 0,
                "security_context": "general",
                "recommendations": [
                    "Install all dependencies for full RAG functionality",
                    "Add security documents to data/documents/ folder",
                    "Check logs for specific dependency errors"
                ],
                "limitations": [
                    "Running in minimal mode due to missing dependencies",
                    "Document retrieval and analysis capabilities limited",
                    "Using fallback response generation"
                ]
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return jsonify({
            "error": f"Query processing failed: {str(e)}",
            "answer": "I encountered an error while processing your security query. Please try again or check system logs.",
            "sources": [],
            "confidence": 0.0,
            "query_time": 0.0,
            "total_chunks_considered": 0,
            "security_context": "error",
            "recommendations": ["Check system logs for details", "Verify installation"],
            "limitations": ["System error prevented analysis"]
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if services_loaded and rag_pipeline:
        stats = rag_pipeline.get_statistics()
    else:
        stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "queries_processed": 0,
            "avg_query_time": 0,
            "last_ingestion": None,
            "mode": "minimal",
            "services_loaded": False,
            "recommendation": "Install dependencies and restart"
        }
    
    return jsonify(stats)

@app.route('/api/setup', methods=['GET'])
def setup_info():
    """Get setup information and instructions"""
    return jsonify({
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "requirements_file": "requirements-fixed.txt",
        "setup_steps": [
            "1. Create virtual environment: python -m venv venv",
            "2. Activate: venv\\Scripts\\activate (Windows)",
            "3. Install: pip install -r requirements-fixed.txt",
            "4. Run: python app.py",
            "5. Open frontend/index.html in browser"
        ],
        "troubleshooting": [
            "If numpy error: pip install numpy==1.23.5",
            "If magic error: pip install python-magic-bin==0.4.14",
            "If FAISS error: pip install faiss-cpu==1.7.4"
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data/documents', exist_ok=True)
    os.makedirs('data/vector_store', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Cybersecurity RAG System Starting...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info(f"Services Loaded: {services_loaded}")
    logger.info("=" * 60)
    
    if not services_loaded:
        logger.warning("Running in minimal mode. Some features disabled.")
        logger.warning("To enable full features:")
        logger.warning("1. Check requirements-fixed.txt")
        logger.warning("2. Install: pip install -r requirements-fixed.txt")
        logger.warning("3. Restart the application")
    
    logger.info("API Endpoints:")
    logger.info("  GET  /api/health      - Health check")
    logger.info("  POST /api/ask         - Ask security questions")
    logger.info("  GET  /api/stats       - System statistics")
    logger.info("  GET  /api/setup       - Setup instructions")
    logger.info("")
    logger.info("Access frontend: Open frontend/index.html in browser")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )