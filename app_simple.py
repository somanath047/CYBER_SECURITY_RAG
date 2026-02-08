from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Simple in-memory storage
documents = []
chunks = []

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running", 
        "version": "1.0.0",
        "mode": "simple",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    # Enhanced response with all expected fields
    response = {
        "answer": f"🔐 **Security Analysis Results**\n\n"
                  f"**Query**: {question}\n\n"
                  f"**Analysis Mode**: Simple Demonstration\n"
                  f"**Status**: System operational\n"
                  f"**Documents loaded**: {len(documents)}\n\n"
                  f"**Findings**:\n"
                  f"• Query received and processed successfully\n"
                  f"• Basic analysis complete\n"
                  f"• System operating in demonstration mode\n\n"
                  f"**Security Context**: General Security Assessment\n\n"
                  f"*To enable full RAG capabilities:*\n"
                  f"1. Install dependencies from requirements-fixed.txt\n"
                  f"2. Add security documents to data/documents/\n"
                  f"3. Restart with full mode for advanced analysis",
        "sources": [],
        "confidence": 0.8,
        "query_time": 0.05,
        "total_chunks_considered": 0,
        "security_context": "general",
        "recommendations": [
            "Install all dependencies for full RAG functionality",
            "Upload security documents for contextual analysis",
            "Enable vector search for better relevance scoring"
        ],
        "limitations": [
            "Running in simple demonstration mode",
            "Document analysis capabilities limited",
            "No vector similarity search available"
        ]
    }
    
    return jsonify(response)

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename"}), 400
    
    # Save file
    filename = file.filename
    filepath = os.path.join('data', 'documents', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    
    documents.append({
        "name": filename,
        "path": filepath,
        "size": os.path.getsize(filepath),
        "upload_time": datetime.now().isoformat()
    })
    
    return jsonify({
        "success": True,
        "message": f"Uploaded {filename}",
        "documents": len(documents),
        "file_info": {
            "name": filename,
            "size": os.path.getsize(filepath),
            "path": filepath
        }
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """Simple stats endpoint for frontend"""
    return jsonify({
        "documents_processed": len(documents),
        "total_chunks": 0,
        "queries_processed": 0,
        "avg_query_time": 0,
        "last_ingestion": None,
        "mode": "simple",
        "documents": [{"name": d["name"], "size": d["size"]} for d in documents]
    })

if __name__ == '__main__':
    os.makedirs('data/documents', exist_ok=True)
    logger.info("=" * 60)
    logger.info("Simple Cybersecurity RAG Starting...")
    logger.info("Access at: http://localhost:5000")
    logger.info("API Endpoints:")
    logger.info("  GET  /api/health  - Health check")
    logger.info("  POST /api/ask     - Ask security questions")
    logger.info("  POST /api/upload  - Upload documents")
    logger.info("  GET  /api/stats   - System statistics")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)