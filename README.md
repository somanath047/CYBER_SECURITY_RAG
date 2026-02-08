# CYBER_SECURITY_RAG
graph TD
    A[User Question] --> B[Question Analyzer]
    B --> C[Query Formatter]
    C --> D[Vector Database]
    D --> E[Context Retrieval]
    E --> F[LLM Integration]
    F --> G[Answer Generator]
    G --> H[Step-by-Step Working]
    H --> I[Formatted Response]


    # Python 3.8 or higher
python --version

# Package manager
pip install --upgrade pip


git clone https://github.com/somanath047/CYBER_SECURITY_RAG.git
cd rag-question-analysis


# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt


