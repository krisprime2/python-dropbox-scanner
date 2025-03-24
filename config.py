import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'eine-schwer-zu-erratende-zeichenfolge'

    # OpenAI Konfiguration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')

    # Dropbox Konfiguration
    DROPBOX_ACCESS_TOKEN = os.environ.get('DROPBOX_ACCESS_TOKEN')
    DROPBOX_PDF_PATH = os.environ.get('DROPBOX_PDF_PATH', '/Dropbox-Scanner')  # Pfad zu deinen PDFs in Dropbox

    # Qdrant Konfiguration
    QDRANT_URL = os.environ.get('QDRANT_URL', 'http://localhost:6333')
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.environ.get('QDRANT_COLLECTION_NAME', 'pdf_documents')

    # PDF-Verarbeitungskonfiguration
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))

    # Anwendungskonfiguration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_pdfs')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)