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
    DROPBOX_PDF_PATH = os.environ.get('DROPBOX_PDF_PATH', '')  # Pfad zu deinen PDFs in Dropbox

    # Qdrant Konfiguration
    QDRANT_URL = os.environ.get('QDRANT_URL', 'http://localhost:6333')
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.environ.get('QDRANT_COLLECTION_NAME', 'pdf_documents')

    # Google Vision AI Konfiguration
    GOOGLE_VISION_CREDENTIALS_PATH = os.environ.get('GOOGLE_VISION_CREDENTIALS_PATH')
    GOOGLE_VISION_USE_LAYOUT = os.environ.get('GOOGLE_VISION_USE_LAYOUT', 'True').lower() in ('true', '1', 't')
    GOOGLE_VISION_REQUEST_TIMEOUT = int(os.environ.get('GOOGLE_VISION_REQUEST_TIMEOUT', '300'))  # Timeout in Sekunden

    # PDF-Verarbeitungskonfiguration
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))
    MAX_PAGES_PER_PDF = int(os.environ.get('MAX_PAGES_PER_PDF', '0'))  # 0 = unbegrenzt

    # Anwendungskonfiguration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_pdfs')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Logging-Konfiguration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE')  # Falls None, wird nur in die Konsole geloggt

    # Erweitertes Antwortformat
    INCLUDE_METADATA_IN_RESPONSE = os.environ.get('INCLUDE_METADATA_IN_RESPONSE', 'True').lower() in ('true', '1', 't')
    MAX_SOURCES_IN_RESPONSE = int(os.environ.get('MAX_SOURCES_IN_RESPONSE', '5'))

    # Cache-Konfiguration
    ENABLE_RESPONSE_CACHE = os.environ.get('ENABLE_RESPONSE_CACHE', 'True').lower() in ('true', '1', 't')
    CACHE_EXPIRATION_SECONDS = int(os.environ.get('CACHE_EXPIRATION_SECONDS', '3600'))  # 1 Stunde