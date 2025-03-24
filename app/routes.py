from flask import Blueprint, render_template, request, jsonify, current_app
import os
import json
from app.utils.dropbox_handler import DropboxHandler
from app.utils.pdf_processor import PDFProcessor
from app.utils.vector_store import VectorStore
from app.utils.openai_handler import OpenAIHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)


@main.route('/')
def index():
    """Startseite der Anwendung"""
    return render_template('/index.html')


@main.route('/api/index-documents', methods=['POST'])
def index_documents():
    """Dokumente aus Dropbox laden und indexieren"""
    try:
        # Dropbox-Verbindung herstellen
        dropbox_token = current_app.config['DROPBOX_ACCESS_TOKEN']
        dropbox_path = current_app.config['DROPBOX_PDF_PATH']
        upload_folder = current_app.config['UPLOAD_FOLDER']

        # Qdrant-Einstellungen
        qdrant_url = current_app.config['QDRANT_URL']
        qdrant_api_key = current_app.config['QDRANT_API_KEY']
        collection_name = current_app.config['QDRANT_COLLECTION_NAME']

        # OpenAI-Einstellungen
        openai_api_key = current_app.config['OPENAI_API_KEY']
        embedding_model = current_app.config['OPENAI_EMBEDDING_MODEL']

        # PDF-Verarbeitungseinstellungen
        chunk_size = current_app.config['CHUNK_SIZE']
        chunk_overlap = current_app.config['CHUNK_OVERLAP']

        # Handler initialisieren
        dropbox_handler = DropboxHandler(dropbox_token)
        pdf_processor = PDFProcessor(chunk_size, chunk_overlap)
        vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)
        openai_handler = OpenAIHandler(openai_api_key, embedding_model=embedding_model)

        # Bisherigen Index löschen
        vector_store.clear_collection()

        # PDFs von Dropbox herunterladen
        logger.info(f"Lade PDFs von Dropbox-Pfad: {dropbox_path}")
        pdf_paths = dropbox_handler.download_all_pdfs(dropbox_path, upload_folder)

        if not pdf_paths:
            return jsonify({
                "success": False,
                "message": "Keine PDF-Dateien in Dropbox gefunden"
            }), 404

        # PDFs verarbeiten und in Chunks aufteilen
        chunks = pdf_processor.process_multiple_pdfs(pdf_paths)

        # Embeddings erstellen
        chunks_with_embeddings = openai_handler.get_embeddings_batch(chunks)

        # In Vektordatenbank speichern
        vector_store.store_embeddings(chunks_with_embeddings)

        # Dateien nach Verarbeitung löschen
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

        return jsonify({
            "success": True,
            "message": f"{len(chunks)} Chunks aus {len(pdf_paths)} PDFs erfolgreich indexiert"
        })
    except Exception as e:
        logger.error(f"Fehler beim Indexieren der Dokumente: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Fehler beim Indexieren: {str(e)}"
        }), 500


@main.route('/api/ask', methods=['POST'])
def ask_question():
    """Frage stellen und Antwort generieren"""
    try:
        # JSON-Daten aus dem Request extrahieren
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "message": "Keine Frage übermittelt"
            }), 400

        question = data['question']

        # OpenAI-Einstellungen
        openai_api_key = current_app.config['OPENAI_API_KEY']
        openai_model = current_app.config['OPENAI_MODEL']
        embedding_model = current_app.config['OPENAI_EMBEDDING_MODEL']

        # Qdrant-Einstellungen
        qdrant_url = current_app.config['QDRANT_URL']
        qdrant_api_key = current_app.config['QDRANT_API_KEY']
        collection_name = current_app.config['QDRANT_COLLECTION_NAME']

        # Handler initialisieren
        openai_handler = OpenAIHandler(openai_api_key, openai_model, embedding_model)
        vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)

        # Embedding für die Frage erstellen
        question_embedding = openai_handler.get_embedding(question)

        # Ähnliche Dokumente finden
        similar_docs = vector_store.search_similar(question_embedding, limit=5)

        if not similar_docs:
            return jsonify({
                "success": True,
                "answer": "Ich kann diese Frage nicht beantworten, da keine relevanten Informationen in den Dokumenten gefunden wurden.",
                "sources": []
            })

        # Antwort generieren
        answer = openai_handler.generate_answer(question, similar_docs)

        # Quellen für die Antwort extrahieren
        sources = []
        for doc in similar_docs:
            if doc['filename'] not in [s['filename'] for s in sources]:
                sources.append({
                    'filename': doc['filename'],
                    'score': doc['score']
                })

        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        logger.error(f"Fehler bei der Beantwortung der Frage: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Fehler bei der Beantwortung: {str(e)}"
        }), 500