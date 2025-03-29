from flask import Blueprint, render_template, request, jsonify, current_app
import os
import json
import time
import hashlib
from app.utils.dropbox_handler import DropboxHandler
from app.utils.pdf_processor import VisionAIDocumentProcessor
from app.utils.vector_store import VectorStore
from app.utils.openai_handler import OpenAIHandler
from app.utils.vision_handler import GoogleVisionHandler
import logging

# Logging konfigurieren
log_level = getattr(logging, current_app.config.get('LOG_LEVEL', 'INFO'))
log_file = current_app.config.get('LOG_FILE')

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file
)
logger = logging.getLogger(__name__)

# In-Memory-Cache für Antworten
response_cache = {}

main = Blueprint('main', __name__)


@main.route('/')
def index():
    """Startseite der Anwendung"""
    return render_template('/index.html')


@main.route('/api/index-documents', methods=['POST'])
def index_documents():
    """Dokumente aus Dropbox laden und mit Google Vision AI indexieren"""
    start_time = time.time()
    processing_stats = {
        "downloaded_files": 0,
        "processed_files": 0,
        "total_chunks": 0,
        "total_tokens": 0,
        "document_types": {},
        "processing_time": 0
    }

    try:
        # Konfiguration laden
        dropbox_token = current_app.config['DROPBOX_ACCESS_TOKEN']
        dropbox_path = current_app.config['DROPBOX_PDF_PATH']
        upload_folder = current_app.config['UPLOAD_FOLDER']

        qdrant_url = current_app.config['QDRANT_URL']
        qdrant_api_key = current_app.config['QDRANT_API_KEY']
        collection_name = current_app.config['QDRANT_COLLECTION_NAME']

        openai_api_key = current_app.config['OPENAI_API_KEY']
        embedding_model = current_app.config['OPENAI_EMBEDDING_MODEL']

        chunk_size = current_app.config['CHUNK_SIZE']
        chunk_overlap = current_app.config['CHUNK_OVERLAP']
        max_pages = current_app.config['MAX_PAGES_PER_PDF']

        google_credentials_path = current_app.config.get('GOOGLE_VISION_CREDENTIALS_PATH')
        use_layout = current_app.config.get('GOOGLE_VISION_USE_LAYOUT', True)

        # Parameter aus dem Request-Body holen (falls vorhanden)
        data = request.get_json(silent=True) or {}
        reset_index = data.get('reset_index', True)
        specific_files = data.get('files', [])  # Liste bestimmter Dateien, falls angegeben

        # Prüfen, ob die Google Vision Credentials gesetzt sind
        if not google_credentials_path or not os.path.exists(google_credentials_path):
            logger.warning("Google Vision Credentials nicht gefunden oder Pfad ungültig")
            return jsonify({
                "success": False,
                "message": "Google Vision API Anmeldedaten fehlen oder sind ungültig. Bitte überprüfen Sie die Konfiguration."
            }), 400

        # Handler initialisieren
        logger.info("Initialisiere Handler...")
        dropbox_handler = DropboxHandler(dropbox_token)

        pdf_processor = VisionAIDocumentProcessor(
            chunk_size,
            chunk_overlap,
            credentials_path=google_credentials_path,
            use_layout=use_layout,
            max_pages=max_pages
        )

        vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)
        openai_handler = OpenAIHandler(openai_api_key, embedding_model=embedding_model)

        # Bisherigen Index löschen, falls gewünscht
        if reset_index:
            logger.info("Lösche bisherigen Index...")
            vector_store.clear_collection()

        # PDFs von Dropbox herunterladen
        logger.info(f"Lade PDFs von Dropbox-Pfad: {dropbox_path}")

        if specific_files:
            # Nur bestimmte Dateien herunterladen
            pdf_paths = []
            for file_path in specific_files:
                full_path = os.path.join(dropbox_path, file_path) if dropbox_path else file_path
                try:
                    local_path = dropbox_handler.download_pdf(full_path,
                                                              os.path.join(upload_folder, os.path.basename(file_path)))
                    pdf_paths.append(local_path)
                except Exception as e:
                    logger.error(f"Fehler beim Herunterladen von {file_path}: {str(e)}")

            if not pdf_paths:
                return jsonify({
                    "success": False,
                    "message": "Keine der angegebenen PDF-Dateien konnte heruntergeladen werden."
                }), 404
        else:
            # Alle PDFs im angegebenen Pfad herunterladen
            pdf_paths = dropbox_handler.download_all_pdfs(dropbox_path, upload_folder)

        processing_stats["downloaded_files"] = len(pdf_paths)

        if not pdf_paths:
            return jsonify({
                "success": False,
                "message": "Keine PDF-Dateien in Dropbox gefunden"
            }), 404

        # PDFs verarbeiten und in Chunks aufteilen
        logger.info(f"Starte OCR-Verarbeitung mit Google Vision AI für {len(pdf_paths)} Dokumente...")
        chunks = pdf_processor.process_multiple_pdfs(pdf_paths)
        processing_stats["processed_files"] = len(pdf_paths)
        processing_stats["total_chunks"] = len(chunks)

        # Statistiken über Dokumenttypen sammeln
        for chunk in chunks:
            doc_type = chunk.get("doc_type", "unbekannt")
            if doc_type not in processing_stats["document_types"]:
                processing_stats["document_types"][doc_type] = 0
            processing_stats["document_types"][doc_type] += 1

        # Embeddings erstellen
        logger.info(f"Erstelle Embeddings für {len(chunks)} Chunks...")
        chunks_with_embeddings = openai_handler.get_embeddings_batch(chunks)

        # Tokens zählen (ungefähre Schätzung)
        total_text = sum(len(chunk.get("chunk_text", "")) for chunk in chunks)
        processing_stats["total_tokens"] = total_text // 4  # Grobe Schätzung: ~4 Zeichen pro Token

        # In Vektordatenbank speichern
        logger.info("Speichere Embeddings in Vektordatenbank...")
        vector_store.store_embeddings(chunks_with_embeddings)

        # Dateien nach Verarbeitung löschen
        logger.info("Lösche temporäre PDF-Dateien...")
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

        # Gesamtzeit berechnen
        processing_stats["processing_time"] = round(time.time() - start_time, 2)

        # Mehr Informationen über die Vektordatenbank abrufen
        try:
            vector_stats = vector_store.get_document_stats()
            processing_stats["vector_store"] = vector_stats
        except Exception as stats_error:
            logger.warning(f"Konnte keine Vektorstatistiken abrufen: {str(stats_error)}")

        return jsonify({
            "success": True,
            "message": f"{len(chunks)} Chunks aus {len(pdf_paths)} PDFs erfolgreich indexiert",
            "stats": processing_stats
        })
    except Exception as e:
        logger.error(f"Fehler beim Indexieren der Dokumente: {str(e)}")

        # Fehlerdetails erfassen
        processing_stats["error"] = str(e)
        processing_stats["processing_time"] = round(time.time() - start_time, 2)

        return jsonify({
            "success": False,
            "message": f"Fehler beim Indexieren: {str(e)}",
            "stats": processing_stats
        }), 500


@main.route('/api/ask', methods=['POST'])
def ask_question():
    """Frage stellen und Antwort generieren mit optimierter Verarbeitung"""
    try:
        # JSON-Daten aus dem Request extrahieren
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "message": "Keine Frage übermittelt"
            }), 400

        question = data['question']

        # Optionale Parameter
        filters = data.get('filters', {})  # Filter für die Suche
        use_cache = data.get('use_cache', current_app.config.get('ENABLE_RESPONSE_CACHE', True))

        # Cache-Schlüssel generieren (basierend auf Frage und Filtern)
        cache_key = hashlib.md5(f"{question}:{json.dumps(filters, sort_keys=True)}".encode()).hexdigest()

        # Wenn Cache aktiviert und Eintrag existiert, aus Cache zurückgeben
        if use_cache and cache_key in response_cache:
            cache_entry = response_cache[cache_key]
            cache_age = time.time() - cache_entry.get("timestamp", 0)
            cache_expiration = current_app.config.get('CACHE_EXPIRATION_SECONDS', 3600)

            if cache_age < cache_expiration:
                logger.info(f"Antwort aus Cache zurückgegeben (Alter: {cache_age:.1f}s)")
                return jsonify(cache_entry["response"])

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

        # Ähnliche Dokumente finden mit optionalen Filtern
        similar_docs = vector_store.search_similar(
            query_embedding=question_embedding,
            limit=8,  # Erhöhtes Limit für mehr Kontext
            filter_dict=filters
        )

        if not similar_docs:
            response = {
                "success": True,
                "answer": "Ich kann diese Frage nicht beantworten, da keine relevanten Informationen in den Dokumenten gefunden wurden.",
                "sources": []
            }

            # In Cache speichern, falls aktiviert
            if use_cache:
                response_cache[cache_key] = {
                    "response": response,
                    "timestamp": time.time()
                }

            return jsonify(response)

        # Informationen über Tabelleninhalte für die Antwortgenerierung hinzufügen
        for doc in similar_docs:
            if doc.get("content_type") == "table":
                doc["is_table"] = True
            else:
                doc["is_table"] = False

        # Antwort generieren mit erweitertem Kontext
        answer = openai_handler.generate_answer(question, similar_docs)

        # Quellen für die Antwort extrahieren mit besserer Dokumentation
        sources = []
        unique_sources = set()

        for doc in similar_docs:
            filename = doc.get('filename', '')
            if filename not in unique_sources:
                unique_sources.add(filename)

                source_info = {
                    'filename': filename,
                    'score': doc.get('score', 0),
                    'doc_type': doc.get('doc_type', 'unbekannt')
                }

                # Zusätzliche Metadaten je nach Dokumenttyp
                if doc.get("content_type") == "table":
                    source_info['content_type'] = 'Tabelle'

                if doc.get("doc_type") == "rechnung":
                    # Für Rechnungen relevante zusätzliche Informationen
                    if 'invoice_number' in doc:
                        source_info['invoice_number'] = doc['invoice_number']
                    if 'invoice_date' in doc:
                        source_info['invoice_date'] = doc['invoice_date']

                sources.append(source_info)

                # Begrenze die Anzahl der Quellen
                max_sources = current_app.config.get('MAX_SOURCES_IN_RESPONSE', 5)
                if len(sources) >= max_sources:
                    break

        # Antwort mit Metadaten
        response = {
            "success": True,
            "answer": answer,
            "sources": sources
        }

        # Optional: erweiterte Metadaten hinzufügen
        if current_app.config.get('INCLUDE_METADATA_IN_RESPONSE', True):
            response["metadata"] = {
                "model": openai_model,
                "document_count": len(similar_docs),
                "document_types": list(set(doc.get('doc_type', 'unbekannt') for doc in similar_docs)),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # In Cache speichern, falls aktiviert
        if use_cache:
            response_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Fehler bei der Beantwortung der Frage: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Fehler bei der Beantwortung: {str(e)}"
        }), 500


@main.route('/api/status', methods=['GET'])
def get_status():
    """Status der Anwendung und des Google Vision API Dienstes prüfen"""
    try:
        status = {
            "app_status": "online",
            "services": {},
            "vector_store": {},
            "config": {}
        }

        # Google Vision API Status prüfen
        google_credentials_path = current_app.config.get('GOOGLE_VISION_CREDENTIALS_PATH')
        if google_credentials_path and os.path.exists(google_credentials_path):
            status["services"]["google_vision"] = "configured"

            # Testversuch der API-Verbindung
            try:
                vision_handler = GoogleVisionHandler(google_credentials_path)
                status["services"]["google_vision"] = "online"
            except Exception as vision_e:
                status["services"]["google_vision"] = f"error: {str(vision_e)}"
        else:
            status["services"]["google_vision"] = "not_configured"

        # Qdrant Status prüfen
        qdrant_url = current_app.config['QDRANT_URL']
        qdrant_api_key = current_app.config['QDRANT_API_KEY']
        collection_name = current_app.config['QDRANT_COLLECTION_NAME']

        try:
            vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)
            # Einen einfachen Aufruf durchführen
            collections = vector_store.client.get_collections()
            status["services"]["qdrant"] = "online"

            # Statistiken über den Vektorspeicher
            try:
                stats = vector_store.get_document_stats()
                status["vector_store"] = stats
            except Exception as stats_e:
                status["vector_store"]["error"] = str(stats_e)
        except Exception as e:
            status["services"]["qdrant"] = f"error: {str(e)}"

        # OpenAI API Status
        openai_api_key = current_app.config['OPENAI_API_KEY']
        if openai_api_key:
            status["services"]["openai"] = "configured"
            # Testversuch der API-Verbindung
            try:
                handler = OpenAIHandler(openai_api_key)
                status["services"]["openai"] = "online"
            except Exception as openai_e:
                status["services"]["openai"] = f"error: {str(openai_e)}"
        else:
            status["services"]["openai"] = "not_configured"

        # Dropbox Status
        dropbox_token = current_app.config['DROPBOX_ACCESS_TOKEN']
        if dropbox_token:
            status["services"]["dropbox"] = "configured"
            try:
                handler = DropboxHandler(dropbox_token)
                status["services"]["dropbox"] = "online"
            except Exception as dropbox_e:
                status["services"]["dropbox"] = f"error: {str(dropbox_e)}"
        else:
            status["services"]["dropbox"] = "not_configured"

        # Konfigurationseinstellungen (ohne Geheimnisse)
        safe_config = {
            "chunk_size": current_app.config['CHUNK_SIZE'],
            "chunk_overlap": current_app.config['CHUNK_OVERLAP'],
            "max_pages_per_pdf": current_app.config['MAX_PAGES_PER_PDF'],
            "use_layout": current_app.config.get('GOOGLE_VISION_USE_LAYOUT', True),
            "enable_cache": current_app.config.get('ENABLE_RESPONSE_CACHE', True),
            "use_metadata": current_app.config.get('INCLUDE_METADATA_IN_RESPONSE', True)
        }
        status["config"] = safe_config

        return jsonify(status)
    except Exception as e:
        return jsonify({
            "app_status": "error",
            "error": str(e)
        }), 500


@main.route('/api/document-types', methods=['GET'])
def get_document_types():
    """Liefert eine Liste der verfügbaren Dokumenttypen und Anzahl der Dokumente"""
    try:
        # Qdrant-Einstellungen
        qdrant_url = current_app.config['QDRANT_URL']
        qdrant_api_key = current_app.config['QDRANT_API_KEY']
        collection_name = current_app.config['QDRANT_COLLECTION_NAME']

        # Vector Store initialisieren
        vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)

        # Statistiken über Dokumente abrufen
        stats = vector_store.get_document_stats()

        return jsonify({
            "success": True,
            "document_types": stats.get("document_types", {}),
            "total_documents": stats.get("total_vectors", 0),
            "content_types": stats.get("content_types", {})
        })
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Dokumenttypen: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Fehler: {str(e)}"
        }), 500


@main.route('/api/documents', methods=['GET'])
def list_documents():
    """Liefert eine Liste der indexierten Dokumente mit optionaler Filterung"""
    try:
        # Parameter aus der Anfrage lesen
        doc_type = request.args.get('type')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))

        # Qdrant-Einstellungen
        qdrant_url = current_app.config['QDRANT_URL']
        qdrant_api_key = current_app.config['QDRANT_API_KEY']
        collection_name = current_app.config['QDRANT_COLLECTION_NAME']

        # Vector Store initialisieren
        vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)

        if doc_type:
            # Filtern nach Dokumenttyp
            result = vector_store.filter_by_document_type(doc_type, limit, offset)
            return jsonify({
                "success": True,
                "documents": result.get("documents", []),
                "total": result.get("total", 0),
                "has_more": result.get("has_more", False),
                "offset": offset,
                "limit": limit
            })
        else:
            # Allgemeine Statistiken zurückgeben
            stats = vector_store.get_document_stats()
            return jsonify({
                "success": True,
                "stats": stats,
                "message": "Bitte geben Sie einen Dokumenttyp an, um die Liste zu filtern"
            })
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Dokumentliste: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Fehler: {str(e)}"
        }), 500


@main.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Löscht den Antwort-Cache"""
    global response_cache
    response_cache = {}
    return jsonify({
        "success": True,
        "message": "Cache erfolgreich geleert"
    })