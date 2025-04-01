# app/scheduled_tasks.py
import logging
import os
import time
from app.utils.dropbox_handler import DropboxHandler
from app.utils.pdf_processor import VisionAIDocumentProcessor
from app.utils.vector_store import VectorStore
from app.utils.openai_handler import OpenAIHandler
from app.routes import filter_new_documents

logger = logging.getLogger(__name__)


def configure_scheduled_jobs(app):
    """Konfiguriert die geplanten Aufgaben für den Scheduler"""
    # JobIDs und Funktionsreferenzen
    jobs = [
        {
            "id": "nightly_indexing",
            "func": "app.scheduled_tasks:nightly_indexing_job",
            "trigger": "cron",
            "hour": 3,  # Um 3 Uhr morgens
            "minute": 0,
            "max_instances": 1,
            "replace_existing": True
        }
    ]

    # Jobs zum Scheduler hinzufügen
    for job in jobs:
        app.scheduler.add_job(**job)

    logger.info(f"Geplante Jobs konfiguriert: {[job['id'] for job in jobs]}")


def nightly_indexing_job():
    """Job für die nächtliche Indexierung neuer PDFs"""
    from flask import current_app

    logger.info("=== STARTE NÄCHTLICHE INDEXIERUNG ===")
    start_time = time.time()

    # Statistiken für den Bericht
    stats = {
        "downloaded_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "total_chunks": 0,
        "error": None
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
        google_credentials_path = current_app.config.get('GOOGLE_VISION_CREDENTIALS_PATH')

        # Prüfen, ob die Google Vision Credentials gesetzt sind
        if not google_credentials_path or not os.path.exists(google_credentials_path):
            error_msg = "Google Vision Credentials nicht gefunden oder Pfad ungültig"
            logger.error(error_msg)
            stats["error"] = error_msg
            return stats

        # Handler initialisieren
        logger.info("Initialisiere Handler für nächtliche Indexierung...")
        dropbox_handler = DropboxHandler(dropbox_token)

        pdf_processor = VisionAIDocumentProcessor(
            chunk_size,
            chunk_overlap,
            credentials_path=google_credentials_path,
        )

        vector_store = VectorStore(qdrant_url, collection_name, api_key=qdrant_api_key)
        openai_handler = OpenAIHandler(openai_api_key, embedding_model=embedding_model)

        # Liste bereits indexierter Dateien abrufen
        indexed_files = vector_store.get_indexed_files()
        logger.info(f"{len(indexed_files)} bereits indexierte Dateien gefunden")

        # Nur neue PDFs herunterladen
        pdf_paths = filter_new_documents(dropbox_handler, dropbox_path, upload_folder, indexed_files)

        stats["downloaded_files"] = len(pdf_paths)
        stats["skipped_files"] = len(indexed_files)

        if not pdf_paths:
            logger.info("Keine neuen PDF-Dateien zum Indexieren gefunden")
            stats["processing_time"] = round(time.time() - start_time, 2)
            return stats

        # PDFs verarbeiten und in Chunks aufteilen
        logger.info(f"Starte OCR-Verarbeitung mit Google Vision AI für {len(pdf_paths)} neue Dokumente...")
        chunks = pdf_processor.process_multiple_pdfs(pdf_paths)
        stats["processed_files"] = len(pdf_paths)
        stats["total_chunks"] = len(chunks)

        # Embeddings erstellen
        logger.info(f"Erstelle Embeddings für {len(chunks)} Chunks...")
        chunks_with_embeddings = openai_handler.get_embeddings_batch(chunks)

        # In Vektordatenbank speichern
        logger.info("Speichere Embeddings in Vektordatenbank...")
        vector_store.store_embeddings(chunks_with_embeddings)

        # Dateien nach Verarbeitung löschen
        logger.info("Lösche temporäre PDF-Dateien...")
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

        stats["processing_time"] = round(time.time() - start_time, 2)
        logger.info(f"=== NÄCHTLICHE INDEXIERUNG ABGESCHLOSSEN ===")
        logger.info(f"Verarbeitet: {stats['processed_files']} Dateien, {stats['total_chunks']} Chunks")
        logger.info(f"Dauer: {stats['processing_time']} Sekunden")

        return stats

    except Exception as e:
        error_msg = f"Fehler bei der nächtlichen Indexierung: {str(e)}"
        logger.error(error_msg)
        stats["error"] = error_msg
        stats["processing_time"] = round(time.time() - start_time, 2)
        return stats