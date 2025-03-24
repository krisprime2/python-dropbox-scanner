import os
import PyPDF2
import re
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Text aus einer PDF-Datei extrahieren"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"

            logger.info(f"Text aus {pdf_path} extrahiert")
            return text
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des Textes aus {pdf_path}: {str(e)}")
            raise

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Text in überlappende Chunks aufteilen ohne NLTK"""
        # Einfache Satzendenerkennung
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s'
        sentences = re.split(sentence_endings, text)

        # Leere Sätze entfernen
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Wenn das Hinzufügen des nächsten Satzes den Chunk zu groß macht
            if len(current_chunk) + len(sentence) > self.chunk_size:
                # Aktuellen Chunk speichern
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_text"] = current_chunk
                    chunks.append(chunk_metadata)

                    # Neuen Chunk mit Überlappung beginnen
                    current_chunk_words = current_chunk.split()
                    overlap_words = current_chunk_words[-self.chunk_overlap:] if len(
                        current_chunk_words) > self.chunk_overlap else current_chunk_words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    # Falls der Satz allein größer als chunk_size ist, teile ihn auf
                    current_chunk = sentence
            else:
                # Füge den Satz zum aktuellen Chunk hinzu
                current_chunk += " " + sentence if current_chunk else sentence

        # Letzten Chunk hinzufügen, falls vorhanden
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_text"] = current_chunk
            chunks.append(chunk_metadata)

        logger.info(f"Text in {len(chunks)} Chunks aufgeteilt")
        return chunks

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """PDF-Datei vollständig verarbeiten: Text extrahieren und in Chunks aufteilen"""
        try:
            filename = os.path.basename(pdf_path)
            text = self.extract_text_from_pdf(pdf_path)

            metadata = {
                "source": pdf_path,
                "filename": filename
            }

            chunks = self.chunk_text(text, metadata)
            return chunks
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {pdf_path}: {str(e)}")
            raise

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Mehrere PDF-Dateien verarbeiten"""
        all_chunks = []

        for pdf_path in pdf_paths:
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)

        logger.info(f"Insgesamt {len(all_chunks)} Chunks aus {len(pdf_paths)} PDFs extrahiert")
        return all_chunks