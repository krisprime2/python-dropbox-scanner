import os
import PyPDF2
import re
import logging
from typing import List, Dict, Any
import pytesseract
from pdf2image import convert_from_path
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRDocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Text aus einer PDF-Datei extrahieren mit OCR-Fallback"""
        try:
            # Erst versuchen wir es mit PyPDF2 für digitalen Text
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"--- Seite {page_num + 1} ---\n{page_text}\n\n"

            # Wenn kein Text gefunden wurde oder dieser sehr kurz ist, OCR anwenden
            if len(text.strip()) < 100:  # Annahme: Wenn weniger als 100 Zeichen, dann wahrscheinlich kein digitaler Text
                logger.info(f"Wenig oder kein Text mit PyPDF2 gefunden. Versuche OCR für {pdf_path}")
                text = self._extract_text_with_ocr(pdf_path)

            logger.info(f"Text aus {pdf_path} extrahiert")
            return text
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des Textes aus {pdf_path}: {str(e)}")
            logger.info("Versuche OCR als Fallback...")
            try:
                return self._extract_text_with_ocr(pdf_path)
            except Exception as ocr_e:
                logger.error(f"Auch OCR ist fehlgeschlagen: {str(ocr_e)}")
                raise

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """Text aus PDF mit OCR extrahieren"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # PDF in Bilder konvertieren
                images = convert_from_path(pdf_path, dpi=300)  # Höhere Auflösung für bessere OCR

                full_text = ""
                for i, image in enumerate(images):
                    # Bild temporär speichern
                    temp_image = os.path.join(temp_dir, f'page_{i}.png')
                    image.save(temp_image, 'PNG')

                    # OCR auf dem Bild ausführen (für deutsche Dokumente)
                    try:
                        page_text = pytesseract.image_to_string(temp_image, lang='deu')
                    except:
                        # Fallback auf Englisch, falls deutsches Sprachpaket nicht installiert ist
                        page_text = pytesseract.image_to_string(temp_image, lang='eng')

                    full_text += f"--- Seite {i + 1} ---\n{page_text}\n\n"

                logger.info(f"OCR-Text aus {pdf_path} extrahiert")
                return full_text
        except Exception as e:
            logger.error(f"Fehler bei der OCR-Textextraktion aus {pdf_path}: {str(e)}")
            raise

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Text in semantisch sinnvolle Chunks aufteilen"""
        # Definiere Muster für semantische Trennung (Abschnitte, Paragraphen, etc.)
        section_patterns = [
            r'§\s*\d+[a-z]?\.?\s+[A-Z]',  # Paragraphen in Verträgen, z.B. "§ 1. Vertragsgegenstand"
            r'^\s*\d+\.\s+[A-Z]',  # Nummerierte Abschnitte, z.B. "1. Allgemeines"
            r'--- Seite \d+ ---',  # Seitenmarkierungen
            r'^[A-Z][A-Za-zÄÖÜäöüß\s]{3,}$'  # Überschriften (komplett großgeschrieben)
        ]

        # Textblöcke durch semantische Muster teilen
        chunks = []
        current_chunk = ""
        current_section = "Allgemein"

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Prüfen, ob die Zeile ein neuer Abschnitt ist
            is_new_section = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    # Speichere den aktuellen Chunk, wenn er nicht leer ist
                    if current_chunk.strip():
                        chunk_metadata = metadata.copy()
                        chunk_metadata["section"] = current_section
                        chunk_metadata["chunk_text"] = current_chunk.strip()
                        chunks.append(chunk_metadata)

                    # Beginne einen neuen Chunk mit der aktuellen Zeile als Überschrift
                    current_section = line
                    current_chunk = line + "\n"
                    is_new_section = True
                    break

            if not is_new_section:
                # Füge die Zeile zum aktuellen Chunk hinzu
                current_chunk += line + "\n"

                # Wenn der aktuelle Chunk zu groß wird, teile ihn auf
                if len(current_chunk) > self.chunk_size:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["section"] = current_section
                    chunk_metadata["chunk_text"] = current_chunk[:self.chunk_size].strip()
                    chunks.append(chunk_metadata)

                    # Überlappung für den nächsten Chunk
                    overlap_start = max(0, self.chunk_size - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]

        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["section"] = current_section
            chunk_metadata["chunk_text"] = current_chunk.strip()
            chunks.append(chunk_metadata)

        logger.info(f"Text in {len(chunks)} semantische Chunks aufgeteilt")
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
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
                logger.info(f"PDF {pdf_path} erfolgreich verarbeitet, {len(chunks)} Chunks extrahiert")
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung von {pdf_path}, überspringe Datei: {str(e)}")
                continue

        logger.info(f"Insgesamt {len(all_chunks)} Chunks aus {len(pdf_paths)} PDFs extrahiert")
        return all_chunks