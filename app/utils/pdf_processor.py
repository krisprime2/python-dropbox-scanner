import os
import re
import logging
from typing import List, Dict, Any, Tuple
import time
from app.utils.vision_handler import GoogleVisionHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionAIDocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, credentials_path=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vision_handler = GoogleVisionHandler(credentials_path)

        # Dokumenttyp-Erkennungsmuster
        self.doc_type_patterns = {
            "rechnung": [
                r"rechnung", r"invoice", r"faktura", r"mwst", r"umsatzsteuer",
                r"zahlungsziel", r"betrag", r"rechnungsdatum", r"kundennummer"
            ],
            "angebot": [
                r"angebot", r"quotation", r"kostenvoranschlag", r"gültig bis",
                r"angebotsnummer", r"unverbindlich"
            ],
            "vertrag": [
                r"vertrag", r"contract", r"vereinbarung", r"unterzeichner",
                r"vertragspartei", r"kündigung", r"laufzeit"
            ],
            "formular": [
                r"formular", r"antrag", r"bitte ausfüllen", r"form",
                r"unterschrift", r"datum"
            ],
            "technisch": [
                r"technische daten", r"spezifikation", r"anleitung", r"technical",
                r"specification", r"manual", r"handbuch"
            ]
        }

    def detect_document_type(self, text: str) -> str:
        """
        Erkennt den Dokumenttyp basierend auf dem extrahierten Text.
        """
        text_lower = text.lower()
        scores = {}

        # Bewerte jeden Dokumenttyp anhand der gefundenen Schlüsselwörter
        for doc_type, patterns in self.doc_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[doc_type] = score

        # Wenn kein klarer Typ erkannt wurde, verwende "allgemein"
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return "allgemein"

        # Gib den Typ mit dem höchsten Score zurück
        for doc_type, score in scores.items():
            if score == max_score:
                return doc_type

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extrahiert Text direkt aus PDF mit Google Vision AI.
        Kein Umwandeln in Bilder erforderlich.

        Returns:
            Tuple mit (extrahierter Text, Metadaten inkl. Dokumenttyp)
        """
        try:
            # Struktur-Info initialisieren
            structure_info = {
                "doc_type": "unbekannt",
                "processing_method": "google_vision_api_direct"
            }

            # Google Vision AI für die direkte PDF-Verarbeitung verwenden
            logger.info(f"Extrahiere Text direkt aus PDF {pdf_path} mit Google Vision API")
            extracted_text, page_count = self.vision_handler.process_pdf_document(pdf_path)

            # Page Count in Struktur-Info speichern
            structure_info["page_count"] = page_count

            # Dokumenttyp erkennen
            doc_type = self.detect_document_type(extracted_text)
            structure_info["doc_type"] = doc_type

            logger.info(f"Dokumenttyp erkannt: {doc_type} für {pdf_path} mit {page_count} Seiten")
            return extracted_text, structure_info

        except Exception as e:
            logger.error(f"Fehler bei der direkten Google Vision API PDF-Verarbeitung von {pdf_path}: {str(e)}")
            raise

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Text in semantisch sinnvolle Chunks aufteilen, mit spezieller Behandlung von Tabellen
        """
        # Definiere Muster für semantische Trennung
        section_patterns = [
            r'§\s*\d+[a-z]?\.?\s+[A-Z]',  # Paragraphen in Verträgen, z.B. "§ 1. Vertragsgegenstand"
            r'^\s*\d+\.\s+[A-Z]',  # Nummerierte Abschnitte, z.B. "1. Allgemeines"
            r'--- Seite \d+ ---',  # Seitenmarkierungen
            r'^[A-Z][A-Za-zÄÖÜäöüß\s]{3,}$'  # Überschriften (komplett großgeschrieben)
        ]

        # Erkennung von Tabellenstrukturen (Zeilen mit Tab-Trennzeichen)
        table_pattern = r'.*\t.*'

        # Textblöcke durch semantische Muster teilen
        chunks = []
        current_chunk = ""
        current_section = "Allgemein"
        in_table = False
        table_content = ""

        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            # Prüfe, ob wir eine Tabellenzeile haben
            is_table_row = bool(re.match(table_pattern, line))

            # Wenn wir beginnen, eine Tabelle zu erkennen
            if is_table_row and not in_table:
                # Speichere den aktuellen Chunk, bevor wir mit der Tabelle beginnen
                if current_chunk.strip():
                    chunk_metadata = metadata.copy()
                    chunk_metadata["section"] = current_section
                    chunk_metadata["chunk_text"] = current_chunk.strip()
                    chunk_metadata["content_type"] = "text"
                    chunks.append(chunk_metadata)
                    current_chunk = ""

                in_table = True
                table_content = f"Tabelle in Abschnitt: {current_section}\n\n{line}\n"

                # Sammele weitere Tabellenzeilen
                while i < len(lines) and (not lines[i].strip() or re.match(table_pattern, lines[i].strip())):
                    table_content += lines[i].strip() + "\n"
                    i += 1

                # Tabelle als eigenen Chunk speichern
                table_metadata = metadata.copy()
                table_metadata["section"] = current_section
                table_metadata["chunk_text"] = table_content.strip()
                table_metadata["content_type"] = "table"
                chunks.append(table_metadata)

                in_table = False
                table_content = ""
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
                        chunk_metadata["content_type"] = "text"
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
                    chunk_metadata["content_type"] = "text"
                    chunks.append(chunk_metadata)

                    # Überlappung für den nächsten Chunk
                    overlap_start = max(0, self.chunk_size - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]

        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["section"] = current_section
            chunk_metadata["chunk_text"] = current_chunk.strip()
            chunk_metadata["content_type"] = "text"
            chunks.append(chunk_metadata)

        logger.info(f"Text in {len(chunks)} semantische Chunks aufgeteilt (inkl. Tabellenerkennung)")
        return chunks

    def _chunk_invoice(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Spezielle Chunking-Methode für Rechnungen.
        """
        chunks = []

        # Trenne Text nach Seiten
        pages = re.split(r"---\s+Seite\s+\d+\s+---", text)
        pages = [p.strip() for p in pages if p.strip()]

        # Jede Seite als eigener Chunk mit klarem Kontext
        for i, page_text in enumerate(pages):
            chunk_metadata = metadata.copy()
            chunk_metadata["page_number"] = i + 1
            chunk_metadata["section"] = f"Rechnung Seite {i + 1}"
            chunk_metadata["chunk_text"] = page_text
            chunk_metadata["content_type"] = "invoice_page"

            # Wichtige Rechnungsdaten extrahieren
            rechnungsnr_match = re.search(r"Rech?nu?ngs\s*-?n?[ur]?[.:]\s*([A-Za-z0-9-]+)", page_text, re.IGNORECASE)
            datum_match = re.search(r"Datum\s*[.:]\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", page_text, re.IGNORECASE)
            summe_match = re.search(r"Gesamt\s*summe\s*[.:]\s*([0-9.,]+)", page_text, re.IGNORECASE)

            if rechnungsnr_match:
                chunk_metadata["invoice_number"] = rechnungsnr_match.group(1)
            if datum_match:
                chunk_metadata["invoice_date"] = datum_match.group(1)
            if summe_match:
                chunk_metadata["invoice_total"] = summe_match.group(1)

            chunks.append(chunk_metadata)

        return chunks

    def _chunk_contract(self, text: str, metadata: Dict[str, Any], chunk_size=1000, chunk_overlap=400) -> List[
        Dict[str, Any]]:
        """
        Spezielle Chunking-Methode für Verträge und Angebote.
        """
        # Muster für typische Vertragsabschnitte oder Paragraphen
        section_patterns = [
            r'§\s*\d+[a-z]?\.?\s+[A-Z]',  # Paragraphen wie "§ 1. Vertragsgegenstand"
            r'^\s*\d+\.\s+[A-Z]',  # Nummerierte Abschnitte wie "1. Allgemeines"
            r'^\s*[A-ZÄÖÜ][A-ZÄÖÜa-zäöüß\s]+\s*$',  # Überschriften
            r'Artikel\s+\d+',  # Artikelbezeichnungen wie "Artikel 5"
            r'Abschnitt\s+[IVX]+',  # Römische Abschnittsnummerierung
            r'--- Seite \d+ ---'  # Seitenmarkierungen
        ]

        # Textblöcke durch rechtliche Abschnittsmuster teilen
        chunks = []
        current_chunk = ""
        current_section = "Präambel"

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
                        chunk_metadata["content_type"] = "contract_clause"
                        chunks.append(chunk_metadata)

                    # Beginne einen neuen Chunk mit aktueller Zeile als Überschrift
                    current_section = line
                    current_chunk = line + "\n"
                    is_new_section = True
                    break

            if not is_new_section:
                # Füge die Zeile zum aktuellen Chunk hinzu
                current_chunk += line + "\n"

                # Wenn der aktuelle Chunk zu groß wird, teile ihn auf
                if len(current_chunk) > chunk_size:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["section"] = current_section
                    chunk_metadata["chunk_text"] = current_chunk[:chunk_size].strip()
                    chunk_metadata["content_type"] = "contract_clause"
                    chunks.append(chunk_metadata)

                    # Überlappung für den nächsten Chunk
                    overlap_start = max(0, chunk_size - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]

        # Letzten Chunk hinzufügen
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["section"] = current_section
            chunk_metadata["chunk_text"] = current_chunk.strip()
            chunk_metadata["content_type"] = "contract_clause"
            chunks.append(chunk_metadata)

        logger.info(f"Vertrag in {len(chunks)} rechtliche Abschnitte aufgeteilt")
        return chunks

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """PDF-Datei vollständig verarbeiten: Text extrahieren und in Chunks aufteilen"""
        try:
            filename = os.path.basename(pdf_path)
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Größe in MB

            # Text direkt aus PDF extrahieren mit Google Vision API
            text, doc_info = self.extract_text_from_pdf(pdf_path)

            # Erweiterte Metadaten erstellen
            metadata = {
                "source": pdf_path,
                "filename": filename,
                "file_size_mb": round(file_size, 2),
                "doc_type": doc_info.get("doc_type", "unbekannt"),
                "page_count": doc_info.get("page_count", 0),
                "processing_method": "google_vision_api_direct",
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Angepasste Chunking-Strategie basierend auf Dokumenttyp
            if metadata["doc_type"] == "rechnung":
                # Für Rechnungen: Keine Überlappung, ganze Seiten als Chunks
                chunks = self._chunk_invoice(text, metadata)
            elif metadata["doc_type"] in ["vertrag", "angebot"]:
                # Für Verträge: Semantische Abschnitte mit höherer Überlappung
                chunks = self._chunk_contract(text, metadata,
                                              chunk_size=self.chunk_size,
                                              chunk_overlap=self.chunk_overlap * 2)
            else:
                # Standard-Chunking für alle anderen Dokumente
                chunks = self.chunk_text(text, metadata)

            # Ergänzende Informationen zu jedem Chunk hinzufügen
            for chunk in chunks:
                chunk["doc_type"] = metadata["doc_type"]
                chunk["total_chunks"] = len(chunks)

            return chunks
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {pdf_path}: {str(e)}")
            raise

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Mehrere PDF-Dateien verarbeiten mit detaillierter Statistik"""
        all_chunks = []
        processed_files = 0
        skipped_files = 0
        doc_types_stats = {}

        start_time = time.time()

        for pdf_path in pdf_paths:
            try:
                logger.info(f"Verarbeite PDF {processed_files + skipped_files + 1}/{len(pdf_paths)}: {pdf_path}")
                file_start_time = time.time()

                chunks = self.process_pdf(pdf_path)
                processed_files += 1

                # Statistiken sammeln
                doc_type = chunks[0].get("doc_type", "unbekannt") if chunks else "unbekannt"
                if doc_type not in doc_types_stats:
                    doc_types_stats[doc_type] = {"count": 0, "chunks": 0}
                doc_types_stats[doc_type]["count"] += 1
                doc_types_stats[doc_type]["chunks"] += len(chunks)

                # Performance-Messung
                processing_time = time.time() - file_start_time
                logger.info(f"PDF {pdf_path} erfolgreich verarbeitet in {processing_time:.2f}s: "
                            f"{len(chunks)} Chunks extrahiert, Dokumenttyp: {doc_type}")

                all_chunks.extend(chunks)
            except Exception as e:
                skipped_files += 1
                logger.error(f"Fehler bei der Verarbeitung von {pdf_path}, überspringe Datei: {str(e)}")
                continue

        total_time = time.time() - start_time
        avg_time_per_file = total_time / max(processed_files, 1)

        # Ausführliche Logs über die Verarbeitung
        logger.info(f"PDF-Verarbeitung abgeschlossen in {total_time:.2f}s "
                    f"(durchschnittlich {avg_time_per_file:.2f}s pro Datei)")
        logger.info(f"Verarbeitete Dateien: {processed_files}, Übersprungene Dateien: {skipped_files}")
        logger.info(f"Insgesamt {len(all_chunks)} Chunks aus {processed_files} PDFs extrahiert")
        logger.info(f"Dokumenttyp-Statistik: {doc_types_stats}")

        return all_chunks