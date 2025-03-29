import os
from google.cloud import vision
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleVisionHandler:
    def __init__(self, credentials_path=None):
        """
        Google Vision API Client initialisieren.

        Args:
            credentials_path (str, optional): Pfad zur Google Cloud Credentials JSON-Datei.
                Falls None, wird die Umgebungsvariable GOOGLE_APPLICATION_CREDENTIALS verwendet.
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        self.client = vision.ImageAnnotatorClient()
        logger.info("Google Vision API Client initialisiert")

    def process_pdf_document(self, pdf_path: str) -> Tuple[str, int]:
        """
        Prozessiert ein PDF-Dokument direkt mit der Google Vision API.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            Tuple[str, int]: (Extrahierter Text, Anzahl der Seiten)
        """
        try:
            # PDF-Datei öffnen und Inhalt als Bytes lesen
            with open(pdf_path, 'rb') as pdf_file:
                content = pdf_file.read()

            # PDF-Input für Vision API vorbereiten
            input_config = vision.InputConfig(
                content=content,
                mime_type='application/pdf'
            )

            # Feature für PDF-Textextraktion konfigurieren
            features = [
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            ]

            # Request zum Extrahieren von Text aus allen Seiten
            request = vision.AsyncAnnotateFileRequest(
                input_config=input_config,
                features=features
            )

            # Asynchrone Operation starten
            operation = self.client.async_batch_annotate_files(requests=[request])

            logger.info(f"Asynchrone Google Vision PDF-Verarbeitung für {pdf_path} gestartet")

            # Warten auf Ergebnis
            response = operation.result(timeout=180)  # Timeout nach 3 Minuten

            # Extrahierten Text aus allen Seiten zusammenfügen
            full_text = ""
            page_count = 0

            for result in response.responses[0].responses:
                if result.full_text_annotation.text:
                    page_count += 1
                    page_text = result.full_text_annotation.text
                    full_text += f"--- Seite {page_count} ---\n{page_text}\n\n"

            logger.info(f"PDF-Textextraktion abgeschlossen: {pdf_path} ({page_count} Seiten, {len(full_text)} Zeichen)")

            return full_text, page_count

        except Exception as e:
            logger.error(f"Fehler bei der direkten PDF-Verarbeitung mit Google Vision: {str(e)}")
            raise

    def process_document_with_layout(self, pdf_path: str) -> Tuple[str, int]:
        """
        Erweiterte Verarbeitung eines PDF-Dokuments mit Layout-Erkennung.
        Versucht, Layout-Informationen wie Tabellen besser zu erhalten.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            Tuple[str, int]: (Extrahierter Text mit Layout-Informationen, Anzahl der Seiten)
        """
        try:
            # PDF-Datei öffnen und Inhalt als Bytes lesen
            with open(pdf_path, 'rb') as pdf_file:
                content = pdf_file.read()

            # PDF-Input für Vision API vorbereiten
            gcs_source = vision.GcsSource(uri=None)
            input_config = vision.InputConfig(
                content=content,
                mime_type='application/pdf'
            )

            # Features für detaillierte Layout-Erkennung konfigurieren
            features = [
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LAYOUT_DETECTION)
            ]

            # Output-Konfiguration (für detaillierte Informationen)
            # Bei JSONL-Output enthält die Ausgabe detaillierte Layout-Informationen
            output_config = vision.OutputConfig(
                batch_size=100  # Max. Anzahl Seiten pro Batch
            )

            # Asynchrone Anfrage erstellen
            request = vision.AsyncAnnotateFileRequest(
                input_config=input_config,
                features=features,
                output_config=output_config
            )

            # Asynchrone Operation starten
            operation = self.client.async_batch_annotate_files(requests=[request])

            logger.info(f"Asynchrone Google Vision Layout-Erkennung für {pdf_path} gestartet")

            # Warten auf Ergebnis
            response = operation.result(timeout=300)  # Timeout nach 5 Minuten

            # Extrahierten Text mit Layout-Informationen verarbeiten
            full_text = ""
            page_count = 0

            # Verarbeitet die Antwort und extrahiert Text mit Struktur
            for document in response.responses:
                for page in document.full_text_annotation.pages:
                    page_count += 1
                    page_text = f"--- Seite {page_count} ---\n"

                    # Text nach Blöcken organisieren und Tabellen erkennen
                    blocks_by_y = {}

                    for block in page.blocks:
                        if not block.bounding_box:
                            continue

                        # Block-Position ermitteln (für Tabellenerkennung)
                        vertices = block.bounding_box.vertices
                        block_y = sum(v.y for v in vertices) / len(vertices)
                        block_y_key = int(block_y / 10) * 10  # Ähnliche Y-Positionen gruppieren

                        # Text aus dem Block extrahieren
                        block_text = ""
                        for paragraph in block.paragraphs:
                            para_text = ""
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                para_text += word_text + " "
                            block_text += para_text.strip() + "\n"

                        # Nach Y-Position gruppieren (für Tabellenzeilen)
                        if block_y_key not in blocks_by_y:
                            blocks_by_y[block_y_key] = []

                        blocks_by_y[block_y_key].append({
                            'text': block_text.strip(),
                            'x': min(v.x for v in vertices)  # Linkeste X-Koordinate für Sortierung
                        })

                    # Jede Zeile (Blöcke mit ähnlicher Y-Position) verarbeiten
                    for y_key in sorted(blocks_by_y.keys()):
                        # Blöcke in dieser Zeile nach X-Koordinate sortieren
                        line_blocks = sorted(blocks_by_y[y_key], key=lambda b: b['x'])

                        # Wenn mehrere Blöcke in dieser Zeile sind, könnte es eine Tabellenzeile sein
                        if len(line_blocks) > 1:
                            # Mit Tabs verbinden, um tabellenartige Struktur zu erhalten
                            line_text = '\t'.join([block['text'] for block in line_blocks])
                        else:
                            # Einfacher Text
                            line_text = line_blocks[0]['text'] if line_blocks else ""

                        page_text += line_text + "\n"

                    full_text += page_text + "\n"

            logger.info(f"Layout-Erkennung abgeschlossen: {pdf_path} ({page_count} Seiten)")

            return full_text, page_count

        except Exception as e:
            logger.error(f"Fehler bei der Layout-Erkennung: {str(e)}")
            # Fallback auf einfache Textextraktion
            logger.info("Versuche Fallback auf einfache PDF-Verarbeitung")
            return self.process_pdf_document(pdf_path)