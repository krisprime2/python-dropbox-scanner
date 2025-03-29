import json
import os
import logging
from google.cloud import vision
from google.cloud import storage
import uuid
from typing import Tuple, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleVisionHandler:
    def __init__(self, credentials_path=None, bucket_name=None):
        """
        Google Vision API Client und Storage Client initialisieren.

        Args:
            credentials_path (str, optional): Pfad zur Google Cloud Credentials JSON-Datei.
                Falls None, wird die Umgebungsvariable GOOGLE_APPLICATION_CREDENTIALS verwendet.
            bucket_name (str, optional): Name des GCS-Buckets für temporäre Dateien.
                Falls None, wird ein Standard-Bucket-Name verwendet.
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        self.client = vision.ImageAnnotatorClient()
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name or "pdf-vision-processing-temp"
        self._ensure_bucket_exists()

        logger.info("Google Vision API und Storage Clients initialisiert")

    def _ensure_bucket_exists(self):
        """Stellt sicher, dass der GCS-Bucket existiert"""
        try:
            # Prüfen, ob Bucket existiert
            if not self.storage_client.lookup_bucket(self.bucket_name):
                # Bucket erstellen, falls nicht vorhanden
                bucket = self.storage_client.create_bucket(self.bucket_name, location="eu")
                logger.info(f"Bucket {self.bucket_name} erfolgreich erstellt in {bucket.location}")
            else:
                logger.info(f"Bucket {self.bucket_name} existiert bereits")
        except Exception as e:
            logger.error(f"Fehler beim Überprüfen/Erstellen des Buckets: {str(e)}")
            raise

    def _upload_to_gcs(self, local_file_path: str) -> str:
        """
        Datei in den GCS-Bucket hochladen

        Args:
            local_file_path (str): Lokaler Pfad zur Datei

        Returns:
            str: GCS-URI der hochgeladenen Datei
        """
        try:
            # Zufälligen Dateinamen generieren, um Konflikte zu vermeiden
            file_name = f"temp_pdf_{uuid.uuid4().hex}_{os.path.basename(local_file_path)}"

            # Bucket-Referenz holen
            bucket = self.storage_client.bucket(self.bucket_name)

            # Blob (Datei) erstellen und hochladen
            blob = bucket.blob(file_name)
            blob.upload_from_filename(local_file_path)

            # GCS-URI für die hochgeladene Datei zurückgeben
            gcs_uri = f"gs://{self.bucket_name}/{file_name}"
            logger.info(f"Datei {local_file_path} nach {gcs_uri} hochgeladen")

            return gcs_uri
        except Exception as e:
            logger.error(f"Fehler beim Hochladen nach GCS: {str(e)}")
            raise

    def _delete_from_gcs(self, gcs_uri: str):
        """
        Temporäre Datei aus GCS löschen

        Args:
            gcs_uri (str): GCS-URI der zu löschenden Datei
        """
        try:
            # URI parsen
            if gcs_uri.startswith("gs://"):
                path_parts = gcs_uri[5:].split("/", 1)
                if len(path_parts) != 2:
                    logger.warning(f"Ungültiges GCS-URI-Format: {gcs_uri}")
                    return

                bucket_name, blob_name = path_parts

                # Bucket und Blob referenzieren
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                # Blob löschen
                blob.delete()
                logger.info(f"Temporäre Datei {gcs_uri} gelöscht")
            else:
                logger.warning(f"URI ist kein GCS-URI: {gcs_uri}")
        except Exception as e:
            logger.warning(f"Fehler beim Löschen von {gcs_uri}: {str(e)}")

    def process_pdf_document(self, pdf_path: str) -> Tuple[str, int]:
        """
        Prozessiert ein PDF-Dokument mit der Google Vision API über GCS.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            Tuple[str, int]: (Extrahierter Text, Anzahl der Seiten)
        """
        gcs_uri = None
        output_gcs_uri = None
        try:
            # 1. PDF nach GCS hochladen
            gcs_uri = self._upload_to_gcs(pdf_path)

            # 2. Asynchrone Anfrage zur Dokumentenerkennung vorbereiten
            input_config = vision.InputConfig(
                gcs_source=vision.GcsSource(uri=gcs_uri),
                mime_type='application/pdf'
            )

            # Features für Texterkennung konfigurieren
            features = [
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            ]

            # Output-Konfiguration hinzufügen (WICHTIG: Fehlender Teil im ursprünglichen Code)
            output_prefix = f"output-{uuid.uuid4().hex}"
            output_gcs_uri = f"gs://{self.bucket_name}/{output_prefix}/"

            output_config = vision.OutputConfig(
                gcs_destination=vision.GcsDestination(uri=output_gcs_uri),
                batch_size=100  # Max. Anzahl Seiten pro Batch
            )

            # 3. Anfrage erstellen (jetzt mit output_config)
            request = vision.AsyncAnnotateFileRequest(
                input_config=input_config,
                features=features,
                output_config=output_config  # Diese Zeile wurde hinzugefügt
            )

            # 4. Asynchrone Operation starten
            operation = self.client.async_batch_annotate_files(requests=[request])
            logger.info(f"Asynchrone Google Vision PDF-Verarbeitung für {pdf_path} gestartet")

            # 5. Auf Ergebnis warten (mit Timeout)
            operation_result = operation.result(timeout=180)  # Timeout nach 3 Minuten

            # 6. Ergebnisdateien aus GCS lesen
            output_bucket = self.storage_client.bucket(self.bucket_name)
            prefix = output_prefix + "/"
            blobs = list(output_bucket.list_blobs(prefix=prefix))

            # Überprüfen, ob Ergebnisdateien gefunden wurden
            if not blobs:
                logger.warning(f"Keine Ergebnisdateien in {output_gcs_uri} gefunden")
                return "", 0

            # 7. Extrahierten Text aus allen Ergebnisdateien zusammenfügen
            full_text = ""
            page_count = 0

            for blob in blobs:
                if blob.name.endswith(".json"):
                    # JSON-Datei herunterladen und parsen
                    json_content = blob.download_as_text()
                    response_json = json.loads(json_content)

                    # Text aus der JSON-Antwort extrahieren
                    if 'responses' in response_json:
                        for response in response_json['responses']:
                            if 'fullTextAnnotation' in response:
                                page_count += 1
                                page_text = response['fullTextAnnotation']['text']
                                full_text += f"--- Seite {page_count} ---\n{page_text}\n\n"

                    # Temporäre Ergebnisdatei löschen
                    blob.delete()

            logger.info(f"PDF-Textextraktion abgeschlossen: {pdf_path} ({page_count} Seiten, {len(full_text)} Zeichen)")
            return full_text, page_count

        except Exception as e:
            logger.error(f"Fehler bei der PDF-Verarbeitung mit Google Vision: {str(e)}")
            raise

        finally:
            # Temporäre Dateien aus GCS löschen
            if gcs_uri:
                self._delete_from_gcs(gcs_uri)

            # Versuchen, alle restlichen temporären Ausgabedateien zu löschen
            if output_gcs_uri:
                try:
                    output_bucket = self.storage_client.bucket(self.bucket_name)
                    output_prefix = output_gcs_uri.split("/")[-2]
                    blobs = list(output_bucket.list_blobs(prefix=output_prefix))
                    for blob in blobs:
                        blob.delete()
                    logger.info(f"Temporäre Ausgabedateien in {output_gcs_uri} gelöscht")
                except Exception as cleanup_error:
                    logger.warning(f"Fehler beim Löschen temporärer Ausgabedateien: {str(cleanup_error)}")

    def process_document_with_layout(self, pdf_path: str) -> Tuple[str, int]:
        """
        Erweiterte Verarbeitung eines PDF-Dokuments mit Layout-Erkennung.
        Versucht, Layout-Informationen wie Tabellen besser zu erhalten.

        Args:
            pdf_path (str): Pfad zur PDF-Datei

        Returns:
            Tuple[str, int]: (Extrahierter Text mit Layout-Informationen, Anzahl der Seiten)
        """
        gcs_uri = None
        output_gcs_uri = None
        try:
            # 1. PDF nach GCS hochladen
            gcs_uri = self._upload_to_gcs(pdf_path)

            # 2. Input für die Vision API konfigurieren
            input_config = vision.InputConfig(
                gcs_source=vision.GcsSource(uri=gcs_uri),
                mime_type='application/pdf'
            )

            # Features für detaillierte Layout-Erkennung konfigurieren
            features = [
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.LAYOUT_DETECTION)
            ]

            # Output-Konfiguration erstellen
            output_prefix = f"layout-{uuid.uuid4().hex}"
            output_gcs_uri = f"gs://{self.bucket_name}/{output_prefix}/"

            output_config = vision.OutputConfig(
                gcs_destination=vision.GcsDestination(uri=output_gcs_uri),
                batch_size=100  # Max. Anzahl Seiten pro Batch
            )

            # 3. Asynchrone Anfrage erstellen
            request = vision.AsyncAnnotateFileRequest(
                input_config=input_config,
                features=features,
                output_config=output_config
            )

            # 4. Asynchrone Operation starten
            operation = self.client.async_batch_annotate_files(requests=[request])
            logger.info(f"Asynchrone Google Vision Layout-Erkennung für {pdf_path} gestartet")

            # 5. Auf Ergebnis warten
            operation_result = operation.result(timeout=300)  # Timeout nach 5 Minuten

            # 6. Ergebnisdateien aus GCS lesen
            output_bucket = self.storage_client.bucket(self.bucket_name)
            prefix = output_prefix + "/"
            blobs = list(output_bucket.list_blobs(prefix=prefix))

            # Überprüfen, ob Ergebnisdateien gefunden wurden
            if not blobs:
                logger.warning(f"Keine Ergebnisdateien in {output_gcs_uri} gefunden")
                # Fallback auf einfache Textextraktion
                logger.info("Versuche Fallback auf einfache PDF-Verarbeitung")
                return self.process_pdf_document(pdf_path)

            # 7. Extrahierten Text mit Layout-Informationen verarbeiten
            full_text = ""
            page_count = 0

            for blob in blobs:
                if blob.name.endswith(".json"):
                    # JSON-Datei herunterladen und parsen
                    json_content = blob.download_as_text()
                    json_data = json.loads(json_content)

                    # Wenn Layout-Erkennung verfügbar ist
                    if 'responses' in json_data:
                        for response in json_data['responses']:
                            page_count += 1
                            page_text = f"--- Seite {page_count} ---\n"

                            # Versuche zuerst, fullTextAnnotation zu verwenden
                            if 'fullTextAnnotation' in response:
                                full_text_annotation = response['fullTextAnnotation']

                                # Page-Informationen parsen
                                if 'pages' in full_text_annotation:
                                    for page in full_text_annotation['pages']:
                                        # Blocks nach Y-Koordinate gruppieren für Tabellenerkennung
                                        blocks_by_y = {}

                                        if 'blocks' in page:
                                            for block in page['blocks']:
                                                if 'boundingBox' not in block:
                                                    continue

                                                # Block-Position ermitteln
                                                vertices = block['boundingBox']['vertices']
                                                block_y = sum(v.get('y', 0) for v in vertices) / len(vertices)
                                                block_y_key = int(block_y / 10) * 10  # Ähnliche Y-Positionen gruppieren

                                                # Text aus dem Block extrahieren
                                                block_text = ""
                                                if 'paragraphs' in block:
                                                    for paragraph in block['paragraphs']:
                                                        para_text = ""
                                                        if 'words' in paragraph:
                                                            for word in paragraph['words']:
                                                                word_text = ""
                                                                if 'symbols' in word:
                                                                    word_text = ''.join(
                                                                        [symbol.get('text', '') for symbol in
                                                                         word['symbols']])
                                                                para_text += word_text + " "
                                                        block_text += para_text.strip() + "\n"

                                                # Nach Y-Position gruppieren (für Tabellenzeilen)
                                                if block_y_key not in blocks_by_y:
                                                    blocks_by_y[block_y_key] = []

                                                blocks_by_y[block_y_key].append({
                                                    'text': block_text.strip(),
                                                    'x': min(v.get('x', 0) for v in vertices)
                                                    # Linkeste X-Koordinate für Sortierung
                                                })

                                        # Blöcke nach Y-Position zu Text zusammenfügen
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
                                else:
                                    # Einfach den gesamten Text verwenden, wenn keine strukturierten Daten verfügbar sind
                                    page_text += full_text_annotation.get('text', '')

                            # Fallback, wenn fullTextAnnotation nicht verfügbar ist
                            elif 'textAnnotations' in response and response['textAnnotations']:
                                page_text += response['textAnnotations'][0].get('description', '')

                            full_text += page_text + "\n"

                    # Temporäre Ergebnisdatei löschen
                    blob.delete()

            logger.info(f"Layout-Erkennung abgeschlossen: {pdf_path} ({page_count} Seiten)")
            return full_text, page_count

        except Exception as e:
            logger.error(f"Fehler bei der Layout-Erkennung: {str(e)}")
            # Fallback auf einfache Textextraktion
            logger.info("Versuche Fallback auf einfache PDF-Verarbeitung")
            return self.process_pdf_document(pdf_path)

        finally:
            # Temporäre Dateien aus GCS löschen
            if gcs_uri:
                self._delete_from_gcs(gcs_uri)

            # Versuchen, alle restlichen temporären Ausgabedateien zu löschen
            if output_gcs_uri:
                try:
                    output_bucket = self.storage_client.bucket(self.bucket_name)
                    output_prefix = output_gcs_uri.split("/")[-2]
                    blobs = list(output_bucket.list_blobs(prefix=output_prefix))
                    for blob in blobs:
                        blob.delete()
                    logger.info(f"Temporäre Ausgabedateien in {output_gcs_uri} gelöscht")
                except Exception as cleanup_error:
                    logger.warning(f"Fehler beim Löschen temporärer Ausgabedateien: {str(cleanup_error)}")