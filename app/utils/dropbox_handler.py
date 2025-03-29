import os
import dropbox
from dropbox.exceptions import AuthError
from flask import current_app as app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DropboxHandler:
    def __init__(self, access_token):
        self.access_token = access_token
        self.dbx = self._get_dropbox_client()

    def _get_dropbox_client(self):
        """Dropbox-Client erstellen und verbinden"""
        try:
            dbx = dropbox.Dropbox(self.access_token)
            dbx.users_get_current_account()
            logger.info("Verbindung zu Dropbox hergestellt")
            return dbx
        except AuthError:
            logger.error("Fehler bei der Authentifizierung mit Dropbox")
            raise

    # In der Datei app/utils/dropbox_handler.py
    # Die list_pdf_files-Methode anpassen, um mit dem leeren Pfad zu arbeiten:

    def list_pdf_files(self, folder_path):
        """Alle PDF-Dateien im angegebenen Ordner und allen Unterordnern auflisten"""
        try:
            files = []

            # Initialer Aufruf mit dem übergebenen Ordnerpfad
            result = self.dbx.files_list_folder(folder_path)

            # Hilfsfunktion zum Verarbeiten von Einträgen
            def process_entries(entries):
                for entry in entries:
                    if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                        files.append({
                            'path': entry.path_lower,
                            'name': entry.name,
                            'id': entry.id
                        })
                    elif isinstance(entry, dropbox.files.FolderMetadata):
                        # Rekursiv in Unterordner schauen
                        try:
                            subresult = self.dbx.files_list_folder(entry.path_lower)
                            process_entries(subresult.entries)

                            # Weitere Seiten des Unterordners verarbeiten
                            while subresult.has_more:
                                subresult = self.dbx.files_list_folder_continue(subresult.cursor)
                                process_entries(subresult.entries)
                        except Exception as folder_error:
                            logger.warning(
                                f"Fehler beim Durchsuchen des Unterordners {entry.path_lower}: {str(folder_error)}")
                            continue

            # Erste Seite verarbeiten
            process_entries(result.entries)

            # Weitere Seiten verarbeiten, falls vorhanden
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                process_entries(result.entries)

            logger.info(f"{len(files)} PDF-Dateien in {folder_path} und Unterordnern gefunden")
            return files
        except Exception as e:
            logger.error(f"Fehler beim Auflisten der PDF-Dateien: {str(e)}")
            raise

    def download_pdf(self, file_path, output_path):
        """PDF-Datei von Dropbox herunterladen mit Dateinamen-Bereinigung"""
        try:
            # Dateiname bereinigen - ungültige Zeichen ersetzen
            # Hier ersetzen wir ungültige Zeichen durch Unterstriche oder andere zulässige Zeichen
            dir_name = os.path.dirname(output_path)
            file_name = os.path.basename(output_path)

            # Unzulässige Zeichen für Windows-Dateisystem ersetzen
            sanitized_name = file_name
            # Windows verbietet: \ / : * ? " < > |
            for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
                sanitized_name = sanitized_name.replace(char, '_')

            # Sanitized path erstellen
            sanitized_output_path = os.path.join(dir_name, sanitized_name)

            # Datei herunterladen
            with open(sanitized_output_path, 'wb') as f:
                metadata, response = self.dbx.files_download(file_path)
                f.write(response.content)

            logger.info(f"Datei {file_path} nach {sanitized_output_path} heruntergeladen")
            return sanitized_output_path
        except Exception as e:
            logger.error(f"Fehler beim Herunterladen der Datei {file_path}: {str(e)}")
            raise

    def download_all_pdfs(self, folder_path, output_folder):
        """Alle PDF-Dateien aus einem Dropbox-Ordner herunterladen"""
        pdf_files = self.list_pdf_files(folder_path)
        downloaded_paths = []

        for pdf in pdf_files:
            try:
                output_path = os.path.join(output_folder, pdf['name'])
                downloaded_path = self.download_pdf(pdf['path'], output_path)
                downloaded_paths.append(downloaded_path)
            except Exception as e:
                logger.error(f"Fehler beim Herunterladen von {pdf['path']}, überspringe Datei: {str(e)}")
                continue

        return downloaded_paths