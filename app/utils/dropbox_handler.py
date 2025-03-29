import os
import dropbox
from dropbox.exceptions import AuthError
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

    def list_pdf_files(self, folder_path):
        """Alle PDF-Dateien im angegebenen Ordner auflisten"""
        try:
            files = []
            result = self.dbx.files_list_folder(folder_path)

            # Sammle alle PDF-Dateien
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                    files.append({
                        'path': entry.path_lower,
                        'name': entry.name,
                        'id': entry.id
                    })

            # Prüfe auf weitere Seiten
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                        files.append({
                            'path': entry.path_lower,
                            'name': entry.name,
                            'id': entry.id
                        })

            logger.info(f"{len(files)} PDF-Dateien in {folder_path} gefunden")
            return files
        except Exception as e:
            logger.error(f"Fehler beim Auflisten der PDF-Dateien: {str(e)}")
            raise

    def download_pdf(self, file_path, output_path):
        """PDF-Datei von Dropbox herunterladen"""
        try:
            with open(output_path, 'wb') as f:
                metadata, response = self.dbx.files_download(file_path)
                f.write(response.content)
            logger.info(f"Datei {file_path} nach {output_path} heruntergeladen")
            return output_path
        except Exception as e:
            logger.error(f"Fehler beim Herunterladen der Datei {file_path}: {str(e)}")
            raise

    def download_all_pdfs(self, folder_path, output_folder):
        """Alle PDF-Dateien aus einem Dropbox-Ordner herunterladen"""
        pdf_files = self.list_pdf_files(folder_path)
        downloaded_paths = []

        for pdf in pdf_files:
            output_path = os.path.join(output_folder, pdf['name'])
            self.download_pdf(pdf['path'], output_path)
            downloaded_paths.append(output_path)

        return downloaded_paths