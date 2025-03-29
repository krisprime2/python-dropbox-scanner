import os
import dropbox
from dropbox.exceptions import AuthError, ApiError
from dotenv import load_dotenv

# .env-Datei laden
load_dotenv()


def check_dropbox_permissions(dbx):
    """Überprüft, ob das Token die richtigen Berechtigungen hat"""
    print("Überprüfe Dropbox-Berechtigungen...")

    # Liste der benötigten Berechtigungen
    required_scopes = ['files.metadata.read', 'files.content.read']

    try:
        # Token-Info abrufen
        token_info = dbx.check_user(query="echo")

        # Ausgabe der vorhandenen Scopes, wenn verfügbar
        if hasattr(token_info, 'scopes'):
            print("Vorhandene Berechtigungen:")
            for scope in token_info.scopes:
                print(f" - {scope}")

            # Überprüfen, ob alle erforderlichen Scopes vorhanden sind
            missing_scopes = [scope for scope in required_scopes if scope not in token_info.scopes]
            if missing_scopes:
                print("\nFEHLENDE BERECHTIGUNGEN:")
                for scope in missing_scopes:
                    print(f" - {scope}")
                print("\nBitte aktualisiere die Berechtigungen deiner App in der Dropbox App Console:")
                print("https://www.dropbox.com/developers/apps")
                print("Und generiere danach ein neues Access Token.")
                return False
            else:
                print("\nAlle erforderlichen Berechtigungen sind vorhanden.")
                return True
        else:
            print("Warnung: Konnte Berechtigungen nicht überprüfen. Versuche dennoch fortzufahren...")
            return True

    except Exception as e:
        print(f"Fehler bei der Berechtigungsprüfung: {str(e)}")
        return False


def list_dropbox_structure(access_token, start_path=''):
    """
    Listet alle Ordner und PDF-Dateien in Dropbox auf, beginnend mit einem bestimmten Pfad.
    """
    try:
        # Dropbox-Client initialisieren
        print("Verbinde mit Dropbox...")
        dbx = dropbox.Dropbox(access_token)

        # Testen, ob das Token gültig ist
        try:
            account = dbx.users_get_current_account()
            print(f"Verbunden mit Dropbox-Konto: {account.name.display_name}")
            print(f"E-Mail: {account.email}")
            print("-" * 50)
        except AuthError:
            print("ERROR: Ungültiges Access Token. Bitte prüfe dein Token in der .env-Datei.")
            return

        # Berechtigungen überprüfen
        if not check_dropbox_permissions(dbx):
            return

        # Zugriff auf "files/list_folder" testen
        print(f"\nTeste Zugriff auf Dropbox-Verzeichnis: {start_path or '/'}")
        try:
            # Versuche, den Inhalt des Startordners aufzulisten
            dbx.files_list_folder(start_path)
            print("Zugriff erfolgreich!")
        except ApiError as e:
            error_message = str(e)
            if "files.metadata.read" in error_message:
                print("FEHLER: Deine App hat nicht die erforderliche Berechtigung 'files.metadata.read'.")
                print("Bitte füge diese Berechtigung in der Dropbox App Console hinzu und generiere ein neues Token.")
            else:
                print(f"FEHLER beim Zugriff auf Verzeichnis: {error_message}")
            return

        # Struktur rekursiv ausgeben
        print("\nDropbox-Struktur:")
        print("-" * 50)
        list_structure_recursive(dbx, start_path, 0)

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {str(e)}")


def list_structure_recursive(dbx, path, level):
    """
    Gibt rekursiv die Ordner- und Dateistruktur aus.
    """
    try:
        # Ordnerinhalt abrufen
        res = dbx.files_list_folder(path)

        # Funktion zum Verarbeiten von Einträgen
        def process_entries(entries):
            for entry in entries:
                indent = "  " * level

                if isinstance(entry, dropbox.files.FolderMetadata):
                    # Ordner
                    print(f"{indent}📁 {entry.name} (Pfad: {entry.path_display})")
                    # Rekursiv in den Ordner gehen
                    list_structure_recursive(dbx, entry.path_lower, level + 1)

                elif isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                    # PDF-Datei
                    size_mb = entry.size / 1024 / 1024
                    print(f"{indent}📄 {entry.name} ({size_mb:.2f} MB) (Pfad: {entry.path_display})")

        # Erste Seite verarbeiten
        process_entries(res.entries)

        # Weitere Seiten verarbeiten, falls vorhanden
        while res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
            process_entries(res.entries)

    except dropbox.exceptions.ApiError as e:
        print(f"Fehler beim Durchsuchen von {path}: {str(e)}")


if __name__ == "__main__":
    # Access Token aus der .env-Datei lesen
    access_token = os.environ.get('DROPBOX_ACCESS_TOKEN')

    if not access_token:
        print("FEHLER: Kein Dropbox Access Token gefunden.")
        print("Bitte stelle sicher, dass eine .env-Datei mit DROPBOX_ACCESS_TOKEN vorhanden ist.")
        exit(1)

    # Standardpfad aus .env-Datei lesen oder Standardwert verwenden
    start_path = os.environ.get('DROPBOX_PDF_PATH', '')

    # Dropbox-Struktur ausgeben
    print(f"Durchsuche Dropbox (Startpfad: '{start_path or 'Wurzelverzeichnis'}')")
    list_dropbox_structure(access_token, start_path)