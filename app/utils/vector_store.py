import os
from typing import List, Dict, Any, Optional
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, url, collection_name, api_key=None, vector_size=1536):
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self.vector_size = vector_size  # Standard für OpenAI Embeddings
        self.client = self._get_client()
        self._ensure_collection_exists()

    def _get_client(self):
        """Qdrant-Client erstellen"""
        try:
            # Mit API-Schlüssel verbinden, falls vorhanden
            if self.api_key:
                client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                client = QdrantClient(url=self.url)

            logger.info(f"Verbindung zu Qdrant auf {self.url} hergestellt")
            return client
        except Exception as e:
            logger.error(f"Fehler bei der Verbindung zu Qdrant: {str(e)}")
            raise

    def _ensure_collection_exists(self):
        """Sicherstellen, dass die benötigte Collection existiert"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                # Collection erstellen, wenn sie nicht existiert
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    # Optimierte Indexierungsparameter für bessere Performanz
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,  # Auto-Indexierung nach 20.000 Vektoren
                        memmap_threshold=100000  # Memmap für große Kollektionen
                    )
                )
                logger.info(f"Collection '{self.collection_name}' erstellt")

                # Füge Payload-Indizes hinzu für effiziente Filterung
                self._create_payload_indexes()
            else:
                logger.info(f"Collection '{self.collection_name}' existiert bereits")

                # Überprüfen, ob Payload-Indizes existieren, ansonsten erstellen
                self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"Fehler bei der Collection-Überprüfung: {str(e)}")
            raise

    def _create_payload_indexes(self):
        """Erstellt Payload-Indizes für effiziente Filterung"""
        try:
            # Index für Dokumenttyp
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            # Index für Dateiname
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            # Index für Inhaltstyp
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            logger.info(f"Payload-Indizes für Collection '{self.collection_name}' erstellt")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Payload-Indizes: {str(e)}")

    def _ensure_payload_indexes(self):
        """Überprüft, ob Payload-Indizes existieren und erstellt sie ggf."""
        try:
            # Aktuelle Indizes abrufen
            collection_info = self.client.get_collection(self.collection_name)
            existing_indexes = collection_info.payload_schema if hasattr(collection_info, 'payload_schema') else {}

            # Prüfen, ob die benötigten Indizes existieren
            needed_indexes = ["doc_type", "filename", "content_type"]
            missing_indexes = [idx for idx in needed_indexes if idx not in existing_indexes]

            if missing_indexes:
                logger.info(f"Fehlende Payload-Indizes gefunden: {missing_indexes}")
                # Erstelle fehlende Indizes
                for field_name in missing_indexes:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                logger.info(f"Fehlende Payload-Indizes erstellt für Collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Fehler bei der Überprüfung der Payload-Indizes: {str(e)}")

    def store_embeddings(self, chunk_embeddings: List[Dict[str, Any]]):
        """Embeddings in Qdrant speichern mit optimierter Batching-Strategie"""
        try:
            # Batching-Größe anpassen für optimale Performanz
            batch_size = 100
            total_chunks = len(chunk_embeddings)

            logger.info(f"Starte Speicherung von {total_chunks} Embeddings in Batches von {batch_size}")
            start_time = time.time()

            for i in range(0, total_chunks, batch_size):
                batch = chunk_embeddings[i:i + batch_size]

                # Eindeutige IDs für jeden Vektor generieren
                points = []
                for idx, chunk in enumerate(batch):
                    # Generiere eine deterministische UUID basierend auf Dateiname und Chunk-Text
                    # Dies ermöglicht das Aktualisieren vorhandener Dokumente ohne Duplikate
                    unique_id_str = f"{chunk['filename']}:{chunk.get('page_number', 0)}:{chunk.get('section', '')}"
                    unique_id = abs(hash(unique_id_str)) % (10 ** 10)  # Positive 10-stellige Zahl

                    # Extrahiere Embedding
                    embedding = chunk.get("embedding")
                    if not embedding:
                        logger.warning(
                            f"Chunk ohne Embedding übersprungen: {chunk.get('filename')}:{chunk.get('section')}")
                        continue

                    # Bereite Payload vor (alle Metadaten außer dem Embedding)
                    payload = {k: v for k, v in chunk.items() if k != "embedding"}

                    # Füge Zeitstempel hinzu
                    payload["indexed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

                    points.append(
                        models.PointStruct(
                            id=unique_id,
                            vector=embedding,
                            payload=payload
                        )
                    )

                # Batch in Qdrant speichern
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True  # Auf Bestätigung warten für konsistente Daten
                    )
                    logger.info(
                        f"Batch {i // batch_size + 1}/{(total_chunks - 1) // batch_size + 1} gespeichert ({len(points)} Punkte)")

            total_time = time.time() - start_time
            logger.info(
                f"Alle {total_chunks} Embeddings in {total_time:.2f}s gespeichert (ca. {total_chunks / total_time:.1f} pro Sekunde)")

            # Collection-Statistiken ausgeben
            self._log_collection_stats()

            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Embeddings: {str(e)}")
            raise

    def _log_collection_stats(self):
        """Statistiken über die Collection ausgeben"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vectors_count = collection_info.vectors_count

            # Dokumenttypen zählen
            doc_types = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=0,  # Nur Anzahl abrufen
                with_payload=["doc_type"],
                with_vectors=False
            )

            # Einfache Zählung der Dokumenttypen
            doc_type_counts = {}
            for _, point in doc_types:
                doc_type = point.payload.get("doc_type", "unbekannt")
                doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

            logger.info(f"Collection '{self.collection_name}' enthält {vectors_count} Vektoren")
            logger.info(f"Verteilung nach Dokumenttypen: {json.dumps(doc_type_counts)}")
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Collection-Statistiken: {str(e)}")

    def search_similar(self, query_embedding, limit=5, filter_dict=None):
        """
        Ähnliche Dokumente zu einem Query-Embedding finden

        Args:
            query_embedding: Das Embedding der Suchanfrage
            limit: Maximale Anzahl zurückzugebender Ergebnisse
            filter_dict: Optionales Filter-Dictionary für die Suche
                         z.B. {"doc_type": "rechnung"} oder {"filename": "wichtig.pdf"}
        """
        try:
            # Filter für die Suche erstellen, falls vorhanden
            search_filter = None
            if filter_dict:
                filter_conditions = []

                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        # Mehrere Werte für den gleichen Schlüssel (ODER-Verknüpfung)
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        # Einzelner Wert
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

                search_filter = models.Filter(
                    must=filter_conditions
                )

            # Suche mit oder ohne Filter ausführen
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=search_filter,
                with_payload=True,  # Alle Payload-Daten zurückgeben
                score_threshold=0.6  # Nur Ergebnisse mit mindestens 60% Ähnlichkeit
            )

            # Ergebnisse formatieren
            formatted_results = []
            for res in results:
                # Alle Payload-Daten in das Ergebnis übernehmen
                result_dict = {
                    "score": res.score,
                    **res.payload  # Alle Felder aus dem Payload
                }

                formatted_results.append(result_dict)

            logger.info(f"{len(formatted_results)} ähnliche Dokumente gefunden"
                        f"{' mit Filter' if filter_dict else ''}")
            return formatted_results
        except Exception as e:
            logger.error(f"Fehler bei der Suche nach ähnlichen Dokumenten: {str(e)}")
            raise

    def clear_collection(self):
        """Collection leeren (nützlich für Neuladen der Daten)"""
        try:
            # Prüfen, ob Collection existiert
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name in collection_names:
                # Collection löschen und neu erstellen ist effizienter als alle Punkte zu löschen
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Collection '{self.collection_name}' gelöscht")

                # Collection neu erstellen mit derselben Konfiguration
                self._ensure_collection_exists()
                logger.info(f"Collection '{self.collection_name}' neu erstellt")

                return True
            else:
                # Collection existiert nicht, also erstellen
                self._ensure_collection_exists()
                logger.info(f"Collection '{self.collection_name}' existierte nicht und wurde erstellt")
                return True
        except Exception as e:
            logger.error(f"Fehler beim Zurücksetzen der Collection: {str(e)}")
            raise

    def filter_by_document_type(self, doc_type, limit=100, offset=0):
        """
        Dokumente nach Dokumenttyp filtern

        Args:
            doc_type: Der zu filternde Dokumenttyp (z.B. "rechnung", "vertrag")
            limit: Maximale Anzahl zurückzugebender Ergebnisse
            offset: Offset für Paginierung
        """
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_type",
                        match=models.MatchValue(value=doc_type)
                    )
                ]
            )

            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Keine Vektoren benötigt für Filteroperationen
            )

            # Ergebnisse und Pagination-Info formatieren
            documents = []
            for _, point in search_result[0]:
                documents.append(point.payload)

            total_count = len(documents)
            has_more = total_count == limit

            logger.info(f"{total_count} Dokumente vom Typ '{doc_type}' gefunden")

            return {
                "documents": documents,
                "total": total_count,
                "has_more": has_more,
                "offset": offset,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Fehler beim Filtern nach Dokumenttyp: {str(e)}")
            raise

    def get_document_stats(self):
        """
        Liefert Statistiken über die gespeicherten Dokumente
        """
        try:
            # Gesamtzahl der Vektoren
            collection_info = self.client.get_collection(self.collection_name)
            total_vectors = collection_info.vectors_count

            # Abfragen für verschiedene Statistiken
            # 1. Anzahl pro Dokumenttyp
            doc_types = {}
            doc_type_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=10000,  # Anpassen nach Bedarf
                with_payload=["doc_type"],
                with_vectors=False
            )[0]

            for _, point in doc_type_results:
                doc_type = point.payload.get("doc_type", "unbekannt")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            # 2. Anzahl pro Datei
            files = {}
            file_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=10000,
                with_payload=["filename"],
                with_vectors=False
            )[0]

            for _, point in file_results:
                filename = point.payload.get("filename", "unbekannt")
                files[filename] = files.get(filename, 0) + 1

            # 3. Anzahl pro Inhaltstyp
            content_types = {}
            content_type_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=10000,
                with_payload=["content_type"],
                with_vectors=False
            )[0]

            for _, point in content_type_results:
                content_type = point.payload.get("content_type", "text")
                content_types[content_type] = content_types.get(content_type, 0) + 1

            # Ergebnis zusammenstellen
            stats = {
                "total_vectors": total_vectors,
                "document_types": doc_types,
                "files_count": len(files),
                "content_types": content_types
            }

            logger.info(f"Dokumentstatistiken ermittelt: {total_vectors} Vektoren, {len(files)} Dateien")
            return stats

        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Dokumentstatistiken: {str(e)}")
            raise