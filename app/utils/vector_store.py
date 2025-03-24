import os
from typing import List, Dict, Any
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, url, collection_name, api_key=None):
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
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
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Collection erstellen, wenn sie nicht existiert
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Standard-Größe für OpenAI-Embeddings
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{self.collection_name}' erstellt")

    def store_embeddings(self, chunk_embeddings: List[Dict[str, Any]]):
        """Embeddings in Qdrant speichern"""
        try:
            # Chunks in Batches von 100 speichern
            batch_size = 100
            for i in range(0, len(chunk_embeddings), batch_size):
                batch = chunk_embeddings[i:i + batch_size]

                points = []
                for idx, chunk in enumerate(batch):
                    points.append(
                        models.PointStruct(
                            id=idx + i,  # Unique ID
                            vector=chunk["embedding"],
                            payload={
                                "chunk_text": chunk["chunk_text"],
                                "source": chunk["source"],
                                "filename": chunk["filename"]
                            }
                        )
                    )

                # Punkte in die Collection upserten
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            logger.info(f"{len(chunk_embeddings)} Embeddings in Qdrant gespeichert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Embeddings: {str(e)}")
            raise

    def search_similar(self, query_embedding, limit=5):
        """Ähnliche Dokumente zu einem Query-Embedding finden"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            # Ergebnisse formatieren
            formatted_results = []
            for res in results:
                formatted_results.append({
                    "score": res.score,
                    "chunk_text": res.payload.get("chunk_text"),
                    "source": res.payload.get("source"),
                    "filename": res.payload.get("filename")
                })

            logger.info(f"{len(formatted_results)} ähnliche Dokumente gefunden")
            return formatted_results
        except Exception as e:
            logger.error(f"Fehler bei der Suche nach ähnlichen Dokumenten: {str(e)}")
            raise

    def clear_collection(self):
        """Collection leeren (nützlich für Neuladen der Daten)"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection_exists()
            logger.info(f"Collection '{self.collection_name}' zurückgesetzt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Zurücksetzen der Collection: {str(e)}")
            raise