import os
import openai
from typing import List, Dict, Any
import logging
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIHandler:
    def __init__(self, api_key, model="gpt-3.5-turbo", embedding_model="text-embedding-ada-002"):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.encoding = tiktoken.encoding_for_model(model)

    def get_embedding(self, text: str) -> List[float]:
        """Text-Embedding mit OpenAI erstellen"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.info(f"Embedding für Text erstellt (Länge: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Embeddings: {str(e)}")
            raise

    def get_embeddings_batch(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch-Verarbeitung für Embeddings mehrerer Texte"""
        try:
            result_with_embeddings = []

            # Verarbeite in Batches von 20 (OpenAI-Empfehlung)
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                text_batch = [item["chunk_text"] for item in batch]

                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text_batch
                )

                # Füge Embeddings zu den ursprünglichen Daten hinzu
                for j, embedding_data in enumerate(response.data):
                    item_with_embedding = batch[j].copy()
                    item_with_embedding["embedding"] = embedding_data.embedding
                    result_with_embeddings.append(item_with_embedding)

            logger.info(f"Embeddings für {len(result_with_embeddings)} Texte erstellt")
            return result_with_embeddings
        except Exception as e:
            logger.error(f"Fehler beim Erstellen von Batch-Embeddings: {str(e)}")
            raise

    def num_tokens(self, text: str) -> int:
        """Anzahl der Tokens in einem Text berechnen"""
        tokens = self.encoding.encode(text)
        return len(tokens)

    def generate_answer(self, query: str, context_texts: List[Dict[str, Any]]) -> str:
        """Antwort basierend auf dem Kontext und der Frage generieren"""
        try:
            # Kontext aus den gefundenen relevanten Dokumenten erstellen
            context = "\n\n".join([f"Aus {doc['filename']}: {doc['chunk_text']}" for doc in context_texts])

            # System-Prompt, der das Modell einschränkt, nur auf den bereitgestellten Kontext zurückzugreifen
            system_prompt = """
            Du bist ein hilfreicher Assistent, der Fragen basierend auf den bereitgestellten Dokumenten beantwortet.
            Benutze NUR die Informationen aus den bereitgestellten Dokumenten, um die Frage zu beantworten.
            Wenn die Antwort nicht in den bereitgestellten Dokumenten zu finden ist, sage ehrlich: 
            "Ich kann diese Frage nicht beantworten, da die Information nicht in den bereitgestellten Dokumenten enthalten ist."
            Erfinde KEINE Informationen. Zitiere, aus welchem Dokument die Information stammt.
            """

            # Berechne Token und kürze Kontext falls nötig (ca. 16K Tokens ist das Limit für gpt-3.5-turbo)
            system_tokens = self.num_tokens(system_prompt)
            query_tokens = self.num_tokens(query)
            max_context_tokens = 15000 - system_tokens - query_tokens - 100  # Sicherheitspuffer

            # Kürze Kontext bei Bedarf
            if self.num_tokens(context) > max_context_tokens:
                logger.warning(
                    f"Kontext zu groß, wird gekürzt von {self.num_tokens(context)} auf ~{max_context_tokens} Tokens")
                # Teile den Kontext nach Absätzen
                paragraphs = context.split("\n\n")
                shortened_context = ""
                current_tokens = 0

                for para in paragraphs:
                    para_tokens = self.num_tokens(para)
                    if current_tokens + para_tokens <= max_context_tokens:
                        shortened_context += para + "\n\n"
                        current_tokens += para_tokens
                    else:
                        break

                context = shortened_context

            # Erstelle die Nachricht für den API-Aufruf
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hier sind die relevanten Dokumente:\n\n{context}\n\nFrage: {query}"}
            ]

            # OpenAI-API aufrufen
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,  # Niedrige Temperatur für konsistente, faktenbasierte Antworten
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            logger.info("Antwort generiert")
            return answer
        except Exception as e:
            logger.error(f"Fehler beim Generieren der Antwort: {str(e)}")
            raise