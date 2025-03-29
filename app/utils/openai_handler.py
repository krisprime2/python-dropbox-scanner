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
        """Verbesserte Antwortgenerierung basierend auf Kontext und Frage"""
        try:
            # Kontext besser strukturieren mit klaren Abschnitten und Metadaten
            context_blocks = []
            for i, doc in enumerate(context_texts):
                # Dokument-Metadaten hinzufügen für besseren Kontext
                doc_type = doc.get('doc_type', 'Unbekannt')
                doc_id = f"[Dokument {i + 1}]"
                section = doc.get('section', '')

                # Strukturierter Kontext mit Metadaten
                context_block = f"{doc_id} Datei: {doc['filename']} (Typ: {doc_type})\n"
                if section:
                    context_block += f"Abschnitt: {section}\n"
                context_block += f"Inhalt: {doc['chunk_text']}\n"

                # Score hinzufügen für Relevanzinformation
                if 'score' in doc:
                    context_block += f"Relevanz: {doc['score']:.2f}\n"

                context_blocks.append(context_block)

            # Verbesserte Kontext-Zusammenstellung
            context = "\n\n".join(context_blocks)

            # Verbesserter System-Prompt
            system_prompt = """
            Du bist ein erfahrener Dokumentenanalyst, der präzise Fragen zu Dokumenten beantwortet.

            ANWEISUNGEN:
            - Nutze die bereitgestellten Dokumentauszüge, um die Frage zu beantworten
            - Beziehe dich auf die Dokumente mit ihren Nummern (z.B. "Laut [Dokument 2]...")
            - Betrachte auch implizite oder indirekte Informationen in den Dokumenten
            - Verwende logische Schlussfolgerungen, wenn die Antwort nicht direkt im Text steht
            - Wenn du dir nicht sicher bist, teile mit, welche Teile sicher und welche unsicher sind
            - Wähle die relevantesten Dokumente aus (mit höherer Relevanz-Bewertung)
            - Wenn absolut keine relevanten Informationen verfügbar sind, antworte: "Die Dokumente enthalten keine Informationen zu dieser Frage."

            WICHTIG: Konzentriere dich darauf, nützliche Antworten zu geben. Selbst wenn nicht alle Details verfügbar sind, teile mit, was du aus den Dokumenten erschließen kannst.
            """

            # Token-Berechnung und Kontextkürzung falls nötig
            system_tokens = self.num_tokens(system_prompt)
            query_tokens = self.num_tokens(query)
            max_context_tokens = 15000 - system_tokens - query_tokens - 150  # Etwas größerer Sicherheitspuffer

            # Intelligentere Kontextkürzung, priorisiert nach Relevanz
            if self.num_tokens(context) > max_context_tokens:
                logger.warning(
                    f"Kontext zu groß ({self.num_tokens(context)} Tokens), wird gekürzt auf ~{max_context_tokens}")

                # Nach Relevanz sortieren, wenn vorhanden
                sorted_blocks = []
                for i, block in enumerate(context_blocks):
                    # Extrahiere Relevanz wenn vorhanden, sonst 0.5 als Standardwert
                    relevance = 0.5
                    if "Relevanz:" in block:
                        try:
                            relevance_str = block.split("Relevanz:")[1].strip().split("\n")[0]
                            relevance = float(relevance_str)
                        except:
                            pass
                    sorted_blocks.append((block, relevance, i))  # Block, Relevanz, ursprüngliche Position

                # Nach Relevanz sortieren, höhere zuerst, bei gleicher Relevanz original Reihenfolge behalten
                sorted_blocks.sort(key=lambda x: (-x[1], x[2]))

                # Neu zusammensetzen mit Priorisierung relevanterer Blöcke
                shortened_context = ""
                current_tokens = 0

                for block, _, _ in sorted_blocks:
                    block_tokens = self.num_tokens(block)
                    if current_tokens + block_tokens <= max_context_tokens:
                        shortened_context += block + "\n\n"
                        current_tokens += block_tokens
                    else:
                        # Wenn ein Dokument zu groß ist, versuche einen Teil davon einzufügen
                        if current_tokens < max_context_tokens * 0.8 and block_tokens > 1000:
                            # Teile den großen Block in Absätze
                            paragraphs = block.split("\n")
                            for para in paragraphs:
                                para_tokens = self.num_tokens(para + "\n")
                                if current_tokens + para_tokens <= max_context_tokens:
                                    shortened_context += para + "\n"
                                    current_tokens += para_tokens
                                else:
                                    break

                context = shortened_context

            # Verbesserte Nachricht mit einer kleinen Query-Reformulation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
    Hier sind die relevanten Dokumentauszüge:

    {context}

    Frage: {query}

    Um auch indirekte Informationen zu finden, überlege:
    1. Welche Dokumente könnten relevante Hinweise enthalten?
    2. Gibt es implizite Informationen oder könnte man logische Schlüsse aus den Dokumenten ziehen?
    3. Welche Schlüsselwörter aus der Frage könnten in anderer Form in den Dokumenten vorkommen?

    Bitte gib eine vollständige und hilfreiche Antwort basierend auf den verfügbaren Dokumenten.
                """}
            ]

            # OpenAI-API aufrufen mit etwas höherer Temperatur für flexiblere Antworten
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Etwas höher für flexiblere Antwortgenerierung
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            logger.info("Antwort generiert")
            return answer
        except Exception as e:
            logger.error(f"Fehler beim Generieren der Antwort: {str(e)}")
            raise