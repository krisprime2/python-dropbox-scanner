document.addEventListener('DOMContentLoaded', function() {
    // Elemente aus dem DOM abrufen
    const indexButton = document.getElementById('indexButton');
    const indexAllButton = document.getElementById('indexAllButton');
    const resetIndexButton = document.getElementById('resetIndexButton');
    const indexStatus = document.getElementById('indexStatus');
    const questionForm = document.getElementById('questionForm');
    const questionInput = document.getElementById('question');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const answerSection = document.getElementById('answerSection');
    const answerContent = document.getElementById('answerContent');
    const sourcesList = document.getElementById('sourcesList');

    // Event-Listener für das Indexieren neuer Dokumente
    indexButton.addEventListener('click', function() {
        indexDocuments(false, false);
    });

    // Event-Listener für das Indexieren aller Dokumente
    if (indexAllButton) {
        indexAllButton.addEventListener('click', function() {
            indexDocuments(false, true);
        });
    }

    // Event-Listener für das Zurücksetzen und Neuindexieren
    if (resetIndexButton) {
        resetIndexButton.addEventListener('click', function() {
            if (confirm('Möchten Sie wirklich den gesamten Index zurücksetzen und alle Dokumente neu indexieren?')) {
                indexDocuments(true, true);
            }
        });
    }

    // Funktion zum Indexieren von Dokumenten
    function indexDocuments(resetIndex, indexAll) {
        // UI aktualisieren
        disableIndexButtons(true);
        indexStatus.innerHTML = '<div class="alert alert-info">Indexierung gestartet...</div>';

        // API-Anfrage senden
        fetch('/api/index-documents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                reset_index: resetIndex,
                index_all: indexAll
            })
        })
        .then(response => response.json())
        .then(data => {
            // Erfolgs- oder Fehlermeldung anzeigen
            if (data.success) {
                // Detaillierte Erfolgsmeldung mit Statistiken
                let statsHtml = '';
                if (data.stats) {
                    statsHtml = `
                        <ul class="mt-2">
                            <li>Verarbeitete Dateien: ${data.stats.processed_files}</li>
                            <li>Übersprungene Dateien: ${data.stats.skipped_files}</li>
                            <li>Erstellte Chunks: ${data.stats.total_chunks}</li>
                            <li>Verarbeitungszeit: ${data.stats.processing_time} Sekunden</li>
                        </ul>
                    `;
                }
                indexStatus.innerHTML = `<div class="alert alert-success">${data.message}${statsHtml}</div>`;
            } else {
                indexStatus.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
            }
        })
        .catch(error => {
            indexStatus.innerHTML = `<div class="alert alert-danger">Fehler: ${error.message}</div>`;
            console.error('Fehler:', error);
        })
        .finally(() => {
            disableIndexButtons(false);
        });
    }

    // Funktion zum Deaktivieren/Aktivieren der Index-Buttons
    function disableIndexButtons(disabled) {
        indexButton.disabled = disabled;
        if (indexAllButton) indexAllButton.disabled = disabled;
        if (resetIndexButton) resetIndexButton.disabled = disabled;
    }

    // Event-Listener für das Frageformular
    questionForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const question = questionInput.value.trim();

        if (!question) {
            alert('Bitte geben Sie eine Frage ein.');
            return;
        }

        // UI aktualisieren
        loadingIndicator.classList.remove('d-none');
        answerSection.classList.add('d-none');

        // API-Anfrage senden
        fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Antwort anzeigen
                answerContent.textContent = data.answer;

                // Quellen anzeigen
                sourcesList.innerHTML = '';
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(source => {
                        const li = document.createElement('li');
                        li.className = 'list-group-item';
                        li.textContent = `${source.filename} (Relevanz: ${(source.score * 100).toFixed(1)}%)`;
                        sourcesList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.textContent = 'Keine spezifischen Quellen gefunden.';
                    sourcesList.appendChild(li);
                }

                answerSection.classList.remove('d-none');
            } else {
                alert(`Fehler: ${data.message}`);
            }
        })
        .catch(error => {
            alert(`Fehler bei der Anfrage: ${error.message}`);
            console.error('Fehler:', error);
        })
        .finally(() => {
            loadingIndicator.classList.add('d-none');
        });
    });
});