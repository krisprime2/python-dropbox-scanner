document.addEventListener('DOMContentLoaded', function() {
    // Elemente aus dem DOM abrufen
    const indexButton = document.getElementById('indexButton');
    const indexStatus = document.getElementById('indexStatus');
    const questionForm = document.getElementById('questionForm');
    const questionInput = document.getElementById('question');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const answerSection = document.getElementById('answerSection');
    const answerContent = document.getElementById('answerContent');
    const sourcesList = document.getElementById('sourcesList');

    // Event-Listener für das Indexieren der Dokumente
    indexButton.addEventListener('click', function() {
        // UI aktualisieren
        indexButton.disabled = true;
        indexStatus.innerHTML = '<div class="alert alert-info">Indexierung gestartet...</div>';

        // API-Anfrage senden
        fetch('/api/index-documents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Erfolgs- oder Fehlermeldung anzeigen
            if (data.success) {
                indexStatus.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
            } else {
                indexStatus.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
            }
        })
        .catch(error => {
            indexStatus.innerHTML = `<div class="alert alert-danger">Fehler: ${error.message}</div>`;
            console.error('Fehler:', error);
        })
        .finally(() => {
            indexButton.disabled = false;
        });
    });

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