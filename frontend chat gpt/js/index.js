document.addEventListener('DOMContentLoaded', function() {
    const btn = document.getElementById("button");
    const questionsList = document.getElementById('questionsList');
    const toast = document.getElementById('toast');
    const workerScript = `
        onmessage = function(event) {
            const text = event.data.text;
            const delay = event.data.delay;
            let i = 0;

            function typing() {
                if (i < text.length) {
                    postMessage(text.charAt(i)); // Отправляем по одному символу
                    i++;
                    setTimeout(typing, delay);
                } else {
                    postMessage('\\n'); // В конце добавляем перенос строки
                }
            }

            typing();
        };
    `;

    function createWorker() {
        const blob = new Blob([workerScript], { type: 'application/javascript' });
        return new Worker(URL.createObjectURL(blob));
    }

    function typeText(element, text, callback) {
        const worker = createWorker();
        element.textContent = '';

        worker.postMessage({ text: text, delay: 10 });

        worker.onmessage = function(event) {
            element.textContent += event.data;
            if (event.data === '\n' && callback) {
                worker.terminate();
                callback();
            }
        };
    }

    btn.addEventListener("click", function () {
        // Добавляем CSS для управления разрывами страниц
        const style = document.createElement('style');
        style.textContent = `
            p {
                page-break-inside: avoid;
                page-break-before: auto;
                page-break-after: auto;
            }
        `;
        document.head.appendChild(style);

        const options = {
            margin: 1,
            filename: 'тестик.pdf',
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };
        html2pdf().from(questionsList).set(options).save().then(() => {
            document.head.removeChild(style); // Удаляем стиль после генерации PDF
        });
    });

    const form = document.getElementById('topicForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const topic = document.getElementById('topic').value;

        toast.classList.add('show');

        const response = await fetch(`/generate-questions?topic=${topic}`);
        const questions = await response.json();

        questionsList.innerHTML = '';

        toast.classList.remove('show');

        function printQuestionsSequentially(index) {
            if (index < questions.length) {
                const paragraph = document.createElement('p');
                questionsList.appendChild(paragraph);
                typeText(paragraph, questions[index], () => {
                    printQuestionsSequentially(index + 1);
                });
            }
        }

        printQuestionsSequentially(0);
    });
});
