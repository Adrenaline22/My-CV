document.getElementById("registrationForm").onsubmit = function(event) {
    event.preventDefault();

    const formData = {
        firstName: document.getElementById("firstName").value,
        lastName: document.getElementById("lastName").value,
        email: document.getElementById("email").value,
        password: document.getElementById("password").value
    };

    fetch('/api/registration', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
        .then(response => {
            if (response.ok) {
                return response.text();
            } else {
                throw new Error('Ошибка регистрации');
            }
        })
        .then(data => {
            console.log('Успешный вход:', data);
            window.location.href = '/login';
        })
        .catch(error => {
            console.error('Ошибка:', error);
            const errorMessage = document.getElementById("errorMessage");
            errorMessage.style.display = "block";
            errorMessage.textContent = 'Электронная почта уже используется';
        });
};