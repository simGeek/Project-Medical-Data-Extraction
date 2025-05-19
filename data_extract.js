document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM fully loaded and parsed');

    const uploadextractButton = document.getElementById('uploadextractButton');
    const fileInput = document.getElementById('file-upload');
    const fileFormatDropdown = document.getElementById('dropdown_id');
    const uploadMessage = document.getElementById('uploadMessage');
    const resultContainerkey = document.getElementById('extractionResultkey');
    const resultContainervalue = document.getElementById('extractionResultvalue');

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    async function handleUploadAndExtract(file) {
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        if (file.type !== 'application/pdf') {
            alert('Only PDF files are allowed.');
            return;
        }

        const fileFormat = fileFormatDropdown.value;
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_format', fileFormat);

        try {
            const csrfToken = getCookie('csrftoken');
            const response = await fetch('/uploadextract/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken,
                },
                body: formData,
                credentials: 'same-origin', // Important for CSRF
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            displayExtractionResultskey(data);
            displayExtractionResultsvalue(data);
            uploadMessage.innerText = 'File uploaded and extraction completed successfully!';
        } catch (error) {
            alert(`Upload failed: ${error.message}`);
            uploadMessage.innerText = '';
        }
    }

    function displayExtractionResultskey(data) {
        resultContainerkey.innerHTML = '';
        Object.keys(data).forEach(key => {
            const keyElement = document.createElement('div');
            keyElement.textContent = key;
            keyElement.classList.add('result-key');
            resultContainerkey.appendChild(keyElement);
        });
    }

    function displayExtractionResultsvalue(data) {
        resultContainervalue.innerHTML = '';
        Object.values(data).forEach(value => {
            const inputElement = document.createElement('input');
            inputElement.type = 'text';
            inputElement.value = value;
            inputElement.classList.add('result-value');
            resultContainervalue.appendChild(inputElement);
        });
    }

    uploadextractButton?.addEventListener('click', () => {
        const file = fileInput?.files[0];
        handleUploadAndExtract(file);
    });
});
