document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM fully loaded and parsed');

    // Get references to HTML elements
    const uploadextractButton = document.getElementById('uploadextractButton');
    const fileInput = document.getElementById('file-upload');
    const fileFormatDropdown = document.getElementById('dropdown_id');
    const uploadMessage = document.getElementById('uploadMessage');
    const resultContainer = document.getElementById('extractionResult');

    // Function to upload the file to EC2 and trigger extraction
    async function handleUploadAndExtract(file) {
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        // Check if the selected file is a PDF
        if (file.type !== 'application/pdf') {
            alert('Only PDF files are allowed.');
            return;  // Stop further execution if the file type is not PDF
        }

        const fileName = file.name;
        const fileFormat = fileFormatDropdown.value;

        // Prepare the data for upload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_format', fileFormat);  // Sending the file format from the dropdown

        try {
            const csrfToken = getCookie('csrftoken');  // Get CSRF token from the cookie
            const response = await fetch('/uploadextract/', {
                method: 'POST',
                body: formData,  // Sending form data as body for file upload and file format
                headers: {
                    'X-CSRFToken': csrfToken,  // Include CSRF token in headers
                },
            });

            if (response.ok) {
                uploadMessage.innerText = 'File uploaded and extraction completed successfully!';

                // Parse the response JSON
                const data = await response.json();

                resultContainer.innerHTML = ''; // Clear previous results

                // Loop through extracted data and create HTML elements for each key-value pair
                for (const [key, value] of Object.entries(data)) {
                    const keyElement = document.createElement('span');
                    keyElement.textContent = key;
                    keyElement.classList.add('result-key');

                    const valueElement = document.createElement('span');
                    valueElement.textContent = value;
                    valueElement.classList.add('result-value');

                    const pairElement = document.createElement('div');
                    pairElement.classList.add('result-pair');
                    pairElement.appendChild(keyElement);
                    pairElement.appendChild(valueElement);

                    resultContainer.appendChild(pairElement);
                }
            } else {
                alert('Error during file upload or extraction.');
            }
        } catch (error) {
            alert('Error during file upload: ' + error.message);
        }
    }

    // Utility function to get the CSRF token from the cookie
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Event listener for the 'Upload and Extract' button
    if (uploadextractButton && fileInput) {
        uploadextractButton.addEventListener('click', async function () {
            const file = fileInput.files[0];
            await handleUploadAndExtract(file); // Upload the file and trigger extraction
        });
    }
});
