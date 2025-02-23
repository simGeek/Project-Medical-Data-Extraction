document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM fully loaded and parsed');

    // Get references to HTML elements
    const uploadextractButton = document.getElementById('uploadextractButton');
    const fileInput = document.getElementById('file-upload');
    const fileFormatDropdown = document.getElementById('dropdown_id');
    const uploadMessage = document.getElementById('uploadMessage');
    const resultContainerkey = document.getElementById('extractionResultkey');
    const resultContainervalue = document.getElementById('extractionResultvalue');

    // Function to upload the file to EC2 and trigger extraction
    async function handleUploadAndExtract(file) {
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        // Check if the selected file is a PDF
        if (file.type !== 'application/pdf') {
            alert('Only PDF files are allowed.');
            return;
        }

        const fileFormat = fileFormatDropdown.value;

        // Prepare form data for upload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_format', fileFormat);

        try {
            const csrfToken = getCookie('csrftoken'); // Get CSRF token from cookies
            const response = await fetch('/uploadextract/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': csrfToken, // Include CSRF token in headers
                },
            });

            if (!response.ok) {
                throw new Error('Error during file upload or extraction.');
            }

            const data = await response.json();
            displayExtractionResultskey(data);
            displayExtractionResultsvalue(data);
            uploadMessage.innerText = 'File uploaded and extraction completed successfully!';
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    // Function to display extraction results in editable text boxes
    function displayExtractionResultskey(data) {
        resultContainerkey.innerHTML = ''; // Clear previous results
    
        Object.entries(data).forEach(([key]) => {
            const keyElement = document.createElement('div');
            keyElement.textContent = key;
            keyElement.classList.add('result-key'); // Class for styling
    
            resultContainerkey.appendChild(keyElement);
        });
    }
    
    function displayExtractionResultsvalue(data) {
        resultContainervalue.innerHTML = ''; // Clear previous results
    
        Object.entries(data).forEach(([_, value]) => {
            const inputElement = document.createElement('input');
            inputElement.type = 'text';
            inputElement.value = value;
            inputElement.classList.add('result-value'); // Class for styling
    
            resultContainervalue.appendChild(inputElement);
        });
    }
    

    // Utility function to get the CSRF token from cookies
    function getCookie(name) {
        return document.cookie
            .split('; ')
            .find(row => row.startsWith(name + '='))
            ?.split('=')[1];
    }

    // Event listener for 'Upload and Extract' button
    uploadextractButton?.addEventListener('click', async () => {
        const file = fileInput?.files[0];
        await handleUploadAndExtract(file);
    });
});
