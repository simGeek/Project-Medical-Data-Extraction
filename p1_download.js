// Wait until the entire DOM content is fully loaded
document.addEventListener('DOMContentLoaded', function () {    
    
    // Get references to download link elements from the HTML
    const download1 = document.getElementById('downloadFile1'); // Element for downloading Prescription File
    const download2 = document.getElementById('downloadFile2'); // Element for downloading Patient Details File
    const download3 = document.getElementById('downloadFile3'); // Element for downloading Vaccination Record File
    const download4 = document.getElementById('downloadFile4'); // Element for downloading Medical History File

    // Check if all download link elements are present
    if (download1 && download2 && download3 && download4) {
        // Add click event listener to the first download link
        download1.addEventListener('click', function() {
            var link = document.createElement('a'); // Create a new <a> element
            link.href = 'static/download_files/pre.pdf'; // Set the URL of the file to be downloaded
            link.download = 'pre.pdf'; // Set the default file name for the downloaded file
            document.body.appendChild(link); // Append the <a> element to the body
            link.click(); // Programmatically click the <a> element to trigger the download
            document.body.removeChild(link); // Remove the <a> element from the body
        });

        // Add click event listener to the second download link
        download2.addEventListener('click', function() {
            var link = document.createElement('a'); // Create a new <a> element
            link.href = 'static/download_files/pd.pdf'; // Set the URL of the file to be downloaded
            link.download = 'pd.pdf'; // Set the default file name for the downloaded file
            document.body.appendChild(link); // Append the <a> element to the body
            link.click(); // Programmatically click the <a> element to trigger the download
            document.body.removeChild(link); // Remove the <a> element from the body
        });

        // Add click event listener to the third download link
        download3.addEventListener('click', function() {
            var link = document.createElement('a'); // Create a new <a> element
            link.href = 'static/download_files/vr.pdf'; // Set the URL of the file to be downloaded
            link.download = 'vr.pdf'; // Set the default file name for the downloaded file
            document.body.appendChild(link); // Append the <a> element to the body
            link.click(); // Programmatically click the <a> element to trigger the download
            document.body.removeChild(link); // Remove the <a> element from the body
        });

        // Add click event listener to the fourth download link
        download4.addEventListener('click', function() {
            var link = document.createElement('a'); // Create a new <a> element
            link.href = 'static/download_files/mh.pdf'; // Set the URL of the file to be downloaded
            link.download = 'mh.pdf'; // Set the default file name for the downloaded file
            document.body.appendChild(link); // Append the <a> element to the body
            link.click(); // Programmatically click the <a> element to trigger the download
            document.body.removeChild(link); // Remove the <a> element from the body
        });
    } else {
        console.error('One or more download elements are not found.'); // Log an error if any download elements are missing
    }
});