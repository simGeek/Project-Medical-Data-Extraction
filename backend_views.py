# Importing important modules
import os
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from home.extractor import extract_prescription, extract_patient_details, extract_medical, extract_vaccination

def data_extract(request):
    return render(request, "data_extract.html")

def sample_download(request):
    return render(request, "sample_download.html")  

def uploadextract(request):
    if request.method == 'POST':
        # Check if 'file' exists in request.FILES
        if 'file' not in request.FILES:
            return JsonResponse({'message': 'No file uploaded'}, status=400)
        
        file = request.FILES['file']
        file_format = request.POST.get('file_format')

        # Only allow PDF files
        if file.name.endswith('.pdf'):
            upload_dir = settings.PDF_FILES_UPLOAD_DIR

            # Ensure upload directory exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            # Use FileSystemStorage to save the file
            fs = FileSystemStorage(location=upload_dir)
            filename = fs.save(file.name, file)
            file_url = fs.url(filename)

            # Get the path where the file is saved
            file_path = os.path.join(settings.MEDIA_ROOT, 'pdf_files', filename)

            # Process the file based on its format
            if file_format == 'vr':
                data = extract_vaccination(file_path, file_format)
            elif file_format == 'mh':
                data = extract_medical(file_path, file_format)
            elif file_format == 'pre':
                data = extract_prescription(file_path, file_format)
            elif file_format == 'pd':
                data = extract_patient_details(file_path, file_format)
            else:
                raise ValueError("Unsupported file format")

            # Optionally remove the file after processing (if needed)
            os.remove(file_path)

            return JsonResponse(data, safe=False)
        else:
            return JsonResponse({'message': 'Invalid file format. Only PDF files are allowed.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)
