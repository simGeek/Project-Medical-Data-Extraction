🔹PROJECT-MEDICAL-DATA-EXTRACTION🔹

1️⃣ PURPOSE 

To extract useful information from the medical pdf documents.

2️⃣ PROBLEM 

Solves the problem of manual data retrieval from patient files during insurance claims, data analysis and medical records management.

3️⃣ TECH STACK 

🔹Frontend: HTML, JS, CSS

🔹Backend: Python, Django

🔹Infrastructure: AWS

4️⃣ LIBRARIES USED [See 'requirements.txt' for specific versions]

5️⃣ WORKFLOW

🔹 UPLOAD: The user uploads a medical PDF file.

🔹 PROCESSING: The system extracts relevant text and data using PDF parsing, OCR, and NLP techniques.

🔹 DATA EXTRACTION: Key information such as patient details, prescriptions, medical history, and vaccination records is identified and structured.

🔹 OUTPUT: The extracted data is formatted and returned in a readable format.

🔹 FILE REMOVAL: The uploaded file is deleted from storage after processing to ensure privacy.

6️⃣ INSTALLATION (with VS Code)

🔹Install:
    VS Code (https://code.visualstudio.com/download)
    Python (https://www.python.org/downloads/)

🔹Open VS Code

🔹Click on 'Clone Git Repository' and paste 'https://github.com/simGeek/Project-Medical-Data-Extraction.git' in the space prompted for the git url

🔹Hit 'Enter' and select a folder locally where to include the project files; open the folder in VS Code

🔹Open Terminal > New Terminal

🔹Run following commands:
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

🔹Click on 'Release-24.02.0-0.zip' after going to 'https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.02.0-0' on browser.
  Click on 'tesseract-ocr-w64-setup-5.5.0.20241111.exe' after going to 'https://github.com/UB-Mannheim/tesseract/wiki' on browser.
  Install both inside C drive.

🔹Run following commands:
    django-admin startproject my_project
    python manage.py startapp home
    
🔹Delete 'views.py' from 'home'; cut and paste 'urls.py' and 'views.py' from the cloned files to 'home'

🔹Create new folders inside the 'project-medical-data-extraction' named 'templates', 'static' and 'media'

🔹Create new folders named 'js', 'css' and 'download_files' inside 'static'

🔹Create folder 'pdf_files' in 'media'

🔹Cut and paste the following to the respective folders:
      .html files --> templates
      .css files --> css inside static
      .js files --> js inside static
      .pdf files --> download_files inside static
      .py files --> home (except manage.py)

🔹Add the following in settings.py inside my_project:
      import os
      STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
      MEDIA_URL = '/media/'
      MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
      PDF_FILES_UPLOAD_DIR = 'media/pdf_files/'

🔹Inside settings.py --> 'INSTALLED_APPS', include 'home'

🔹Inside settings.py --> 'TEMPLATES', paste 'os.path.join(BASE_DIR, 'templates')' in DIRS = [**PASTE HERE**]

🔹Add in 'my_project' --> 'urls.py',
        from django.contrib import admin
        from django.urls import path, include
        from django.conf import settings
        from django.conf.urls.static import static
        urlpatterns = [
          path('admin/', admin.site.urls),
          path('', include('home.urls')), 
        
  ]
        if settings.DEBUG:
            urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

🔹Run 'python manage.py migrate' in the VS Code terminal

🔹Run 'python manage.py runserver' in the VS Code terminal
    
7️⃣ CHALLENGES

8️⃣ KEY LEARNINGS

9️⃣ PROJECT IMAGES

![p1-1](https://github.com/user-attachments/assets/1712c051-0631-4cb9-8e37-b664e65152c2)

![p1-2](https://github.com/user-attachments/assets/b2533be9-3f04-452f-9ecc-d61cefb3dd0b)


