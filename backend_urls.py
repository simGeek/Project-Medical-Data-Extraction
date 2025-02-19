from django.contrib import admin  # Importing Django's admin module to manage the admin interface.
from django.urls import path  # Importing path function to define URL patterns.
from home import views  # Importing views from the home application to link to specific functions.

# Defining URL patterns for the home application.
urlpatterns = [
    path("", views.index, name='home'),  # URL path for the homepage, mapped to the 'index' view.
    path("projects", views.projects, name='projects'),  # URL for the Projects page, handled by 'projects' view.

    # Project-specific URLs, each mapped to their respective views.
    path("project1", views.project1, name='project1'),  # URL for Project 1.
    path("project2", views.project2, name='project2'),  # URL for Project 2.
    path("project3", views.project3, name='project3'),  # URL for Project 3.
    path("project4", views.project4, name='project4'),  # URL for Project 4.
    path("project5", views.project5, name='project5'),  # URL for Project 5.
    path("p1_expdfs", views.p1_expdfs, name='p1_expdfs'),
    
    # Paths related to Project 1 functionalities, handling uploads and data extraction.
    path('p1_uploadextract/', views.p1_uploadextract, name='p1_uploadextract'),  # URL to upload data for Project 1.
        
    # Additional functionalities for Project 2 and subsequent projects.
    path('p2_regression/', views.p2_regression, name='p2_regression'),  # URL for regression analysis in Project 2.
    
    # Image upload functionality for Project 3.
    path('p3_imageUpload', views.p3_imageUpload, name='p3_imageUpload'),  # URL for image uploads in Project 3.
    path('p3_success', views.p3_success, name='p3_success'),  # URL for success page after image upload in Project 3.
    
    # Plagiarism detection and reporting for Project 4.
    path('p4_upload_plagi/', views.p4_upload_plagi, name='p4_upload_plagi'),  # URL for uploading files to check for plagiarism in Project 4.
    path('p4_detect_plagi/', views.p4_detect_plagi, name='p4_detect_plagi'),  # URL to detect plagiarism in uploaded content.
    path('p4_generate_report-report/', views.p4_generate_report, name='p4_generate_report'),  # URL for generating reports after plagiarism check.
    path('p4_textIn', views.p4_textIn, name='p4_textIn'),  # URL for text input page for plagiarism detection.
    path('p4_textInHandle', views.p4_textInHandle, name='p4_textInHandle'),  # URL to handle text input for plagiarism check.
]
