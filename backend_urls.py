from django.contrib import admin  # Importing Django's admin module to manage the admin interface.
from django.urls import path  # Importing path function to define URL patterns.
from home import views  # Importing views from the home application to link to specific functions.

# Defining URL patterns for the home application.
urlpatterns = [
    path("data_extract", views.data_extract, name='data_extract'),  # URL for Project 1.
    path("sample_download", views.sample_download, name='sample_download'),
    
    # Paths related to Project 1 functionalities, handling uploads and data extraction.
    path('uploadextract/', views.uploadextract, name='uploadextract'),  # URL to upload data for Project 1.
]
