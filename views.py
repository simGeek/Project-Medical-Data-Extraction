# Importing important modules
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
import os
import cv2
import joblib
import re
import nltk
import pdfplumber
import torch
import io
import base64
from home.p1_extractor import extract_prescription, extract_patient_details, extract_medical, extract_vaccination
from .forms import ImageUploadForm
import numpy as np
from tensorflow import keras
from facenet_pytorch import MTCNN
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter, inch
from reportlab.pdfgen import canvas
from PIL import Image
from django.core.mail import send_mail
from django.http import HttpResponseRedirect
from django.urls import reverse
import logging

logger = logging.getLogger(__name__)

nltk.download('punkt')  # Tokenization
nltk.download('punkt_tab')
nltk.download('stopwords')  # Stopwords
nltk.download('wordnet')  # Lemmatization
nltk.download('omw-1.4')

# Creating important objects
stop_words = set(stopwords.words('english'))  # Set of English stopwords
stemmer = PorterStemmer()  # Stemmer for word stemming
lemmatizer = WordNetLemmatizer()  # Lemmatizer for word lemmatization

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenizer for BERT model
model = BertModel.from_pretrained('bert-base-uncased')  # BERT model for text embeddings


# Starting defining views

def index(request):
    return render(request, "projects.html")

def projects(request):
    return render(request, "projects.html")

def project1(request):
    return render(request, "project1.html")

def project2(request):
    return render(request, "project2.html")

def project3(request):
    return render(request, "project3.html")

def project4(request):
    return render(request, "project4.html")

def project5(request):
    return render(request, "project5.html")

def p1_expdfs(request):
    return render(request, "p1_expdfs.html")  

def p4_text_to_vector_bert(text, tokenizer, model):
    # Tokenize the input text and convert it to tensor format
    # 'return_tensors' specifies that the output should be in tensor format
    # 'truncation' ensures that text longer than the model's maximum length is truncated
    # 'padding' ensures that text shorter than the model's maximum length is padded
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Disable gradient calculations as we're not training the model
    # This reduces memory usage and speeds up computation
    with torch.no_grad():
        # Pass the tokenized inputs through the BERT model to obtain outputs
        outputs = model(**inputs)
    
    # Calculate the mean of all token embeddings to get a single vector representation
    # 'last_hidden_state' contains the embeddings for each token
    # 'mean(dim=1)' computes the average across all tokens
    # 'squeeze()' removes any singleton dimensions (e.g., dimensions with size 1)
    # 'numpy()' converts the tensor to a NumPy array
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def p4_convert_texts_to_vectors_bert(texts, tokenizer, model):
    # For each text in the list 'texts', convert it to a vector using the p4_text_to_vector_bert function
    # The list comprehension iterates over all texts and applies the conversion function
    vectors = [p4_text_to_vector_bert(text, tokenizer, model) for text in texts]
    
    # Convert the list of vectors (NumPy arrays) into a single NumPy array
    # This array will have shape (number_of_texts, vector_dimension) where vector_dimension is the size of the BERT vector
    return np.array(vectors)


def p4_preprocess_texts(texts):
    # Initialize an empty list to store preprocessed texts
    preprocessed_texts = []
    
    # Iterate over each text in the input list 'texts'
    for text in texts:
        # Convert the entire text to lowercase to ensure uniformity
        text = text.lower()
        
        # Remove punctuation using regular expressions
        # '[^\w\s]' matches any character that is not a word character or whitespace
        # This effectively removes punctuation marks
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers using regular expressions
        # '\d+' matches one or more digits
        # This effectively removes all numeric characters
        text = re.sub(r'\d+', '', text)
        
        # Tokenize the cleaned text into individual words (tokens)
        # This splits the text into a list of words
        tokens = word_tokenize(text)
        
        # Remove stop words from the list of tokens
        # Stop words are common words that are often removed during preprocessing
        # 'stop_words' is assumed to be a predefined list or set of stop words
        tokens = [word for word in tokens if word not in stop_words]
        
        # Perform lemmatization on each token
        # Lemmatization reduces words to their base or root form
        # 'lemmatizer' is assumed to be a predefined lemmatizer object
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join the tokens back into a single string with spaces between words
        preprocessed_text = ' '.join(tokens)
        
        # Append the preprocessed text to the list
        preprocessed_texts.append(preprocessed_text)
    
    # Return the list of preprocessed texts
    return preprocessed_texts

def p4_extract_text_from_pdf(file):
    # Initialize an empty string to accumulate text from the PDF
    pdf_text = ""

    # Open the PDF file from the given file stream
    # 'file' should be a file-like object (e.g., from a file upload)
    with pdfplumber.open(file) as pdf:
        # Iterate over each page in the PDF
        for page in pdf.pages:  # Directly iterate over pages
            # Extract text from the current page and add it to the accumulated text
            pdf_text += page.extract_text() or ""  # Handle case where no text is found
    
    # Return the accumulated text from all pages in the PDF
    return pdf_text

def p4_upload_plagi(request):
    # Check if the request method is POST (indicating form submission with files)
    if request.method == 'POST':
        # Check if 'file' is a key in the uploaded files dictionary
        if 'file' not in request.FILES:
            # Return a JSON response indicating no files were uploaded
            return JsonResponse({'message': 'No files uploaded'}, status=400)

        # Get the list of files uploaded with the key 'file'
        files = request.FILES.getlist('file')

        # Check if any files were selected for upload
        if not files:
            # Return a JSON response indicating no files were selected
            return JsonResponse({'message': 'No files selected'}, status=400)

        # Initialize lists and counters for uploaded files
        uploaded_files = []
        num_files_uploaded = 0
        invalid_files = []
        filenames = []

        # Define the directory where files will be uploaded
        upload_dir = settings.P4_PDF_FILES_UPLOAD_DIR
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            print(f'Created upload directory: {upload_dir}')

        # Create a FileSystemStorage object to handle file storage
        fs = FileSystemStorage(location=upload_dir)

        # Iterate over each uploaded file
        for file in files:
            # Check if the file has a '.pdf' extension
            if file.name.endswith('.pdf'):
                # Save the file to the defined upload directory
                filename = fs.save(file.name, file)
                filenames.append(filename)  # Store the filename for later reference
                file_path = os.path.join(upload_dir, filename)  # Full path to the uploaded file
                num_files_uploaded += 1  # Increment the count of successfully uploaded files
                uploaded_files.append(file_path)  # Save the full file path
                print(f'File uploaded: {file_path}')
            else:
                # Add invalid file names to the list of invalid files
                invalid_files.append(file.name)

        # Check if there were any invalid file formats uploaded
        if invalid_files:
            # Return a JSON response indicating which files were invalid
            return JsonResponse({
                'message': 'Invalid file formats uploaded (only PDF(s) allowed)',
                'invalid_files': invalid_files
            }, status=400)

        # Check if at least two PDF files were uploaded
        if num_files_uploaded < 2:
            # Return a JSON response indicating that at least two files are required
            return JsonResponse({'message': 'Upload at least two PDF files to check plagiarism...'}, status=400)

        # Instead of using session, we just return the file paths in the response
        return JsonResponse({'message': f'Successfully uploaded {num_files_uploaded} PDF files', 'uploaded_files': uploaded_files})

    # Return a JSON response indicating that the request method is not allowed
    return JsonResponse({'message': 'Invalid request method'}, status=405)

import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from django.shortcuts import render
from django.http import JsonResponse

def p4_detect_plagi(request):
    try:
        texts = []
        preprocessed_texts = []

        # Get PDF file paths
        files_folder = '/srv/portfolio_project/media/plagi_files/*.pdf'
        pdf_files = glob.glob(files_folder)

        if not pdf_files:
            return JsonResponse({'message': 'No uploaded files found.'}, status=400)

        # Extract filenames for reference
        fnames = [os.path.basename(pdf) for pdf in pdf_files]

        # Extract text from PDFs
        for pdf_path in pdf_files:
            with open(pdf_path, 'rb') as file:
                text = p4_extract_text_from_pdf(file)
                texts.append(text)

        # Preprocess extracted texts
        preprocessed_texts = p4_preprocess_texts(texts)

        # Convert texts to vectors using a BERT model
        vectors = p4_convert_texts_to_vectors_bert(preprocessed_texts, tokenizer, model)
        num_vectors = vectors.shape[0]

        # Compute pairwise Manhattan distances
        distances = np.zeros((num_vectors, num_vectors))
        results = []

        for i in range(num_vectors):
            for j in range(num_vectors):
                if i != j:
                    distances[i, j] = np.sum(np.abs(vectors[i] - vectors[j]))

        # Compare documents for plagiarism
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                distance = distances[i, j]
                is_plagiarized = distance < 50
                results.append((f"{fnames[i]} - {fnames[j]}", round(distance, 2), str(is_plagiarized)))

        # Create a heatmap
        distance_df = pd.DataFrame(distances, index=fnames, columns=fnames)
        plt.figure(figsize=(7, 4), dpi=80)
        sns.heatmap(distance_df, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8})

        plt.title('Manhattan Distance Matrix')
        plt.xlabel('Document Names')
        plt.ylabel('Document Names')

        # Save heatmap image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        buffer.seek(0)

        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response_data = {'text_result': results, 'visualize': img_str}
        return render(request, 'p4result.html', response_data)

    except Exception as e:
        return JsonResponse({'message': str(e)}, status=400)


def p4_textIn(request):
    return render(request, 'p4_textIn.html')

def p4_textInHandle(request):
    # Check if the request method is POST (indicating form submission with text data)
    if request.method == 'POST':
        # Retrieve the texts from the POST request
        text1 = request.POST.get('text1')
        text2 = request.POST.get('text2')
        
        # Combine the texts into a list
        texts = [text1, text2]
        
        # Preprocess the texts (e.g., lowercasing, removing punctuation)
        preprocessed_texts = p4_preprocess_texts(texts)
        
        # Convert the preprocessed texts to vectors using a BERT model
        vectors = p4_convert_texts_to_vectors_bert(preprocessed_texts, tokenizer, model)
        
        # Initialize a 2x2 matrix to store pairwise distances between vectors
        distances = np.zeros((2, 2))
        results = []

        # Compute pairwise distances between the vectors
        for i in range(2):
            for j in range(2):
                if i != j:
                    # Compute the Manhattan distance (sum of absolute differences) between vectors
                    distances[i, j] = np.sum(np.abs(vectors[i] - vectors[j]))

        # Generate results based on computed distances
        for i in range(2):
            for j in range(i + 1, 2):
                distance = distances[i, j]
                # Determine if the texts are considered plagiarized based on a threshold distance
                is_plagiarized = distance < 50
                results.append(f"Distance: {round(distance, 2)}")
                results.append(is_plagiarized)
            
        # Create a DataFrame for visualizing distances
        distance_df = pd.DataFrame(distances, index=['Text 1', 'Text 2'], columns=['Text 1', 'Text 2'])
        
        # Generate a heatmap visualization of the distance matrix
        plt.figure(figsize=(4, 4), dpi=80)
        sns.heatmap(distance_df, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 12, "color": "white"})
        
        # Customize text and labels for the heatmap
        plt.title('Manhattan Distance Matrix', color='white')
        plt.xlabel('Texts', color='white')
        plt.ylabel('Texts', color='white')
        
        # Customize color bar (for heatmap) to ensure visibility with a white background
        cbar = plt.gcf().axes[-1]  # The color bar is usually the last axis
        cbar.tick_params(labelcolor='white')  # Set color of color bar labels to white
        cbar.set_facecolor('none')  # Transparent background for color bar

        # Ensure ticks on the x and y axes are visible with a white color
        plt.gca().xaxis.set_tick_params(labelcolor='white')
        plt.gca().yaxis.set_tick_params(labelcolor='white')
        
        # Set the background color of the heatmap and figure to be transparent
        plt.gca().set_facecolor('none')
        plt.gcf().patch.set_facecolor('none')
        
        # Save the heatmap to a BytesIO object with a transparent background
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        buffer.seek(0)
        
        # Encode the saved heatmap image as a base64 string for embedding in HTML
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Prepare response data with results and heatmap image
        response_data = {
            'result': results,
            'heatmap': img_str
        }
        
        # Render the results to an HTML template with the response data
        return render(request, 'p4_textResult.html', response_data)
    
    # Return a JSON response indicating an error if the request method is not POST
    return JsonResponse({'error': 'An error occurred'})


def p4_generate_report(request):
    # Retrieve the plagiarism results and heatmap image string from the session
    results = request.session.get('results', [])
    heatmap_img_str = request.session.get('heatmap', '')

    # Create a BytesIO buffer to hold the PDF data
    pdf_buffer = io.BytesIO()
    
    # Create a PDF canvas with a letter-sized page
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter  # Dimensions of the letter-sized page

    # Set up title for the PDF
    c.setFont("Helvetica-Bold", 16)  # Font and size for the title
    title = "Plagiarism Report"
    title_width = c.stringWidth(title, "Helvetica-Bold", 16)  # Calculate width of the title
    c.drawString((width - title_width) / 2, height - 100, title)  # Center the title at the top of the page

    # Add results to the PDF
    c.setFont("Helvetica", 12)  # Font and size for the results
    y_position = height - 150  # Starting vertical position for results
    # Format each result for display
    text_lines = [f"{result[0]} - {result[1]} - Plagiarized: {result[2]}" for result in results]
    text_height = len(text_lines) * 20  # Approximate height needed for the results
    # Add each result line to the PDF
    for line in text_lines:
        line_width = c.stringWidth(line, "Helvetica", 12)  # Calculate width of the line
        c.drawString((width - line_width) / 2, y_position, line)  # Center the line horizontally
        y_position -= 20  # Move down for the next line

    # Add the heatmap image to the PDF if it exists
    if heatmap_img_str:
        # Decode the base64-encoded image string
        img_data = base64.b64decode(heatmap_img_str)
        img_buffer = io.BytesIO(img_data)  # Create a buffer for the image data

        # Define the path for saving the temporary image file
        media_dir = os.path.join(settings.MEDIA_ROOT, 'plagi_files')
        temp_img_path = os.path.join(media_dir, 'temp_heatmap.png')

        # Open and process the image
        with Image.open(img_buffer) as img:
            # Create a black background image slightly larger than the heatmap
            background_size = (int(img.width * 1.1), int(img.height * 1.1))  # Increase size by 10%
            black_background = Image.new('RGB', background_size, (0, 0, 0))
            
            # Calculate position to center the heatmap on the black background
            position = ((background_size[0] - img.width) // 2, (background_size[1] - img.height) // 2)
            
            # Paste the heatmap image onto the black background
            black_background.paste(img, position, img.convert('RGBA').getchannel('A') if img.mode == 'RGBA' else None)

            # Save the processed image
            black_background.save(temp_img_path, format='PNG')

            # Calculate the size and position of the image on the PDF
            img_width, img_height = black_background.size
            pdf_img_width = 6 * inch  # Width of the image on the PDF
            pdf_img_height = (pdf_img_width / img_width) * img_height  # Maintain aspect ratio
            x_position = (width - pdf_img_width) / 2  # Center horizontally
            y_position = (height - pdf_img_height) / 2 - 100  # Center vertically and adjust for title and results space

            # Draw the image on the PDF
            c.drawImage(temp_img_path, x_position, y_position, width=pdf_img_width, height=pdf_img_height)

    # Finalize the PDF
    c.showPage()  # End the current page
    c.save()  # Save the PDF to the buffer

    # Get the PDF data from the BytesIO buffer
    pdf = pdf_buffer.getvalue()
    pdf_buffer.close()  # Close the buffer

    # Create an HTTP response with the PDF file content
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="plagiarism_report.pdf"'  # Set the file name for the download
    
    # Clear temporary files used in the report generation
    p4_clear_plagi_files()

    return response  # Return the HTTP response containing the PDF


def p4_clear_plagi_files():
    # Define the directory where plagiarism files are stored
    plagi_files_dir = os.path.join(settings.MEDIA_ROOT, 'plagi_files')
    
    # Iterate through all files in the directory
    for filename in os.listdir(plagi_files_dir):
        # Construct the full file path
        file_path = os.path.join(plagi_files_dir, filename)
        
        try:
            # Check if the path is a file (not a directory)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
        except Exception as e:
            # Print an error message if file deletion fails
            print(f"Failed to delete {file_path}. Reason: {str(e)}")


def p3_clear_uploaded_images():
    # Define the directory where uploaded images are stored
    uploaded_images_directory = os.path.join(settings.MEDIA_ROOT, 'images')
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(uploaded_images_directory):
        # Construct the full file path for each file
        file_path = os.path.join(uploaded_images_directory, filename)
        
        try:
            # Check if the path is a file (not a directory)
            if os.path.isfile(file_path):
                # Remove the file from the filesystem
                os.remove(file_path)
        except Exception as e:
            # Print an error message if file deletion fails
            print(f"Failed to delete {file_path}. Reason: {str(e)}")

def p3_clear_processed_images():
    # Define the directory where processed images are stored
    processed_images_directory = os.path.join(settings.MEDIA_ROOT, 'processed_images')
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(processed_images_directory):
        # Construct the full file path for each file
        file_path = os.path.join(processed_images_directory, filename)
        
        try:
            # Check if the path is a file (not a directory)
            if os.path.isfile(file_path):
                # Remove the file from the filesystem
                os.remove(file_path)
        except Exception as e:
            # Print an error message if file deletion fails
            print(f"Failed to delete {file_path}. Reason: {str(e)}")

def p3_success(request):
    
    if request.method == 'GET':
        try:
            # Clear previously processed images
            p3_clear_processed_images()

            # Define the directory where uploaded images are stored
            images_directory = os.path.join(settings.MEDIA_ROOT, 'images')
            # List all files in the directory
            uploaded_images = os.listdir(images_directory)
            
            if uploaded_images:
                # Assuming processing the latest uploaded image
                latest_image_filename = uploaded_images[0]  # Modify this logic if needed to select the correct image
                
                # Construct the full path to the latest uploaded image
                uploaded_image_path = os.path.join(images_directory, latest_image_filename)
                
                # Load the pre-trained model for face expression recognition
                model = keras.models.load_model('ds_models/face_model.h5')
                class_names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 
                               'SURPRISE']

                # Initialize the MTCNN face detector
                mtcnn = MTCNN()
                # Read the image from the file path
                image = cv2.imread(uploaded_image_path)
                if image is None:
                    return JsonResponse({'error': 'Could not open or find the image'}, status=400)

                # Prepare for face detection and recognition
                output_image = image.copy()
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(rgb_image)
                if boxes is not None:
                    labels = []
                    boxes = boxes.astype(int)
                    for (x, y, w, h) in boxes:
                        # Draw bounding boxes around detected faces
                        cv2.rectangle(output_image, (x, y), (w, h), (0, 0, 255), 4)
                        # Extract face from the image
                        face = image[y:h, x:w]
                        # Resize face to match model input size
                        resized_face = cv2.resize(face, (96, 96))
                        reshaped_face = np.expand_dims(resized_face, axis=0)
                        # Predict face expression
                        prediction = model.predict(reshaped_face)
                        expression = np.argmax(prediction)
                        label = f"{class_names[expression]}"
                        labels.append(label)
                        # Draw the predicted label on the image
                        cv2.putText(output_image, label, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                 
                 
                    # Save the processed image to the 'processed_images' directory
                    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_images', latest_image_filename)
                    processed_image_url = os.path.join(settings.MEDIA_URL, 'processed_images', latest_image_filename)

                    # Write the output image to file
                    cv2.imwrite(processed_image_path, output_image)
                    # Remove the original uploaded image
                    os.remove(uploaded_image_path)
        
                    # Render the response with the processed image URL and detected labels
                    context = {'processed_image_url': processed_image_url, 'label': set(labels), 'file_name': latest_image_filename}
                    response = render(request, 'p3face.html', context)
                    return response
                
                else:
                    # If no faces are detected, remove the uploaded image and return an error
                    os.remove(uploaded_image_path)
                    return JsonResponse({'error': 'No faces detected'}, status=400)
                
            else:
                # If no uploaded images are found, return a 404 error
                return JsonResponse({'error': 'No uploaded images found in media/images'}, status=404)

        except Exception as e:
            # In case of an exception, remove the uploaded image (if it exists) and return an error
            if 'uploaded_image_path' in locals():
                os.remove(uploaded_image_path)
            print("Error:", str(e))
            return JsonResponse({'error': 'An error occurred while processing the image'}, status=500)

    else:
        # If the request method is not GET, return a 405 error
        if 'uploaded_image_path' in locals():
            os.remove(uploaded_image_path)
        return JsonResponse({'error': 'Invalid method'}, status=405)

def p2_regression(request):
    if request.method == 'POST':
        try:
            # Retrieve and convert form data from POST request
            field1 = request.POST.get('pm2dot5')
            field2 = request.POST.get('pm10')
            field3 = request.POST.get('no2')
            field4 = request.POST.get('nox')
            field5 = request.POST.get('nh3')
            field6 = request.POST.get('co')

            # Validate inputs (Check for missing or non-numeric values)
            if None in (field1, field2, field3, field4, field5, field6):
                raise ValueError("Missing input values")

            try:
                field1, field2, field3, field4, field5, field6 = map(float, (field1, field2, field3, field4, field5, field6))
            except ValueError:
                raise ValueError("Invalid input: All values must be numeric")

            # Load the pre-trained regression model
            model_path = os.path.join(settings.BASE_DIR, 'ds_models', 'aqi_model.joblib')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = joblib.load(model_path)

            # Prepare input data for prediction (Normalize based on predefined max values)
            input_data = [[
                field1 / 999.99, field2 / 1000.00, field3 / 432.30,
                field4 / 499.20, field5 / 485.52, field6 / 48.52
            ]]

            # Predict the AQI value
            prediction = model.predict(input_data)

            # Denormalize the prediction
            prediction = prediction * 818

            # Round the prediction to two decimal places
            prediction = round(prediction[0], 2)

            # Determine AQI category
            if 0 <= prediction <= 50:
                bucket = "Good"
            elif 51 <= prediction <= 100:
                bucket = "Satisfactory"
            elif 101 <= prediction <= 200:
                bucket = "Moderate"
            elif 201 <= prediction <= 300:
                bucket = "Poor"
            elif 301 <= prediction <= 400:
                bucket = "Very Poor"
            else:
                bucket = "Severe"

            # Prepare response data
            response_data = {
                'predictions': prediction,
                'bucket': bucket,
                'input_data': {
                    'pm2.5': field1, 'pm10': field2, 'no2': field3,
                    'nox': field4, 'nh3': field5, 'co': field6
                }
            }

            # Render p2aqi.html with prediction results
            return render(request, 'p2aqi.html', response_data)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            logger.error(error_message)  # Log the error
            return JsonResponse({'error': error_message}, status=400)

    # If not a POST request, return an error message
    return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)


def p3_imageUpload(request):
    # Clear previously uploaded images
    p3_clear_uploaded_images()
    
    # Check if the request method is POST
    if request.method == 'POST':
        # Create a form instance with POST data and uploaded files
        form = ImageUploadForm(request.POST, request.FILES)
        
        # Validate the form
        if form.is_valid():
            # Save the form data (i.e., the uploaded image)
            form.save()
            
            # Redirect to the 'p3_success' view upon successful upload
            return redirect('p3_success')
        else:
            # Print form validation errors to the console
            print("Form is not valid")
            print(form.errors)
    
    else:
        # For GET requests or other methods, instantiate a new empty form
        form = ImageUploadForm()
    
    # Render the 'project3.html' template with the form instance
    return render(request, 'project3.html', {'form': form})

def p1_uploadextract(request):
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
