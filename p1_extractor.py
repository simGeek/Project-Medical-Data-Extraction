from pdf2image import convert_from_path
import pytesseract
import home.p1_util as p1_util
from home.p1_parser_classes import PrescriptionParser, PatientDetailsParser, MhParser, VrParser

# Path to the Poppler executable required for pdf2image
POPPLER_PATH = r'C:\poppler-23.05.0\Library\bin'
# Path to the Tesseract-OCR executable for optical character recognition
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_prescription(file_path, file_format):
    """
    Extracts prescription data from a PDF file.
    
    :param file_path: Path to the PDF file.
    :param file_format: Format of the file (not used in this function).
    :return: Extracted prescription data.
    """
    # Convert the first page of the PDF file to an image
    pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    document_text = ''

    if len(pages) > 0:
        # Process the first page
        page = pages[0]
        # Preprocess the image for better OCR results
        processed_image = p1_util.preprocess_pres_pd(page)
        # Extract text from the image using Tesseract
        text = pytesseract.image_to_string(processed_image, lang='eng')
        document_text = '\n' + text

    # Parse the extracted text using the PrescriptionParser class
    extracted_data = PrescriptionParser(document_text).parse()

    return extracted_data

def extract_patient_details(file_path, file_format):
    """
    Extracts patient details from a PDF file.
    
    :param file_path: Path to the PDF file.
    :param file_format: Format of the file (not used in this function).
    :return: Extracted patient details.
    """
    # Convert the first page of the PDF file to an image
    pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    document_text = ''

    if len(pages) > 0:
        # Process the first page
        page = pages[0]
        # Preprocess the image for better OCR results
        processed_image = p1_util.preprocess_pres_pd(page)
        # Extract text from the image using Tesseract
        text = pytesseract.image_to_string(processed_image, lang='eng')
        document_text = '\n' + text

    # Parse the extracted text using the PatientDetailsParser class
    extracted_data = PatientDetailsParser(document_text).parse()

    return extracted_data

def extract_vaccination(file_path, file_format):
    """
    Extracts vaccination data from a PDF file.
    
    :param file_path: Path to the PDF file.
    :param file_format: Format of the file (not used in this function).
    :return: Extracted vaccination data.
    """
    # Convert the first page of the PDF file to an image
    pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    document_text = ''

    if len(pages) > 0:
        # Process the first page
        page = pages[0]
        # Preprocess the image for better OCR results
        processed_image = p1_util.preprocess_vr(page)
        # Extract text from the image using Tesseract
        text = pytesseract.image_to_string(processed_image, lang='eng')
        document_text = '\n' + text

    # Parse the extracted text using the VrParser class
    extracted_data = VrParser(document_text).parse()

    return extracted_data

def extract_medical(file_path, file_format):
    """
    Extracts medical information from a PDF file.
    
    :param file_path: Path to the PDF file.
    :param file_format: Format of the file (not used in this function).
    :return: Extracted medical information.
    :raises Exception: If the document format is invalid (no pages found).
    """
    # Convert the first page of the PDF file to an image
    pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)

    if len(pages) > 0:
        # Process the first page
        page = pages[0]
        # Preprocess the image for better OCR results
        im1, im2, im3, im4, im5, im6, im7, im8 = p1_util.preprocess_mh(page)
        # Extract text from each preprocessed image using Tesseract
        text1 = pytesseract.image_to_string(im1, lang='eng')
        text2 = pytesseract.image_to_string(im2, lang='eng')
        text3 = pytesseract.image_to_string(im3, lang='eng')
        text4 = pytesseract.image_to_string(im4, lang='eng')
        text5 = pytesseract.image_to_string(im5, lang='eng')
        text6 = pytesseract.image_to_string(im6, lang='eng')
        text7 = pytesseract.image_to_string(im7, lang='eng')
        text8 = pytesseract.image_to_string(im8, lang='eng')

        # Initialize a dictionary to store extracted data
        extracted_data = {}
        # Parse text from different sections using MhParser class
        extracted_data['Name'] = MhParser(text1).parse()
        extracted_data['Number'] = MhParser(text1).parse2()
        extracted_data['Gender'] = MhParser(text1).parse3()
        extracted_data['Condition'] = MhParser(text2).parse4()
        extracted_data['Symptoms'] = MhParser(text3).parse5()
        extracted_data['Medication'] = MhParser(text4).parse6()
        extracted_data['Allergy'] = MhParser(text5).parse7()
        extracted_data['Tobacco'] = MhParser(text6).parse8()
        extracted_data['Drugs'] = MhParser(text7).parse9()
        extracted_data['Alcohol'] = MhParser(text8).parse10()

    else:
        # Raise an exception if no pages are found in the PDF
        raise Exception(f"Invalid document format: {file_format}")

    return extracted_data
