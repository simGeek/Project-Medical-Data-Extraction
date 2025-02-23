from pdf2image import convert_from_path
import pytesseract
import home.util as util
from home.parser_classes import PrescriptionParser, PatientDetailsParser, MhParser, VrParser

POPPLER_PATH = r'C:\poppler-24.02.0\Library\bin'
pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract-OCR\tesseract.exe'


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
        processed_image = util.preprocess_pres_pd(page)
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
        processed_image = util.preprocess_pres_pd(page)
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
        processed_image = util.preprocess_vr(page)
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
        im1, imn, im3, im4, im5, im6, im7, im8, im9 = util.preprocess_mh(page)
        # Extract text from each preprocessed image using Tesseract
        text1 = pytesseract.image_to_string(im1, lang='eng')
        text2 = pytesseract.image_to_string(imn, lang='eng')
        text3 = pytesseract.image_to_string(im3, lang='eng')    
        text4 = pytesseract.image_to_string(im4, lang='eng')
        text5 = pytesseract.image_to_string(im5, lang='eng')
        text6 = pytesseract.image_to_string(im6, lang='eng')
        text7 = pytesseract.image_to_string(im7, lang='eng')
        text8 = pytesseract.image_to_string(im8, lang='eng') 
        text9 = pytesseract.image_to_string(im9, lang='eng')
        
        # Initialize a dictionary to store extracted data
        extracted_data = {}
        
        # Parse text from different sections using MhParser class
        extracted_data['Gender'] = MhParser(text2).parse3()
        extracted_data['Condition'] = MhParser(text3).parse4()
        extracted_data['Symptoms'] = MhParser(text4).parse5()
        extracted_data['Medication'] = MhParser(text5).parse6()
        extracted_data['Allergy'] = MhParser(text6).parse7()
        extracted_data['Tobacco'] = MhParser(text7).parse8()
        extracted_data['Drugs'] = MhParser(text8).parse9()
        
    else:
        # Raise an exception if no pages are found in the PDF
        raise Exception(f"Invalid document format: {file_format}")

    return extracted_data
