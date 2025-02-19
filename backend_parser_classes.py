import re  # Import the regular expression module for pattern matching
from home.p1_parser_generic import MedicalDocParser  # Import the base parser class

# Define a parser for prescription documents
class PrescriptionParser(MedicalDocParser):
    def __init__(self, text):
        # Initialize the base class with the given text
        MedicalDocParser.__init__(self, text)

    def parse(self):
        # Parse the text and return a dictionary with the extracted fields
        return {
            'Patient Name': self.get_field('patient_name'),  # Extract patient name
            'Patient Address': self.get_field('patient_address'),  # Extract patient address
            'Medicines': self.get_field('medicines'),  # Extract medicine list
            'Directions': self.get_field('directions'),  # Extract directions for use
            'Refills': self.get_field('refills')  # Extract number of refills
        }

    def get_field(self, field_name):
        # Define regex patterns and flags for each field
        pattern_dict = {
            'patient_name': {'pattern': 'Name:(.*)Date', 'flags': 0},  # Regex to extract patient name
            'patient_address': {'pattern': 'Address:(.*)\n', 'flags': 0},  # Regex to extract patient address
            'medicines': {'pattern': '\| (.*)Directions', 'flags': re.DOTALL},  # Regex to extract medicines (dot matches newline)
            'directions': {'pattern': 'Directions:(.*)Refill', 'flags': re.DOTALL},  # Regex to extract directions (dot matches newline)
            'refills': {'pattern': 'Refill:(.*)times', 'flags': 0},  # Regex to extract refills
        }

        # Get the regex pattern and flags for the specified field
        pattern_object = pattern_dict.get(field_name)
        if pattern_object:
            # Use regex to find matches for the pattern
            matches = re.findall(pattern_object['pattern'], self.text, flags=pattern_object['flags'])
            if len(matches) > 0:
                # Return the first match after stripping extra whitespace
                return matches[0].strip()

# Define a parser for patient details documents
class PatientDetailsParser(MedicalDocParser):
    def __init__(self, text):
        # Initialize the base class with the given text
        MedicalDocParser.__init__(self, text)

    def parse(self):
        # Parse the text and return a dictionary with the extracted details
        return {
            'Patient Name': self.get_patient_name(),  # Extract patient name
            'Phone Number': self.get_patient_phone_number(),  # Extract phone number
            'Medical Problems': self.get_medical_problems(),  # Extract medical problems
            'Hepatitis Vaccination': self.get_hepatitis_b_vaccination()  # Extract Hepatitis B vaccination status
        }

    def get_patient_name(self):
        # Regex to extract the patient's name from the text
        pattern = 'Patient Information(.*?)\(\d{3}\)'
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        name = ''
        if matches:
            # Remove noise from the extracted name
            name = self.remove_noise_from_name(matches[0])
        return name

    def get_patient_phone_number(self):
        # Regex to extract the phone number from the text
        pattern = 'Patient Information(.*?)(\(\d{3}\) \d{3}-\d{4})'
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if matches:
            # Return the phone number part of the match
            return matches[0][-1]

    def remove_noise_from_name(self, name):
        # Remove unwanted text such as birth date from the extracted name
        name = name.replace('Birth Date', '').strip()  # Remove "Birth Date" text
        # Regex to match date patterns
        date_pattern = '((Jan|Feb|March|April|May|June|July|Aug|Sep|Oct|Nov|Dec)[ \d]+)'
        date_matches = re.findall(date_pattern, name)
        if date_matches:
            # Remove the date from the name
            date = date_matches[0][0]
            name = name.replace(date, '').strip()
        return name

    def get_hepatitis_b_vaccination(self):
        # Regex to extract the Hepatitis B vaccination status
        pattern = 'Have you had the Hepatitis B vaccination\?.*(Yes|No)'
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if matches:
            # Return the vaccination status
            return matches[0].strip()

    def get_medical_problems(self):
        # Regex to extract medical problems from the text
        pattern = 'List any Medical Problems .*?:(.*)'
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if matches:
            # Return the medical problems list
            return matches[0].strip()

# Define a parser for MH documents
class MhParser(MedicalDocParser):
    def __init__(self, text):
        # Initialize the base class with the given text
        MedicalDocParser.__init__(self, text)

    def parse(self):
        # Regex to extract data based on a specific pattern
        pattern = r'(.*) \(\d{2}\)'  # Extract text followed by a space and a two-digit number in parentheses
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse2(self):
        # Regex to extract phone numbers in a specific format
        pattern = r'\(\d{2}\) \d{3}-\d{7}'  # Extract phone numbers like (12) 345-6789012
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse3(self):
        # Regex to extract gender (Male or Female)
        pattern = r'Male|Female'  # Extract "Male" or "Female"
        flags = re.DOTALL
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse4(self):
        # Regex to extract text following two newlines
        pattern = r'\n\n(.*)$'  # Extract text after two newlines till the end of the string
        flags = re.DOTALL
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse5(self):
        # Regex to extract text following two newlines without DOTALL flag
        pattern = r'\n\n(.*)$'  # Extract text after two newlines till the end of the string
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse6(self):
        # Regex to extract Yes or No
        pattern = r'Yes|No'  # Extract "Yes" or "No"
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse7(self):
        # Regex to extract Yes or No
        pattern = r'Yes|No'  # Extract "Yes" or "No"
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse8(self):
        # Regex to extract Yes or No
        pattern = r'Yes|No'  # Extract "Yes" or "No"
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse9(self):
        # Regex to extract Yes or No
        pattern = r'Yes|No'  # Extract "Yes" or "No"
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

    def parse10(self):
        # Regex to extract text following two newlines without DOTALL flag
        pattern = r'\n\n(.*)$'  # Extract text after two newlines till the end of the string
        flags = 0
        matches = re.findall(pattern, self.text, flags=flags)
        if len(matches) > 0:
            # Return the first match
            return matches[0]

# Define a parser for VR documents
class VrParser(MedicalDocParser):
    def __init__(self, text):
        # Initialize the base class with the given text
        MedicalDocParser.__init__(self, text)

    def parse(self):
        # Parse the text and return a dictionary with vaccination details
        return {
            'Name': self.get_name(),  # Extract name
            'DOB': self.get_dob(),  # Extract date of birth
            'Dosage Dates': self.get_dates(),  # Extract dates of doses
            'Gender': self.get_gender(),  # Extract gender
            'Age': self.get_age(),  # Extract age
            'Dosage Number': self.get_number()  # Extract dosage number
        }

    def get_name(self):
        # Regex to extract the name from the text
        pattern = r'Name(.*?)Date'  # Extract text between "Name" and "Date"
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if len(matches) > 0:
            # Return the first match after stripping extra whitespace
            return matches[0].strip()

    def get_dob(self):
        # Regex to extract the date of birth from the text
        pattern = r'Birth(.*?)Vaccination'  # Extract text between "Birth" and "Vaccination"
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if len(matches) > 0:
            # Return the first match after stripping extra whitespace
            return matches[0].strip()

    def get_gender(self):
        # Regex to extract gender from the text
        pattern = r'ID(.*?)\d'  # Extract text between "ID" and a digit
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if len(matches) > 0:
            # Return the first match after stripping extra whitespace
            return matches[0].strip()

    def get_age(self):
        # Regex to extract age from the text
        pattern = r'Batch No,(.*?)\d{3}'  # Extract text between "Batch No," and a three-digit number
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if len(matches) > 0:
            # Return the first match after stripping extra whitespace
            return matches[0].strip()

    def get_number(self):
        # Regex to extract dosage number from the text
        pattern = r'(\d{1}) \d{4}'  # Extract a single digit followed by a space and four digits
        matches = re.findall(pattern, self.text, flags=re.DOTALL)
        if len(matches) > 0:
            # Return the list of matches
            return matches

    def get_dates(self):
        # Extract dates of doses from the text using multiple patterns
        pattern1 = r'1st|ist Dose (.*?)2nd'  # Extract text between "1st Dose" and "2nd"
        pattern2 = r'2nd Dose (.*?)3rd'  # Extract text between "2nd Dose" and "3rd"
        pattern3 = r'3rd Dose (.*?)Age'  # Extract text between "3rd Dose" and "Age"
        dates = []  # Initialize an empty list to hold dates
        # Find matches for each pattern
        matches1 = re.findall(pattern1, self.text, flags=re.DOTALL)
        matches2 = re.findall(pattern2, self.text, flags=re.DOTALL)
        matches3 = re.findall(pattern3, self.text, flags=re.DOTALL)
        try:
            # Append matches to the dates list if available
            dates.append(matches1[0].strip())
            dates.append(matches2[0].strip())
            dates.append(matches3[0].strip())
        except Exception as e:
            # Return an empty list if there are any exceptions
            return dates
