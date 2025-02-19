import pytest  # Import the pytest library, a testing framework for Python
from home.p1_parser_classes import PrescriptionParser, PatientDetailsParser, MhParser, VrParser

# Define a fixture to provide a PatientDetailsParser object for tests
@pytest.fixture()
def doc_1_kathy():
    # Sample text simulating a patient's medical record
    document_text = '''
    Patient Medical Record . : :

    Patient Information


    Birth Date
    Kathy Crawford May 6 1972
    (737) 988-0851 Weight:
    9264 Ash Dr 95
    New York City, 10005 a
    United States Height:
    190
    In Case of Emergency
    ee oe
    Simeone Crawford 9266 Ash Dr
    New York City, New York, 10005
    Home phone United States
    (990) 375-4621
    Work phone
    Genera! Medical History
    I i
    Chicken Pox (Varicella): Measies:
    IMMUNE IMMUNE

    Have you had the Hepatitis B vaccination?

    No

    List any Medical Problems (asthma, seizures, headaches):

    Migraine
    '''
    # Return an instance of PatientDetailsParser initialized with the sample text
    return PatientDetailsParser(document_text)

# Test the 'get_patient_name' method of PatientDetailsParser
def test_get_patient_name(doc_1_kathy):
    # Assert that the method returns the correct patient name
    assert doc_1_kathy.get_patient_name() == 'Kathy Crawford'

# Test the 'get_patient_phone_number' method of PatientDetailsParser
def test_get_patient_phone_number(doc_1_kathy):
    # Assert that the method returns the correct phone number
    assert doc_1_kathy.get_patient_phone_number() == '(737) 988-0851'

# Test the 'get_hepatitis_b_vaccination' method of PatientDetailsParser
def test_get_hepatitis_b_vaccination(doc_1_kathy):
    # Assert that the method returns the correct vaccination status
    assert doc_1_kathy.get_hepatitis_b_vaccination() == 'No'

# Test the 'get_medical_problems' method of PatientDetailsParser
def test_get_medical_problems(doc_1_kathy):
    # Assert that the method returns the correct list of medical problems
    assert doc_1_kathy.get_medical_problems() == 'Migraine'

# Test the 'parse' method of PatientDetailsParser
def test_parse(doc_1_kathy):
    # Call the 'parse' method and store the result
    record_kathy = doc_1_kathy.parse()
    # Assert that the parsed data matches the expected values
    assert record_kathy['patient_name'] == 'Kathy Crawford'
    assert record_kathy['phone_number'] == '(737) 988-0851'
    assert record_kathy['medical_problems'] == 'Migraine'
    assert record_kathy['hepatitis_b_vaccination'] == 'No'

# Define a fixture to provide an MhParser object for tests with sample document text
@pytest.fixture()
def doc_lalli1():
    document_text = 'Full Name Phone Number What is your Gender?\nLali Farah (91) 725-5945333 Male'
    # Return an instance of MhParser initialized with the sample text
    return MhParser(document_text)

# Define more fixtures for different document texts used in tests
@pytest.fixture()
def doc_lalli2():
    document_text = 'Check the conditions that apply to you or to any members of your immediate relatives:\n\nHypertension'
    return MhParser(document_text)

@pytest.fixture()
def doc_lalli3():
    document_text = 'Check the symptoms that youre currently experiencing:\n\nPsychiatric'
    return MhParser(document_text)

@pytest.fixture()
def doc_lalli4():
    document_text = '‘Ave you currently taking any medication?\n\nNo'
    return MhParser(document_text)

@pytest.fixture()
def doc_lalli5():
    document_text = 'Do you have any medication allergies?\nNo'
    return MhParser(document_text)

@pytest.fixture()
def doc_lalli6():
    document_text = 'Do you use or do you hava history of using tobacco?\n\nYes'
    return MhParser(document_text)

@pytest.fixture()
def doc_lalli7():
    document_text ='Do youuse or do you have history of using illegal drugs?\n\nNo'
    return MhParser(document_text)

@pytest.fixture()
def doc_lalli8():
    document_text = 'How often de you consume alcohol?\n\nOccasionally'
    return MhParser(document_text)

# Test the 'parse' method of MhParser for the full name field
def test_parse(doc_lalli1):
    # Assert that the parsed name matches the expected value
    assert doc_lalli1.parse() == 'Lali Farah'

# Test the 'parse2' method of MhParser for the phone number field
def test_parse2(doc_lalli1):
    # Assert that the parsed phone number matches the expected value
    assert doc_lalli1.parse2() == '(91) 725-5945333'

# Test the 'parse3' method of MhParser for the gender field
def test_parse3(doc_lalli1):
    # Assert that the parsed gender matches the expected value
    assert doc_lalli1.parse3() == 'Male'

# Test the 'parse4' method of MhParser for medical conditions
def test_parse4(doc_lalli2):
    # Assert that the parsed medical condition matches the expected value
    assert doc_lalli2.parse4() == 'Hypertension'

# Test the 'parse5' method of MhParser for symptoms
def test_parse5(doc_lalli3):
    # Assert that the parsed symptoms match the expected value
    assert doc_lalli3.parse5() == 'Psychiatric'

# Test the 'parse6' method of MhParser for medication status
def test_parse6(doc_lalli4):
    # Assert that the parsed medication status matches the expected value
    assert doc_lalli4.parse6() == 'No'

# Test the 'parse7' method of MhParser for allergy status
def test_parse7(doc_lalli5):
    # Assert that the parsed allergy status matches the expected value
    assert doc_lalli5.parse7() == 'No'

# Test the 'parse8' method of MhParser for tobacco use
def test_parse8(doc_lalli6):
    # Assert that the parsed tobacco use status matches the expected value
    assert doc_lalli6.parse8() == 'Yes'

# Test the 'parse9' method of MhParser for drug use
def test_parse9(doc_lalli7):
    # Assert that the parsed drug use status matches the expected value
    assert doc_lalli7.parse9() == 'No'

# Test the 'parse10' method of MhParser for alcohol consumption frequency
def test_parse10(doc_lalli8):
    # Assert that the parsed alcohol consumption frequency matches the expected value
    assert doc_lalli8.parse10() == 'Occasionally'

# Define a fixture to provide a PrescriptionParser object for tests
@pytest.fixture()
def doc_2_virat():
    # Sample text simulating a prescription
    document_text = '''
Dr John Smith, M.D

2 Non-Important street,
New York, Phone (900)-323-2222

Name:  Virat Kohli Date: 2/05/2022

Address: 2 cricket blvd, New Delhi

Omeprazole 40 mg

Directions: Use two tablets daily for three months
Refill: 3 times
'''
    # Return an instance of PrescriptionParser initialized with the sample text
    return PrescriptionParser(document_text)

# Define a fixture to provide an empty PrescriptionParser object for tests
@pytest.fixture()
def doc_3_empty():
    # Return an instance of PrescriptionParser initialized with an empty string
    return PrescriptionParser('')

# Test the 'get_field' method of PrescriptionParser for the patient name field
def test_get_name(doc_2_virat, doc_3_empty):
    # Assert that the method returns the correct patient name for a valid document
    assert doc_2_virat.get_field('patient_name') == 'Virat Kohli'
    # Assert that the method returns None for an empty document
    assert doc_3_empty.get_field('patient_name') is None

# Test the 'get_field' method of PrescriptionParser for the patient address field
def test_get_address(doc_2_virat, doc_3_empty):
    # Assert that the method returns the correct patient address for a valid document
    assert doc_2_virat.get_field('patient_address') == '2 cricket blvd, New Delhi'
    # Assert that the method returns None for an empty document
    assert doc_3_empty.get_field('patient_address') is None

# Test the 'get_field' method of PrescriptionParser for the medicines field
def test_get_medicines(doc_2_virat, doc_3_empty):
    # Assert that the method returns the correct medicines for a valid document
    assert doc_2_virat.get_field('medicines') == 'Omeprazole 40 mg'
    # Assert that the method returns None for an empty document
    assert doc_3_empty.get_field('medicines') is None

# Test the 'get_field' method of PrescriptionParser for the directions field
def test_get_directions(doc_2_virat, doc_3_empty):
    # Assert that the method returns the correct directions for a valid document
    assert doc_2_virat.get_field('directions') == 'Use two tablets daily for three months'
    # Assert that the method returns None for an empty document
    assert doc_3_empty.get_field('directions') is None

# Test the 'parse' method of PrescriptionParser for both valid and empty documents
def test_parse(doc_2_virat, doc_3_empty):
    # Call the 'parse' method for a valid document and store the result
    record_virat = doc_2_virat.parse()
    # Assert that the parsed data matches the expected values
    assert record_virat == {
        'patient_name': 'Virat Kohli',
        'patient_address': '2 cricket blvd, New Delhi',
        'medicines': 'Omeprazole 40 mg',
        'directions': 'Use two tablets daily for three months',
        'refills': '3'
    }
    # Call the 'parse' method for an empty document and store the result
    record_empty = doc_3_empty.parse()
    # Assert that the parsed data contains None values for an empty document
    assert record_empty == {
        'patient_name': None,
        'patient_address': None,
        'medicines': None,
        'directions': None,
        'refills': None
    }

# Define a fixture to provide a VrParser object for tests
@pytest.fixture()
def doc_jot():
    # Sample text simulating a vaccination record
    document_text = '''
    Name
    jot Kaur

    Date of Birth
    March 21, 2000

    Vaccination Record

    Date
    1st Dose 12-03-2021
    2nd Dose 24-04-2022

    3rd Dose

    Age Batch No,
    23 400
    Gender Patient ID
    Female 101
    Dosage Lot Number Manufacturer Location/site
    2 3000 abe Jalandhar
    1 4000 xyz Ludhiana

    Please keep this record card, it includes the medical information, details and the vaccine you have received. This
    card will show the next schedule of your vaccine. It is important to show this card to the next vaccination schedule

    for health officials to verify.

    Greate your own automated PDFs with Jotform PDF Editor- It's free

    % Jotform

    nm
    arr WUT ST ELE
    '''
    # Return an instance of VrParser initialized with the sample text
    return VrParser(document_text)

# Test the 'get_name' method of VrParser
def test_get_name(doc_jot):
    # Assert that the method returns the correct name
    assert doc_jot.get_name() == 'jot Kaur'

# Test the 'get_dob' method of VrParser
def test_get_dob(doc_jot):
    # Assert that the method returns the correct date of birth
    assert doc_jot.get_dob() == 'March 21, 2000'

# Test the 'get_dates' method of VrParser for vaccination dates
def test_get_dosage_dates(doc_jot):
    # Assert that the method returns the correct list of vaccination dates
    assert doc_jot.get_dates() == ['12-03-2021', '24-04-2022']

# Test the 'get_gender' method of VrParser
def test_get_gender(doc_jot):
    # Assert that the method returns the correct gender
    assert doc_jot.get_gender() == 'Female'

# Test the 'get_age' method of VrParser
def test_get_age(doc_jot):
    # Assert that the method returns the correct age
    assert doc_jot.get_age() == '23'

# Test the 'get_number' method of VrParser for dosage numbers
def test_get_dosage_number(doc_jot):
    # Assert that the method returns the correct list of dosage numbers
    assert doc_jot.get_number() == ['2', '1']

# Test the 'parse' method of VrParser
def test_parse(doc_jot):
    # Call the 'parse' method and store the result
    record = doc_jot.parse()
    # Assert that the parsed data matches the expected values
    assert record['name'] == 'jot Kaur'
    assert record['dob'] == 'March 21, 2000'
    assert record['dosage_dates'] == ['12-03-2021', '24-04-2022']
    assert record['gender'] == 'Female'
    assert record['age'] == '23'
    assert record['dosage_number'] == ['2', '1']