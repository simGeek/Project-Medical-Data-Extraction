import abc  # Import the Abstract Base Class (ABC) module for creating abstract base classes

# Define an abstract base class for medical document parsers
class MedicalDocParser(metaclass=abc.ABCMeta):
    def __init__(self, text):
        # Initialize the MedicalDocParser with the text to be parsed
        self.text = text  # Store the provided text in an instance variable

    @abc.abstractmethod
    def parse(self):
        # Define an abstract method 'parse' that must be implemented by any subclass
        # This method should contain the logic to extract and return relevant data from the text
        pass
        # The 'pass' statement is a placeholder indicating that this method is abstract and 
        # doesn't contain any implementation in the base class
