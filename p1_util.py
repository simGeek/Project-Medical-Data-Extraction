import numpy as np  # Import the NumPy library for numerical operations
import cv2  # Import the OpenCV library for image processing

# Function to preprocess images for prescription and patient details
def preprocess_pres_pd(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image by scaling up by a factor of 1.5
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    
    # Apply adaptive thresholding to the resized image to get a binary image
    processed_image = cv2.adaptiveThreshold(
        resized,  # Input image
        255,  # Max value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding algorithm
        cv2.THRESH_BINARY,  # Thresholding type
        61,  # Block size (size of the local region used to calculate the threshold value)
        11  # Constant subtracted from the mean or weighted mean
    )
    return processed_image

# Function to preprocess images for vaccination records
def preprocess_vr(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image to its original size (no scaling)
    resized = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    
    # Apply adaptive thresholding to the resized image to get a binary image
    processed_img = cv2.adaptiveThreshold(
        resized,  # Input image
        255,  # Max value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding algorithm
        cv2.THRESH_BINARY,  # Thresholding type
        75,  # Block size (size of the local region used to calculate the threshold value)
        31  # Constant subtracted from the mean or weighted mean
    )
    return processed_img

# Function to apply global thresholding
def thresh_global(img):
    # Apply global thresholding to the input image
    _, result_image = cv2.threshold(
        img,  # Input image
        240,  # Threshold value
        255,  # Max value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
        cv2.THRESH_BINARY_INV  # Thresholding type (inverse binary thresholding)
    )
    return result_image

# Function to apply adaptive thresholding
def thresh_adaptive(img):
    # Apply adaptive thresholding to the input image
    result_image = cv2.adaptiveThreshold(
        img,  # Input image
        255,  # Max value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding algorithm
        cv2.THRESH_BINARY_INV,  # Thresholding type (inverse binary thresholding)
        13,  # Block size (size of the local region used to calculate the threshold value)
        2  # Constant subtracted from the mean or weighted mean
    )
    return result_image

# Function to preprocess images for medical history
def preprocess_mh(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image to a specific size (608x800)
    resized = cv2.resize(gray, (608, 800))
    
    # Extract various regions of interest (ROIs) from the resized image
    img1 = resized[157:217, 34:531]  # Crop the image to get the first ROI
    img2 = resized[231:287, 42:531]  # Crop the image to get the second ROI
    img3 = resized[296:348, 43:360]  # Crop the image to get the third ROI
    img4 = resized[358:419, 39:278]  # Crop the image to get the fourth ROI
    img5 = resized[428:482, 40:265]  # Crop the image to get the fifth ROI
    img6 = resized[494:548, 32:343]  # Crop the image to get the sixth ROI
    img7 = resized[559:610, 34:371]  # Crop the image to get the seventh ROI
    img8 = resized[628:687, 26:261]  # Crop the image to get the eighth ROI

    # Apply different thresholding methods to each ROI
    img_1 = thresh_adaptive(img1)  # Adaptive thresholding
    img_2 = thresh_adaptive(img2)  # Adaptive thresholding
    img_3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)  # Adaptive thresholding with specific parameters
    img_4 = thresh_adaptive(img4)  # Adaptive thresholding
    img_5 = thresh_global(img5)  # Global thresholding
    img_6 = thresh_global(img6)  # Global thresholding
    img_7 = thresh_adaptive(img7)  # Adaptive thresholding
    img_8 = cv2.adaptiveThreshold(img8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 24)  # Adaptive thresholding with specific parameters

    return img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8