import cv2
import numpy as np

def load_image_from_upload(uploaded_file) -> np.ndarray:
    """
    Decodes an uploaded image file into an OpenCV BGR numpy array.
    """
    if uploaded_file is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_file.seek(0)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

def preprocess_image(original_image: np.ndarray, target_size: tuple = (8, 8)) -> tuple:
    """
    Converts original image to grayscale and resizes to 8x8 using cv2.INTER_AREA.
    
    Returns:
        (gray_image, small_image, pixel_matrix)
    """
    if original_image is None:
        raise ValueError("Cannot preprocess a None image.")
        
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
    pixel_matrix = small_image.astype(np.uint8)
    
    return gray_image, small_image, pixel_matrix
