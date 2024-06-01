import cv2 as cv
import numpy as np

def morphology_opening(image, kernel_size):
    """
    Perform morphological opening on the image.
    
    Parameters:
    image (numpy.ndarray): Input image (grayscale).
    kernel_size (int): Size of the structuring element.
    
    Returns:
    numpy.ndarray: Image after morphological opening.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return opening

def morphology_closing(image, kernel_size):
    """
    Perform morphological closing on the image.
    
    Parameters:
    image (numpy.ndarray): Input image (grayscale).
    kernel_size (int): Size of the structuring element.
    
    Returns:
    numpy.ndarray: Image after morphological closing.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return closing