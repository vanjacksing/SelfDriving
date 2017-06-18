import cv2
import numpy as np


def normalize_grayscale_hist(img):
    """
    Normalize RGB image histogram
    """
    np.random.seed(42)
    # Convert image to HSV
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Split image channels
    channels = cv2.split(yuv)
    # Normalize brightness channel of an image
    normalized = cv2.equalizeHist(channels[0])
    # Merge normalized channel back to the image
    merged = cv2.merge([normalized, channels[1], channels[2])
    # Convert image back to RGB
    merged_rgb = cv2.cvtColor(merged, cv2.COLOR_YUV2RGB)
    return channels[0]