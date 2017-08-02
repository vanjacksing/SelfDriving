import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Image shape
IMG_SHAPE = (1280, 720)

# Horizontal offset for perspective transformation
H_OFFSET = 300
# Vertical offset for perspective transformation
V_OFFSET = 50

# Calibration chessboard dimensions
NX, NY = 9, 6

# Sobel magnitude filter threshold
MAG_THRESH = (0, 255)
# Sobel direction filter threshold
DIR_THRESH = (0, np.pi/2)
# Sobel absoulute axis threshold
ASB_THRESH = (0, 255)
# Hue and Suturation thresholds (HLS color model)
H_THRESH = (15, 100)
S_THRESH = (1, 255)

# Source image points for perspective warp
src = np.float32([[594, 450], [692, 450],  [1050, 675], [270, 675]])
# Destination image points for perspective warp
dst = np.float32([[H_OFFSET, V_OFFSET], [IMG_SHAPE[0]-H_OFFSET, V_OFFSET], 
                                     [IMG_SHAPE[0]-H_OFFSET, IMG_SHAPE[1]-V_OFFSET], 
                                     [H_OFFSET, IMG_SHAPE[1]-V_OFFSET]])


def get_camera_calibration(cal_list):
    # Get list of 2D coordinated for 3D points on chessboard images
    objpoints = []
    imgpoints = []
    objp = np.zeros((NX*NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1,2)
    for cb in cal_list:
        img = mpimg.imread(cb)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    return ret, mtx, dist, rvecs, tvecs

def get_perspective_transform_matrix():
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	return M, Minv

def abs_sobel_thresh(img, orient='x', thresh_min=ASB_THRESH[0], thresh_max=ASB_THRESH[1]):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=MAG_THRESH):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=DIR_THRESH):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_thresh(img, h_thresh=H_THRESH, s_thresh=S_THRESH):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    binary = np.zeros_like(H)
    binary[((H>=h_thresh[0]) & (H<=h_thresh[1])) & ((S>=s_thresh[0]) & (S<=s_thresh[1]))] = 1
    return binary

def thresh_combined(img):
    gradx = abs_sobel_thresh(img, thresh_min=20, thresh_max=100)
    grady = abs_sobel_thresh(img, thresh_min=20, thresh_max=100, orient='y')
    mag_binary = mag_thresh(img, mag_thresh=(70, 150))
    dir_binary = dir_threshold(img, thresh=(np.pi/8, np.pi/3))
    clr_thresh = color_thresh(img, s_thresh = (130, 255), h_thresh=(15, 100))
    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) & 
             (grady == 1) & 
             (dir_binary == 1) | 
             (mag_binary == 1) | 
             (clr_thresh == 1) |
             (gradx == 1)] = 1
    return combined