import cv2
import numpy as np
from ImageProcessing import IMG_SHAPE

class Line:
    """
    Class representint a lane line on the road
    """

    N = 5 # Number of fits, kept in Line's instance memory
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Curvature radius hist
        self.curv_hist = []
        # last fitted polynomial coefficients
        self.last_fitted = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = np.zeros(3)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Weights for averaging polynomial coefficients
        self.weights = np.array(range(1, Line.N+1))/np.sum(np.array(range(1, Line.N+1)))
    
    def update_fits(self, fit):
        """
        Update Line's instance fitted coefficients:
        1. Update current fit
        2. Add new fit to fitting history
        3. Calculate weighted average of all fits

        """
        # Update current fit
        self.current_fit = fit
        # Add new fit to fitting history and update weighted average
        if (len(self.last_fitted) < Line.N):
            self.last_fitted.append(fit)
            self.best_fit = np.average(np.array(self.last_fitted), axis=0)
        else:
            temp = self.last_fitted[1:]
            temp.append(fit)
            self.last_fitted = temp
            self.best_fit = np.average(np.array(temp), axis=0, weights=self.weights)

    def update_curvature(self):
        """
        Updates curvature radius based on its weighted average
        """
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*IMG_SHAPE[1]*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        if (len(self.curv_hist) < Line.N):
            self.curv_hist.append(curverad)
            self.radius_of_curvature = np.average(np.array(self.curv_hist))
        else:
            temp = self.curv_hist[1:]
            temp.append(curverad)
            self.curv_hist = temp
            self.radius_of_curvature = np.average(np.array(temp), axis=0, weights=self.weights)

			
class Lane:

    """
    Class representing a lane on the road
    Includes 2 Line objects for tracking, a processing Pipeline object
    and two images: source video frame and thresholded image with warped perspective
    """
    
    NWINDOWS = 9
    MARGIN = 100
    MINPIX = 50
    DETECTION_THRESHOLD = 5000
 
    def __init__(self, _pipe):
        self.frame = None
        self.warped = None
        self.pipe = _pipe
        self.left_line = Line()
        self.right_line = Line()
        self.offset = None

    def set_offset(self, offset):
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.offset = offset * xm_per_pix
        
    def set_frame(self, image):
        """
        Updates current frame
        """
        self.frame =  self.pipe.undistort(image)
        self.warped = self.pipe.process(self.frame)
    
    def detect_line(self, is_left=True):
        
        """
        Line detection code is based on a code from lectures
        """

        def get_window(base, y_ticks, window):
            """
            Helper function calculating boundaries for a search window
            """
            y_low = y_ticks[window]
            y_high = y_ticks[window+1]
            x_low = base - Lane.MARGIN
            x_high = base + Lane.MARGIN
            return y_low, y_high, x_low, x_high
        
        def count_pixels(nonzerox, nonzeroy, y_low, y_high, x_low, x_high):
            """
            Hepler function, countig the amount of white pixels in a given window
            """
            good_inds = ((nonzeroy >= y_high) & 
                         (nonzeroy < y_low)  &
                         (nonzerox >= x_low) & 
                         (nonzerox < x_high)).nonzero()[0]
            return good_inds
    
        def get_line_base(is_left=True):
            """
            Helper function that calculates starting X axis point for searching a line
            """
            histogram = np.sum(self.warped[self.warped.shape[0]//2:,:], axis=0)
            midpoint = np.int(histogram.shape[0]//2)
            if is_left:
                return np.argmax(histogram[:midpoint])
            return np.argmax(histogram[midpoint:]) + midpoint
        
        line = None
        if (is_left):
            line = self.left_line
        else:
            line = self.right_line
        nonzero = self.warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Starting x point for search
        basex = get_line_base(is_left)
        window_height = np.int(self.warped.shape[0]//Lane.NWINDOWS)
        shape = self.warped.shape
        line_inds = []
        y_ticks = np.linspace(shape[0], 0, num=Lane.NWINDOWS, dtype=np.int16)
        for window in range(Lane.NWINDOWS-1):
            # Get coordinated for window
            y_low, y_high, x_low, x_high = get_window(basex, y_ticks, window)
            # Get line pixels in given window
            good_inds = count_pixels(nonzerox, nonzeroy, y_low, y_high, x_low, x_high)
            # If enough pixels detected - proceed
            if (len(good_inds)>Lane.MINPIX):
                line_inds.append(good_inds)
                basex = np.int(np.mean(nonzerox[good_inds]))
            else:
                # If pixel count is low, lookup for last detected line
                if line.detected:
                    # Get fit params
                    fit = line.best_fit
                    # Calculate new basex based on previous line position
                    basex = fit[0]*(y_low**2) + fit[1]*y_low + fit[2]
                    # Get updated window coords
                    _, _, x_low, x_high = get_window(basex, y_ticks, window)
                    good_inds = count_pixels(nonzerox, nonzeroy, y_low, y_high, x_low, x_high)
                    # Update line base X position in case line segment is detected
                    if (len(good_inds)>Lane.MINPIX):
                        line_inds.append(good_inds)
                        basex = np.int(np.mean(nonzerox[good_inds]))
        line_inds = np.concatenate(line_inds)
        allx = nonzerox[line_inds]
        ally = nonzeroy[line_inds]
        # Update the line if total amount of detected pixels is above defined threshold
        if (len(allx)>Lane.DETECTION_THRESHOLD):
            fit = np.polyfit(ally, allx, 2)
            line.update_fits(fit)
            line.detected = True
            line.allx = allx
            line.ally = ally
            line.update_curvature()
        if is_left:
            self.left_line = line
        else:
            self.right_line = line
        return None
        
    def draw_lane(self, image):
        """
        Get the resulted image with detected lanes
        """
        self.set_frame(image)
        self.detect_line(is_left=True)
        self.detect_line(is_left=False)
        if (self.left_line.detected & self.right_line.detected):
            warp_zero = np.zeros_like(self.warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            ploty = np.linspace(0, self.warped.shape[0]-1, self.warped.shape[0])
            left_fitx = self.left_line.best_fit[0]*ploty**2 + self.left_line.best_fit[1]*ploty + self.left_line.best_fit[2]
            right_fitx = self.right_line.best_fit[0]*ploty**2 + self.right_line.best_fit[1]*ploty + self.right_line.best_fit[2]
            line_center = (right_fitx[-1] - left_fitx[-1])//2 + left_fitx[-1]
            offset = abs(IMG_SHAPE[0]//2 - line_center)
            self.set_offset(offset)
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.pipe.Minv, (self.warped.shape[1], self.warped.shape[0])) 
            result = cv2.addWeighted(self.frame, 1, newwarp, 0.3, 0)
            return result
        return image
            