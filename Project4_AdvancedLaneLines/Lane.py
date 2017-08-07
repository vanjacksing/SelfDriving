import cv2
import numpy as np

class Line:
    
    N = 5
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
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
        self.current_fit = fit
        if (len(self.last_fitted) < Line.N):
            self.last_fitted.append(fit)
            self.best_fit = np.average(np.array(self.last_fitted), axis=0)
        else:
            temp = self.last_fitted[1:]
            temp.append(fit)
            self.last_fitted = temp
            self.best_fit = np.average(np.array(temp), axis=0, weights=self.weights)
			
class Lane:
    
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
        
    def set_frame(self, image):
        self.frame = image
        self.warped = self.pipe.process(image)
    
    def detect_line(self, is_left=True):
        
        def get_window(base, y_ticks, window):
            y_low = y_ticks[window]
            y_high = y_ticks[window+1]
            x_low = base - Lane.MARGIN
            x_high = base + Lane.MARGIN
            return y_low, y_high, x_low, x_high
        
        def count_pixels(nonzerox, nonzeroy, y_low, y_high, x_low, x_high):
            good_inds = ((nonzeroy >= y_high) & 
                         (nonzeroy < y_low)  &
                         (nonzerox >= x_low) & 
                         (nonzerox < x_high)).nonzero()[0]
            return good_inds
    
        def get_line_base(is_left=True):   
            # Get starting X coordinates for left and rigth lanes
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
                    if (len(good_inds)>Lane.MINPIX):
                        line_inds.append(good_inds)
                        basex = np.int(np.mean(nonzerox[good_inds]))
        line_inds = np.concatenate(line_inds)
        allx = nonzerox[line_inds]
        ally = nonzeroy[line_inds]
        if (len(allx)>Lane.DETECTION_THRESHOLD):
            fit = np.polyfit(ally, allx, 2)
            line.update_fits(fit)
            line.detected = True
            line.allx = allx
            line.ally = ally
        if is_left:
            self.left_line = line
        else:
            self.right_line = line
        return None
        
    def draw_lane(self, image):
        self.set_frame(image)
        self.detect_line(is_left=True)
        self.detect_line(is_left=False)
        if (self.left_line.detected & self.right_line.detected):
            warp_zero = np.zeros_like(self.warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            ploty = np.linspace(0, self.warped.shape[0]-1, self.warped.shape[0])
            left_fitx = self.left_line.best_fit[0]*ploty**2 + self.left_line.best_fit[1]*ploty + self.left_line.best_fit[2]
            right_fitx = self.right_line.best_fit[0]*ploty**2 + self.right_line.best_fit[1]*ploty + self.right_line.best_fit[2]
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
            