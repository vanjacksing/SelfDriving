## Advanced Lane finding project writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All code for image processing, including camera calibration function is contained in module `ImageProcessing.py`
I've created a class `Pipeline`, which is a wrapped for image processing functions. On initilization it takes the name of folder, containing calibariton images, and then processes it to get calibration parameters.
I'm using OpenCV `calibrateCamera` function for this purpose. It takes the following as arguments:
1. objpoints - 3D points in real world
2. imgpoints - points on a 2D image
`objpoints` are initialized in assumption that all points lie in the same plane and have zero Z coordinate
`imgpoints` are calculated with OpenCV `findChessboardCorners` function, which takes calibration chessboard image as input and calculates coordinated of all corners of an image

All images can be found in Jupyter Notebook i provided [here](./Project 4 - Advanced Lane Finding.ipynb)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used OpenCV `undistort` function to create undistorted images, using parameters calculated with `calibrateCamera` function
The code is included in `ImageProcessing.py` module. Image samples are provided in a Notebook linked above.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of Sobel filters and got use of image conversion to HLS color space. The following was used in my workflow:
1. Binary mask based on Sobel filter on both X and Y axis
2. Image conversion to HLS model. Then channels were separted from each other, and each of them was thresholded to obtain binary mask. I also applied histogram normalization to Lightness channel, so than lane lines would always be bright enough to be detected. That was so because images might have different exposures.
All example images are present in a provided notebook

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
I included image wrapping code to `ImageProcessing.py` module. Transformation matricies are calculated on `Pipeline` class instance creation, using `getPerspectiveTransform` method provided by OpenCV. Actually, my method returned both direct and inverse transormation martices. I defined required source points by lane lines on an image of a flat road with straight lines. Destination points were calculated based on original image size minus special horizontal and vertical offset constants. Those points coordinates are also defined in `ImageProcessing.py` module. Using this i obtained parallel lines for both straight and curved lane lines.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This was done in `Lane.py` module. It includes `Line` and `Lane` classes. `Line` class contains info about each line, and some methods for updating its parameters, and `Lane` class contains 2 instances of `Line`. It also makes use of `Pipeline` class, which is used for processing the image. All detection code is also contained in that class. Detected line pixels were fitted using numpy's `polyfit` method with 2 degree polynomial.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature and vehicle offset were calculated in `Lane` class.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is provided in a Notebook

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_lanes.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The solution is quite far from ideal, but at least it can serve as a proof of concept. There's still a lot work to do. That may be following steps:
1. Making use of other image processing techniques. Maybe applying some kind of autoexposure and autocontrast for all of the images, or making use of different overlay modes, or trying to apply Sobel to only to grayscale image, but also different color channels, such as Saturation in HLS
2. Trying different lane searching techniques may help. Quick guess is something like k-means with modified distance metrics, so that it would search for streched along Y axis clusters, for example.
3. Trying other techniques for smoothing lines. I used moving average for averaging polynomial coefficients. Maybe other line fitting algorithms can help to improve detection quality.
