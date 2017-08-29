**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicles]: ./examples/vehicles.png
[not_vehicles]: ./examples/not_vehicles.png
[multiscale_bb]: ./examples/bboxes_multiscale.png
[heatmap]: ./examples/heatmap.png
[bboxes_result]: ./examples/bboxes_result.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for feature extraction is identified in `single_img_features` function in project IPython notebook.
I used spatial and HOG features for training linear SVM classifier.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle images][vehicles]
![Not vehicle images][not_vehicles]


####2. Explain how you settled on your final choice of HOG parameters.

For submission i used the following parameters, as they seemed reasonable for me:
 - (32, 32) spatial resolution for spatial features
 - (8, 8) cells with (8,8) cells per block for HOG feature extraction
 - HLS color space

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I decided not to reinvent the wheel and use Linear SVM for classification. I used regularization parameter `C` equal to 0.0001, as it gave best performance on test set. This could be explained, because training data variance is quite noticeable and chosing that value for `C` parameter helped to prevent overfitting. The code is provided in IPython notebook.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use the approach that was shown in lectures: first calculate HOG features for separate channels of the image, and then use sliding window with combination of scaling to search for vehicles. Here is an image with bounding boxes drawn:

![Image with multi-scale bounding boxes][multiscale_bb]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

At first my classifier produced too much false positives, and i decided to do some hard negative mining. I added positive samples saving to my pipeline code and got around 10000 images. Nearly 3000 of them turned out to be false positives. I added them to my training data and things got much better.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a single frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a heatmap of a frame:

![Frame heatmap][heatmap]

### Here the resulting bounding boxes are drawn onto the frame:
![Vehicles bounding boxes][bboxes_result]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I see the following weak points here:
1. Poor performance. Some quick profiling showed that HOG feature calculation takes significant time. The code isn't optimized, so i see some room for improvement of procesing speed to maybe 2 frames per second instead of 1 frame in 1.5 seconds.
2. The classifier was trained only on car images, so it may have probles with detecting trucks, motorbikes, buses, etc.
3. Traing had images of cars moving the same direction as our car. So, we may obviously have problems detecting vehicles moving towards us. 
4. Training set has images shot in similar lighting conditions and similar environment. So, the classifier may behave poor in different environment (city, for example) or at another daytime (in the evening)
