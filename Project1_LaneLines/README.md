# **Udacity self-driving car Nanodegree** 

## Finding Lane Lines on the road: Project 1

### This is descrtption of a pipeline, used to detect lane lines on images and videos

---

[//]: # (Image References)

[image1]: ./examples/regular_image.jpg "Regular image"
[image2]: ./examples/normalized_image.jpg "Normalized image"
[image3]: ./examples/yellow_lane_image.jpg "Image with yellow lane"
[image4]: ./examples/yellow_masked.jpg "Enchanced image with yellow lane"
[image5]: ./examples/edges_simple.jpg "Image with clear lane lines"
[image6]: ./examples/edges_tricky.jpg "Tricky to detect lane lines"

---


### 1. The pipeline
#### Image histogram normalization
The first step i used in my pipeline was image histogram normaliztion. This means that i made darkest pixels of an image to have brightness of 255, and darkest to 0. This is nonlinear process in RGB color space, so i used the following steps:

1. Convert image to HSV color space 
2. Extract brightness channel 
3. Normalize it with OpenCV function **equalizeHist** 
4. Merge channels back together 

This made lane lines even brighter than other parts of an image

![Image from camera][image1] ![Normalized image][image2]

#### Yellow color detection
Yellow lines themselves are not as bright as white lines, so i needed them to stand out on an image. This was archieved the following way:

1. Convert image to HSV, so than colors are in a separate channel now
2. Set color boundaries for yellow
3. Creare a mask for yellow color using **inRange** function
4. Blend the mask with image from the first step, so that yellow lanes if they exist on an image are brightened

![Image with yellow lane][image3] ![Yellow lane highlighted][image4]

#### Convert image to grayscale

This was quite simple, using helper function **grayscale**

#### Applying gaussian blur to an image

Although gaussian smoothing is a part of Canny edge detection algorithm, i foung out that additional smoothing helps to filter out more noise and leads to better results

#### Applying Canny edge detection algorithm

I decided not to use recommended 1:2 or 1:3 ratio for low/high thresholds in Canny detector. Because i did histogram normalization earlier, lane lines on my images corresponded to the brightest pixels. So, the lower threshold was 240, and higher was 250. This worked fine.

![Canny simple][image5] ![Canny challanging][image6]

#### Apply masking to region of interest

#### Apply hough transormation

Hough transformation was applied to find lines on masked Canny detector output. I uses transformation parametrs nearly the same as in lectures

#### Filter the lines detected by Hough transformation

Not every detected line corresponded to lane line. I
1. First I filtered invalid lines by calculating their slope and rejecting lines with too high or too low slope. 
2. Second I split the lines to left and right by their slope 
3. After that calculated mean and standard deviation for each lane, and then filtered out parameters that were 1 standard deviation away from the mean as outliers. (I know that usually it is 3 std. deviations away from mean to be considered as outlier, but this was not the case)
4. Averaged slope and intercept for each lane line and draw it on an image

### 2. Potential shortcommings

1. This wont't work at winter when roads are covered with snow
2. There may be problems at night time. Can't say for sure for now. Haven't tested it yet
3. There is visible lines jitter on resulting videos
4. Not every road has lane lines, or they are in a good condition (especially in Russia :))

### 3. Possible improvements

1. Preprocess the images better, so that lines are even more visible, for example increase image contrast or blend image with itself using, for example Multiply blending mode.
2. Use advanced filtering techniques to find true values of slope and intercept for lane lines. Someone in Slack advised to use RANSAC algorithm, but i suppose this is a bit overkill. Or maybe to use somthing like k-means clustering. Or simply use weighted average for finding mean values of slope and intercept, using line lenghts as weights.
3. Using information from previous frames. Lanes direction is not something that is changing rapidly, so if our lines parameters distribution has, for example high deviation, we may use info from previous frames to adjust it. 