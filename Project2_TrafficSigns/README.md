# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[snow]: ./test_web_images/snow.jpg "Beware of ice\snow"
[pedestrian]: ./test_web_images/pedestrian.jpg "Pedestrians"
[yield]: ./test_web_images/yield.jpg "Yield"
[60limit]: ./test_web_images/60_limit2.jpg "60 km\h speed limit"
[70limit]: ./test_web_images/70_limit.jpg "70 km\h speed limit"
[test_distribution]: ./examples/test_distr.png "Test distribution"
[train_distribution]: ./examples/train_distr.png "Train distribution"
[dataset_classes]: ./examples/image_variety.png "Image variety"
[before]: ./examples/before.png "Before processing"
[after]: ./examples/after.png "After processing"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vanjacksing/SelfDriving/blob/master/Project2_TrafficSigns/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. [German traffic sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) was used for this project.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32x32 pixels with 3 color channels (RGB)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

This dataset has several features:
1. Classes are skewed: most frequent class in training dataset has 3000 objects (50 km/h speed limit) and least frequent has 270 objects (20 km/h speed limit) - 10 times difference!
2. Most frequent signtype is "Speed limit" - train dataset has 16950 objects of this type
3. Though it is skewed, the distribution of classes in training and test dataset looks pretty same:

![Train dataset classes distribution][train_distribution]
![Test dataset classes distribution][test_distribution]

### Design and Test a Model Architecture

#### 1. Image quality
In general image quality varies greatly:

![Image variety][dataset_classes]

I decided not to overcomplicate the things ans used Pillow library for processing. It has some useful features, that helped me.
My pipeline looked the following way:

1. Increace image contrast. That was done with *ImageOps.autocontrast* function from Pillow. This helped greatly because image contrast varied greatly across the dataset.
2. Increase image sharpness with *ImageEnhance.Sharpness*
3. Convert to grayscale with *ImageOps.grayscale*

Images berore processing look like this:

![Before processing][before]

Images after processing look like this:

![After processing][after]



#### 2. Architecture considerations
One of the main considerations i used in this project were multi-scale features. I tool this idea from [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, output depth is 32 	|
| Leaky RELU					|	alpha = 0.1											|
| Max pooling	      	| 2x2 stride with 2x2 kernel 				|
| Convolution 3x3	    | 1x1 stride, valid padding, output depth is 64      									|
| Leaky RELU					|	alpha = 0.1											|
| Max pooling	      	| 2x2 stride with 2x2 kernel 				|
| Convolution 3x3	    | 1x1 stride, valid padding, output depth is 128      									|
| Leaky RELU					|	alpha = 0.1											|
| Max pooling	      	| 2x2 stride with 2x2 kernel 				|
| Flatten		| Flatten and concatenate output of 3 convolutional layers        									|
| Dropout | Dropout layer with keep probability 0.5 to prevent overfitting |
| Fully conected | Fully connected layer with 1024 nodes |
| Leaky RELU					|	alpha = 0.1											|
| Dropout | Dropout layer with keep probability 0.5 to prevent overfitting |
| Fully conected | Fully connected layer with 256 nodes |
| Leaky RELU					|	alpha = 0.1											|
| Dropout | Dropout layer with keep probability 0.5 to prevent overfitting |
| Output | Fully connected layer with 43 nodes and linear activation function |
 
#### 3. Training

AdamOptimizer with learning rate 0.0003 and 30 epochs with batch size = 128 was used to train the model. Cross entropy was used as a target variable for optimizer. 

#### 4. Game-changing solutions

In my opinion one of the most helpful techniques were:

1. Using multi-scale features as input to fully-connected layer
2. Image enhancement with autocontrast
3. Conversion to grayscale

#### 5. Final results 

* training set accuracy of 1.0
* validation set accuracy of 0.976 
* test set accuracy of 0.96

The first architecture used was Lenet5 from [here](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb). It allowed to reach nearly 88% accuracy. That was not enough. It was originally designed for MNIST problem, which is less complicated. It had less convolution layers, less features, less hidden nodes at fully-connected layers, used only last convolutional layer output and thus was less effective.
To improve it i added one more convolutional layer, incresed all conv layers depths, used outputs of all convolutions as input to fully-connected, added dropout layers and increased number of hidden nodes, so that the model has more learning capability.

I suppose ideal training accuracy was result of an overfitting. This may be reduced by lowering dropout probability and decreasing learning rate. But those steps may also lead to bad model convergence and stucking at local minimum. So, current solution looks fine at the moment.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
---
![Beware of ice/snow][snow] 
---
![Pedestrians][pedestrian] 
---
![Yield][yield] 
---
![Speed limit 60 km/h][60limit] 
---
![Speed limit 70 km/h][70limit]
---

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Beware of ice\snow      		| Slippery road   									| 
| Pedestrians     			| General caution 										|
| Yield					| Yield											|
| 70 km/h	      		| 70 km/h					 				|
| 60 km/h			| 60 km/h      							|

The first two images were misclassified. Probably this happened due to the fact, that there were few of them in training set (600 for the first and 300 for the second). Besides that Classes that were predicted have very much in common with actual classes ("Slippery road" was predicted for "Beware of ice\snow" and "General caution" for "Pedestrians"). See predicted classes probabilities below.

In general accuracy on set of 9 images from web was 78% (7 of 9 correct). This may be explained: images from the web were not very representative and contained images classes, which classification accuracy was below average. More general image classed were predicted correctly.

#### 2. Let's see top 5 probabilities for signes from the web:

First image has class "Beware of ice/snow"

![Snow][snow]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .666         			| Slippery road   									| 
| .294     				| Beware of ice/snow 										|
| .019					| Clildren crossing											|
| .019	      			| Road narrows on the right					 				|
| .0004				    | Right-of-way at the next intersection      							|

Second image has class "Pedestrians"

![Pedestrians][pedestrian]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| General caution   									| 
| > .001     				| Pedestrians 										|
| > .001					| Road narrows on the right											|
| > .001	      			| Double curve					 				|
| > .001				    | Traffic signals      							|

Third image has class "Yield"

![Yield][yield]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yeild   									| 

Other probabilities were as low as 1e-27

Fourth image has class "Speed limit 60 km/h"

![60 km/h speed limit][60limit]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Speed limit (60km/h)   									| 
| .000008     				| Speed limit (80km/h) 										|

Other probabilities were below 1e-6


Fifth image has class "Speed limit 70 km/h"

![Speep limit 70 km/h][70limit]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (70km/h)   									| 

Other probabilities were lower than 1e-10

Generally, it is evident that classifier precisely pridicts sign type (because of its shape) and messes up with signs that are infrequent in train dataset and look quite similar. Proper image augmentation should definetely help.
