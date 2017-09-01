**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./examples/center.jpg "Center image"
[left]: ./examples/left.jpg "Left image"
[right]: ./examples/right.jpg "Right image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

Attached Jupyter Notebook contains code for reading and processing input data, and for model training and validation. Input data format assumes that training images are located in `IMG` folder in the root of the project. Files with driving logs are also located in the root folder and are named the following way: `driving_log*.csv`

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My solution is inspired by [this](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) article. It contains 4 convolutional layers and 5 fully connected layers: 

Layer | Description
------|------------
Normalization| Normalize data to zero mean and (-0.5, 0.5) range
Cropping| Crop meaningless data from top and bottom
Convolution| 24 5x5 filters with 2x2 strides and RELU activation
Convolution| 36 5x5 filters with 2x2 strides and RELU activation
Convolution| 48 5x5 filters with 2x2 strides and RELU activation
Convolution| 64 3x3 filters with 1x1 strides and RELU activation
Flatten| Flatten convolutions result
Dropout| 0.5 dropout probability
Fully connected| Fully connected layer with 1164 nodes
Dropout| 0.5 dropout probability
Fully connected| Fully connected layer with 100 nodes
Fully connected| Fully connected layer with 50 nodes
Fully connected| Fully connected layer with 10 nodes
Output| Output layer with 1 node and linear activation

#### 2. Attempts to reduce overfitting in the model

Dropout layers after Flatten layer and first Fully connected layer are used to prevent model from overfitting. Dropout probability is set to 0.5 in both cases.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I collected my own training data using PS4 gamepad. This allowed to produce consistent steering angles. I used the following tricks when collecting my data:

  1. Driving counter-clockwise.
  2. Flipping images horizontally
  3. Using images from left and right camera and adding offset to corresponding steering angles.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started from simple network with 1 fully-connected layer and training data provided for this project. This prooved that the model is training and the general setup is correct. Then i tried adding more convolutional and fully connected layers. The model was behaving better, but still lacked generalizing capabilities. 

Next i tried to obtain more data, as described above. This helped to create model that was driving the whole track, but at low speed and bouncing from one side of road to another. 

#### 2. Final Model Architecture

The final architecture is based on Nvidia's network for self-driving cars, as described above, but does not repeat it exactly.

#### 3. Creation of the Training Set & Training Process

Data collection process is described above. Here are some sample images from training set

![Center of lane image][center]

![Left camera image][left]

![Right camera image][right]


After the collection process, I had approx 60000 data points. Data, that was fed to the model was converted to YUV color space, according to Nvidia recomendation and resized to 240x120 pixel dimensions.


I finally randomly shuffled the data set and put 20 % of the data into a validation set. 

I used this training data for training the model. Validation and training loss became close after traing 5 epochs and were not changing. This could mean that the model converged.