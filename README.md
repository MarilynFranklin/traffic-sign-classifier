# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a stacked bar chart showing the number of traffic sign classes for each data set:

![](reflection_images/stacked-bar-chart.png?raw=true)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun mentioned in [their paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) that grayscale out performed images with 3 color channels. Next, I normalized by subtracting the mean and dividing by the standard deviation.

```
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def normalize(x):
    x = x.astype('float32')
    mean = np.mean(x)
    std = np.std(x)
    x -= mean
    x /= std
    return x

def rgb2gray(data):
    imgs = np.ndarray((data.shape[0], 32, 32, 1), dtype=np.uint8)
    for i, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        imgs[i] = img
    return imgs

X_train = normalize(rgb2gray(X_train))
X_valid = normalize(rgb2gray(X_valid))
X_test = normalize(rgb2gray(X_test))
```

Here is an example of an original image and an augmented image:
![](reflection_images/original.jpg?raw=true)
![](reflection_images/normalized.jpg?raw=true)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| outputs 120        									|
| Dropout		| .5        									|
| RELU					|												|
| Fully connected		| outputs 10        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a learning rate of 0.001 with a batch size of 256 and
50 epochs. I also used a keep probability of .5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of **96.2%**
* test set accuracy of **94.1%**

I chose LeNet-5 as a starting point for my model because it performs well for image classification with minimal preprocessing.

During training, the validation accuracy stopped increasing at around 96%. This could probably be improved by lowering the learning rate and letting the model train longer. Since the test accuracy is a bit lower at 94%, I could also try adding a few more dropout layers to see if it would prevent overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![](german_traffic_signs/traffic_sign_1.jpg?raw=true)
![](german_traffic_signs/traffic_sign_2.jpg?raw=true)
![](german_traffic_signs/traffic_sign_3.jpg?raw=true)
![](german_traffic_signs/traffic_sign_4.jpg?raw=true)
![](german_traffic_signs/traffic_sign_5.jpg?raw=true)

Image 4 is the only one that has dim lighting since it seems that it was taken
at dusk. The other images each appear to be relatively easy to classify since
they are clear.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Road work     			| Road work 										|
| General caution					| General caution											|
| Bumpy road	      		| Bumpy Road					 				|
| Wild animals crossing			| Wild animals crossing      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1% and is most likely due to the fact that these were all clear, well lit images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The model was very sure for each image prediction. The first three output a probability of "1.00000000e+00" so I'm not sure of the exact value for these, but it is slightly smaller than 1. For the last two images, the top probabilities were both around .99.

Here's the top 5 probabilities for each image in detail:

**Image 1**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1         			| Right-of-way at the next intersection   									| 
| 6.87617796e-10     				| Beware of ice/snow 										|
| 5.51837187e-10					| Slippery road											|
| 8.23607457e-16	      			| End of no passing by vehicles over 3.5 metric tons |
| 8.20775191e-16				    | General caution      							|

**Image 2**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1         			| Road work   									| 
| 1.83471970e-11     				| Beware of ice/snow 										|
| 6.41799114e-13					| Bumpy road											|
| 5.62910515e-16	      			| Dangerous curve to the right |
| 5.76828610e-18				    | Slippery road      							|

**Image 3**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1         			| General caution   									| 
| 3.76839964e-08     				| Traffic signals 										|
| 1.67265544e-14					| Pedestrians											|
| 1.83744562e-21	      			| Road work |
| 7.20547949e-23				    | No passing      							|

**Image 4**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~.99         			| Bumpy road   									| 
| 4.06247773e-06     				| Bicycles crossing 										|
| 1.44250209e-10					| Children crossing											|
| 6.34199013e-11	      			| Traffic signals |
| 1.22539316e-12				    | Road work      							|

**Image 5**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~.99         			| Wild animals crossing   									| 
| 3.49201000e-05     				| Double curve 										|
| 1.50429919e-10					| Road work											|
| 2.16851190e-12	      			| Slippery road |
| 1.22145599e-12				    | Dangerous curve to the left      							|
