# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./wupimgs/image_list.png "Visualization"
[image2]: ./wupimgs/auged_list.png "DataAugmentation"
[image8]: ./wupimgs/webimg_list.png "WebTrafficSigns"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is my [project code](https://github.com/chilopod43/CarND-Traffic-Sign-Classifier-Project).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the built-in python function "len" to calculate the sizes of dataset,
and the numpy and pandas library to calculate the shape of the input and output.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The techniques of data augmentation and the reasons why I use them are as follow:

- Random Brightness
    * Because the dataset contains images of various brightness, such as cloudy and fine weathers.
- Random Contrast
    * Because the dataset contians images with a small difference between light and dark.
- Random Zoom
    * Because the dataset contains images of signs with different sizes.
- Grayscale
    * Because color information does not need to be used in the traffic sign recognition.
- Standardization
    * To improve and accelerate the convergence of CNN weight values.

Here is the images after the above data augmentations:

![alt text][image2]

To execute these data augmentations, I used tensorflow's image module.
The following describes the functions used in each and their settings.

1. Random Brightness
    * tf.image.random_brightness(x, delta＝50)
2. Random Contrast
    * tf.image.random_contrast(x, lower=0.7, upper=1.8)
3. Random Zoom
    * tf.image.crop_and_resize(...) scale : 0.8~1.0
4. Grayscale
    * tf.image.rgb_to_grayscale(x)
5. Standardization
    * tf.image.per_image_standardization(...)

There is no change from the original dataset because these data augmentations are applied to the input x of tf.placeholder.
Therefore, in the inference, only Grayscale and Standardization are applied to the input images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrasScale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Flatten				| outputs 800									|
| Fully connected		| outputs 240 									|
| RELU					|												|
| Fully connected		| outputs 168 									|
| RELU					|												|
| Fully connected		| outputs 43 									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following paramters:

| Parameter				|     Value	        							| 
|:---------------------:|:---------------------------------------------:| 
| type of optimizer		| adam											|
| batch size			| 64											|
| number of epochs		| 15											|
| learning rate			| 0.001											|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.946
* test set accuracy of 0.942

I choose a well known architecture Lenet5.

* What architecture was chosen?
     - Lenet5
     - In order to prevent underfitting, the number of output channels of the first Conv2d is quadrupled, 
       the number of second output channels is doubled, and the hidden layers of Fully connected are all doubled.
* Why did you believe it would be relevant to the traffic sign application?
     - Lenet5 is an architecture created for handwriting recognition, and because it is highly robust, 
       I thought that it could be used to identify a large number of signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
     - Working well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]

The first image might be difficult to classify because  it may be confused with other speedlimit signs(30, 60, 70, 80 km/h).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)	| Speed limit (50km/h) 							| 
| Right of way 			| Right of way 									|
| Priority road			| Priority road									|
| Stop 					| Stop 							 				|
| No entry				| No entry  									|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (50km/h) sign,
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.9999998808			| Speed limit (50km/h)							|
| 0.0000000800			| Speed limit (80km/h)							|
| 0.0000000170			| Stop											|
| 0.0000000049			| Roundabout mandatory							|
| 0.0000000012			| Speed limit (20km/h)							|

For the second image, the model is definitely sure that this is a Right-of-way sign,
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0000000000			| Right-of-way at the next intersection			|
| 0.0000000276			| Double curve									|
| 0.0000000003			| Beware of ice/snow							|
| 0.0000000000			| Pedestrians									|
| 0.0000000000			| Roundabout mandatory							|

For the third image, the model is definitely sure that this is a Priority road sign,
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0000000000			| Priority road									|
| 0.0000000001			| Roundabout mandatory							|
| 0.0000000000			| End of all speed and passing limits			|
| 0.0000000000			| Ahead only									|
| 0.0000000000			| No passing									|

For the 4th image, For the third image, the model is definitely sure that this is a Stop sign,
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0000000000			| Stop											|
| 0.0000000001			| Speed limit (30km/h)							|
| 0.0000000000			| Speed limit (70km/h)							|
| 0.0000000000			| Speed limit (120km/h)							|
| 0.0000000000			| Speed limit (20km/h)							|

For the 4th image, For the third image, the model is definitely sure that this is a No entry sign,
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0000000000			| No entry										|
| 0.0000000000			| Speed limit (20km/h)							|
| 0.0000000000			| Turn right ahead								|
| 0.0000000000			| Traffic signals								|
| 0.0000000000			| Beware of ice/snow							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


