
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

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in the convolution network, the gray image can better obtain the characteristics, I use the image RGB three channel data each take 1/3 value, and then add to get grayscale map. This method calls cv2.cvtColor (SRC, cv2.COLOR_RGB2GRAY) more easily to obtain feature information, and gets higher accuracy in the training process

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it can increase the speed of solving the optimal solution by the gradient descent method, and may improve the accuracy

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64    				|
| Flatten               | 5x5x64 flattened to 1600                      |
| Fully connected		| Input is 1600 output is 240, activate_fn: relu |
| Fully connected		| Input is 240 output is 84, activate_fn: relu |
| Output(fully connected)| Input is 84 output is 43, activate_fn: None  |
| Softmax				| The prediction        		                |

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch size is 64, 30 epochs, and the learning rate is 0.001 by the default

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.74%
* validation set accuracy of 96.14%
* test set accuracy of 96.14%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
    
        The first choice is the LetNet-5 network, which is suitable for image processing, high accuracy, and excellent performance.


* What were some problems with the initial architecture?

        The first problem is that the accuracy rate of training and verification is relatively low, which is around 85%. After that, the accuracy of training data is improved, but the verification data is still not able to improve.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

        At the beginning of the data set showed the accuracy of training is not high it is like under fitting problems, the number of nodes gradually convolutional network depth and connection layer, later found the overfitting problem obviously, adding drop out and normalizer after the good results, but accuracy is not high, after many attempts to change the network results alone is not good, try to adjust the data set in the image into the gray before the use of cv2.cvtColor to convert the change to np.sum (X_train/3, axis=3, keepdims=True), so that the accuracy of the results to enhance the degree of Training Officer


* Which parameters were tuned? How were they adjusted and why?

        The main adjustments are: conv1_deep, conv2_deep, fc1_size, fc2_size
        The main purpose is to improve the accuracy, conv1_deep and conv2_deep are to improve the number of convolution network output features and obtain more features.
        Fc1_size, fc2_size is the number of nodes that adjust the full connection network, improve the accuracy of prediction, and prevent over fitting.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    
        Convolution network will highlight the most unique part of the image, and then learn to judge. This algorithm is very suitable for image learning, so learning traffic signs is definitely no problem. The use of dropout is to discarding a part of the feature value to make the network better learning, to a certain extent, to solve the problem of fitting.


If a well known architecture was chosen:
* What architecture was chosen?

        LetNet-5


* Why did you believe it would be relevant to the traffic sign application?

        LetNet-5 many people use it to do the image prediction, and the response is very good, so I believe it can solve the problem.


* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

        It's how big the gap between the accuracy of the pre test and the real result is, and test the other pictures that have not been used in the training.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

    In the project execution does not download pictures from the site, and put the test picture on the website download package, randomly selected 5 pictures of these pictures in accordance with the instructions on the website simply deal with it, and then converted to predict project input image formats, which draw pictures in the HTML file of the project, you need to show all the information are displayed in the project

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| Keep right  									| 
| End of no passing by vehicles over 3.5 metric    			| End of no passing by vehicles over 3.5 metric	|
| Yield					| Yield											|
| Speed limit (30km/h)  | Speed limit (30km/h)                          |
| Priority road		    | Priority road    					     		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the fourth image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Speed limit (30km/h)							| 
| .15     				| Speed limit (20km/h)							|
| .10					| Stop											|
| .04	      			| Speed limit (70km/h)			 				|
| .04				    | End of all speed and passing limits    		|


Other image, almost is 100%

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
