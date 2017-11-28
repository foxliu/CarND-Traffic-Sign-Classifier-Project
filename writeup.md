
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
    
        刚开始选择的是LetNet-5的网络，这个网络在图像处理上很适用，准确率高，性能优异


* What were some problems with the initial architecture?

        首先出现的问题是训练和验证的准确率都比较低，大概在85%左右，之后就是训练数据的准确率提升了但是验证数据的依然无法提高


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

        刚开始的时候数据集训练表现出现的准确率不高很像是欠似合的问题，逐步卷积网络的深度和全连接层的节点数，后来发现明显的过拟合问题，添加了drop out和 normalizer后结果好了些，但是准确度依然不高，经过许多次的尝试，单纯的更改网络结果并不好，试着调整数据集，在图像转化为灰度这方面把之前的使用 cv2.cvtColor来转换 更改为 np.sum(X_train/3, axis=3, keepdims=True)， 这样使训练结果的准确度大副度提升


* Which parameters were tuned? How were they adjusted and why?

        主要调整过的参有： conv1_deep, conv2_deep, fc1_size, fc2_size
        目的主要是为了提高准确度， conv1_deep 和 conv2_deep 是为了提高卷积网络输出的特征数，获取更多的特征，
        fc1_size, fc2_size 是调整全连接网络的节点数，提高预测的准确度，并防止过拟合


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    
        卷积网络会把图像中最独特的部分凸显出来，然后来进行学习判断，这样的算法很适合于图像学习，所以学习交通标志是肯定没有问题的。 使用 dropout 是丢弃掉一部分特征值来使网络更好的学习，一定程度上解决掉了过拟合的问题


If a well known architecture was chosen:
* What architecture was chosen?

        LetNet-5


* Why did you believe it would be relevant to the traffic sign application?

        LetNet-5很多人都在用它来做图像预测，反响结果也很好，所以我相信它能解决问题


* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

        就是看它预测试的准确度与真实的结果的差距有多大，并在其他的未在训练中使用过的图片进行测试 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

在项目中执行中没有从网站上下载图片，而且把网站上的测试图片包下载下来，随机抽取5张图片，对这些图片依照网站上的说明简单处理一下，然后转换为项目输入的图片格式进行预测，所抽取的图片在项目的html文件中，所有的需要展示的信息一并在项目中展示出来

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
