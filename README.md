# Mango-Shelf-Life-Prediction
## Introduction
The prediction of shelf life in the food industry is a crucial aspect that can help avoid food waste and maintain food quality. The shelf life of a batch of food products is the duration that a product is acceptable for consumption or sale before it deteriorates and becomes unfit for consumption. In this project, the task is to predict the shelf life of a batch of mango using deep learning. The shelf life prediction system requires the use of an image dataset of the mango batch. The objective is to train a deep learning model using the dataset to predict the shelf life of a batch of mango. This report describes the methodology used to develop the deep learning system for the shelf life prediction of a batch of mango. The images of the mango batch were taken by IoT device Raspberry Pi High Quality Camera which is connected to the IoT controller  Raspberry pi. This IoT device had taken 218 images of this mango batch during a course of 30 days. Based on this data our aim is to predict the shelf life of this mango batch. 

## Import Packages
The required packages are imported and deployed initially. The code is constructed with Python 3.9. 

pandas 
numpy 
matplotlib
cv2
os
sklearn.model_selection.train_test_split
keras.models.Sequential
keras.layers.Dense
from sklearn.metrics.confusion_matrix

## Dataset Construction 
To make any predictions we need to train a model and to do that we need a well constructed dataset which is in the machine readable form, basically a long data format. So in order to do so I first had to perform image processing and since 312 images is a very small quantity to train a model for image classification task, I divided each batch of mango image to 9 individual images hence expanding our dataset to a total of 2,808 images. To do this I used os library and openCV. After cropping the images I saved the cropped images to a folder in the working directory. After this the task was to label these images. I chose manual labeling since the best way to check if the model is performing accurately on livestock commodities is to judge it based on a human judgment criteria. For this I created three folders in the working directory and named it “Grade1”, “Grade2” and “Grade3”. Then I moved the images to these folders manually based on the quality of the mangoes. The definitions of each class is as follows,
Grade 1: The mangoes that belong to this class are fresh and can be sold at the market price.
Grade 2: These mangoes are a little too ripe and cannot be sold at the market price, instead they are sold at a discounted price.
Grade 3: Mangoes that belong to this category are rotten and are not fit to sell.

Now we have our data labeled, next I needed to enhance this dataset for which I required features that explain the images, more importantly the key features that are highly correlated with the degradation process, e.g., what feature of an image might change as the mango starts to ripe? The R,G,and B components, as a mango ripens the green component reduces and the red component increases, also among other features I chose Hue, Intensity and Saturation as another set of features. Apart from these I also chose Entropy and Percent of Blemished Skin as my feature in the feature vector. To calculate these features from an image I once again used the openCV library and its functions. 

The R,G and B components were directly extracted from the images using cv2.split function.
 
To calculate HSI features of the images, the function converts the image from the BGR color space to the RGB color space and then to the HSI color space. It calculates the mean values of the hue, saturation, and intensity channels in the HSI image.

To calculate the Entropy of the images, the function calculates the image histogram and normalizes it to have values between 0 and 1. Then, the function computes the entropy of the image using the formula 
-sum(p_i * log2(p_i)),  
where p_i is the probability of the pixel value i occurring in the image. The value 1e-7 is added to the probability before taking the logarithm to avoid taking the logarithm of zero, which would result in undefined behavior. Finally, the negative sum is returned as the entropy value. The entropy is a measure of the amount of information present in the image, where higher values indicate greater uncertainty or complexity in the image.

Finally I calculate the percent of blemished skin. To do this I first extract the red channel from the image and apply Gaussian blur with a kernel size of 7. Then, I binarize the red channel using a mean threshold. I also extract the saturation channel from the image in the HSV color space, and use it as a metric for detecting light spots. To do this, I compute the mean saturation value of the image and set a threshold of 0.7 times this value to identify light spots.

Next, I combine the binary images of dark and light spots using a bitwise OR operation and process the resulting image. I create a binary mask to exclude the edges of the image and dilate the resulting image using an elliptical kernel. I then use the findContours function of OpenCV to detect contours in the resulting image.

To filter out small contours and segments, I iterate through the detected contours and compute their area and roundness. If a contour has an area less than 40 pixels or a roundness less than 0.1, I discard it. Additionally, I fit an ellipse to the contour and compute the distance between its center and a fixed point in the image. If this distance is less than 85 pixels or the ratio of the contour area to the total mask area is less than 0.03, I remove the contour.

Finally, I compute the percent of blemished area in the image by dividing the total number of blemished pixels by the total number of pixels in the mask and multiplying by 100. This value is appended to a list for further analysis.

Now our feature vector is ready. I extracted these features from all the images present in our dataset and merged it with the label dataset. This completes our dataset creation.

## Training our Deep Learning Model 
To predict the class among the three grades that I created earlier we need to train a deep learning model. Based on our limited dataset, I have chosen an Artificial Neural Network to make our predictions for us. To do this I split our dataset into training and testing using the train_test_split function from the sklarn.model_selection module. I have chosen the test_size to be 0.2 that means it is an 80/20 split into training and testing dataset. 

I then created a neural network using the Keras Sequential model and added three Dense layers with 16, 8, and 3 units, respectively, using the ReLU and softmax activation functions. The model was compiled with categorical_crossentropy loss, adam optimizer, and accuracy metrics. The model was then trained on the training set using pd.get_dummies() to convert the target variable into one-hot encoded vectors and a batch size of 10 for 81 epochs. I arrived at 81 epochs after studying the loss function vs. the number of epoch’s graph. This completed the training of our model.

## Testing our ANN Model (Results) 
Next I tested this model on the testing dataset and on some other images of mangoes. On the testing dataset the accuracy came out to be 93.94%. Which was a really good number considering the limitations of our dataset. I also used the confusion_matrix to study the model’s performance. Further I plotted the number of mangoes in each grade changing with time to study and calculate the shelf life of the batch. We Define Shelf Life of a batch of mango in terms of its economic value in the market. Empirically Speaking, It is defined as the point in time where the number of Grade3 mangoes (The ones which cannot be sold) exceed the number of Grade2 mangoes (The ones which can be sold at some discounted price at least), Since the batch is then believed to have lost the market value. Thus the transport has to take place before this time period, which in our case lies just before the end of 3rd week, i.e, somewhere around 18-20 days as can be seen from the graph below,

![Untitled (1)](https://github.com/Vinayak-Pandey-2001/Mango-Shelf-Life-Prediction/assets/108610717/c8765b25-bade-4a9e-bb41-1235dcf21e7f)
![Untitled](https://github.com/Vinayak-Pandey-2001/Mango-Shelf-Life-Prediction/assets/108610717/09a2dfba-ca48-4aeb-8af1-f561a682fb75)


## Conclusion
Though we have achieved a good accuracy in our predictions there is a lot of scope for improvement. The dataset of images could be of multiple batches, instead of cropping the images one can use image annotation models like YOLOv8 to make the images consistent. Labeling of the acquired grades can be done against strictly known physical parameters like BRIX, and pH instead of manually labeling them.
