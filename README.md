# FaceRecognitionWithCNN
Implementation of a face recognition system in python using convolution neural network

The dataset used in this project is a collection of images of seven different individuals: Jumana, 
Farida, Maher, Khaled, Kroush , Ammar , Joana. The dataset was created with the team’s 
members and their friends. The images were then manually labeled with the corresponding 
individual's name.
The dataset contains a total of 23 images, with each individual having approximately 3-4 images. 
The images have varying resolutions, lighting conditions, and different angles making the task of 
face recognition challenging

We first preprocess the dataset by detecting faces using a Haar 
Cascade Classifier and resizing the detected face region to a target size of 388x388 pixels. We
then save the images to a new folder directory with their labels. Then we use tensor flow to 
load the images from the dataset into two variables train and validation.

A CNN model is defined using TensorFlow's Sequential class. It consists of four convolutional 
layers (hidden layers with activation function relu), followed by two dense layers. The output of 
the final dense layer is fed into a softmax activation function to produce the classification 
output. The model is compiled using the compile method with adam optimizer, 
sparse_categorical_crossentropy as the loss function, and accuracy as the evaluation metric. 
The model is then trained using the training and validation data with having seven epochs.
