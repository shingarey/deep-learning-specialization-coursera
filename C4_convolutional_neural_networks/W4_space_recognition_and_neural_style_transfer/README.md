# Foundations of Convolutional Neural Networks

This repository contains the code and materials for the "Foundations of Convolutional Neural Networks" course in the Deep Learning Specialization on Coursera.

## About

Explore how CNNs can be applied to multiple fields, including art generation and face recognition, then implement your own algorithm to generate art and recognize faces!

## Learning Objectives

- Differentiate between face recognition and face verification
- Implement one-shot learning to solve a face recognition problem
- Apply the triplet loss function to learn a network's parameters in the context of face recognition
- Explain how to pose face recognition as a binary classification problem
- Map face images into 128-dimensional encodings using a pretrained model
- Perform face verification and face recognition with these encodings
- Implement the Neural Style Transfer algorithm
- Generate novel artistic images using Neural Style Transfer
- Define the style cost function for Neural Style Transfer
- Define the content cost function for Neural Style Transfer

## Notes

### FaceNet

- Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem.
- Triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
- The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.

#### References

[Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

Further inspiration was found here: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

And here: https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb

### Style Transfer

- The content cost takes a hidden layer activation of the neural network, and measures how different 𝑎(𝐶) and 𝑎(𝐺) are.
- When you minimize the content cost later, this will help make sure 𝐺 has similar content as 𝐶. 

- The style of an image can be represented using the Gram matrix of a hidden layer's activations.
- You get even better results by combining this representation from multiple different layers.
- This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
- Minimizing the style cost will cause the image 𝐺 to follow the style of the image 𝑆. 
- The total cost is a linear combination of the content cost 𝐽𝑐𝑜𝑛𝑡𝑒𝑛𝑡(𝐶,𝐺) and the style cost 𝐽𝑠𝑡𝑦𝑙𝑒(𝑆,𝐺).
- 𝛼 and 𝛽 are hyperparameters that control the relative weighting between content and style.


- Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
- It uses representations (hidden layer activations) based on a pretrained ConvNet.
- The content cost function is computed using one hidden layer's activations.
- The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
- Optimizing the total cost function results in synthesizing new images.


#### References

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)


Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style
Harish Narayanan, Convolutional neural networks for artistic style transfer.
Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style".
Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition MatConvNet.
