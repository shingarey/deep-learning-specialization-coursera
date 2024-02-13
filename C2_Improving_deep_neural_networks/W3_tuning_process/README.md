## About

Explore TensorFlow, a deep learning framework that allows you to build neural networks quickly and easily, then train a neural network on a TensorFlow dataset.  

### Learning Objectives

- Master the process of hyperparameter tuning
- Describe softmax classification for multiple classes
- Apply batch normalization to make your neural network more robust
- Build a neural network in TensorFlow and train it on a TensorFlow dataset
- Describe the purpose and operation of GradientTape
- Use tf.Variable to modify the state of a variable
- Apply TensorFlow decorators to speed up code
- Explain the difference between a variable and a constant

## Useful links



## Notes

### Using One Hot Encodings

Many times in deep learning you will have a 𝑌 vector with numbers ranging from 0 to 𝐶−1, where 𝐶 is the number of classes. If 𝐶 is for example 4, then you might have the following y vector which you will need to convert like this:

y = [1 2 3 0 2 1] is often converted to matrix:

0   0   0   1   0   0   -> class 0
1   0   0   0   0   1   -> class 1
0   1   0   0   1   0   -> class 2
0   0   1   0   0   0   -> class 3

This is called "one hot" encoding, because in the converted representation, exactly one element of each column is "hot" (meaning set to 1). To do this conversion in numpy, you might have to write a few lines of code. In TensorFlow, you can use one line of code:

`tf.one_hot(labels, depth, axis=0)`

axis=0 indicates the new axis is created at dimension 0

In this assignment, you were introducted to tf.GradientTape, which records operations for differentation. Here are a couple of resources for diving deeper into what it does and why:

Introduction to Gradients and Automatic Differentiation: https://www.tensorflow.org/guide/autodiff

GradientTape documentation: https://www.tensorflow.org/api_docs/python/tf/GradientTape




