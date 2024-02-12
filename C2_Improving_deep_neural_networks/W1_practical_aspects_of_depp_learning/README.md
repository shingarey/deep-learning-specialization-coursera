## About

Discover and experiment with a variety of different initialization methods, apply L2 regularization and dropout to avoid model overfitting, then apply gradient checking to identify errors in a fraud detection model. 

### Learning Objectives

* Give examples of how different types of initializations can lead to different results
* Examine the importance of initialization in complex neural networks
* Explain the difference between train/dev/test sets
* Diagnose the bias and variance issues in your model
* Assess the right time and place for using regularization methods such as dropout or L2 regularization
* Explain Vanishing and Exploding gradients and how to deal with them
* Use gradient checking to verify the accuracy of your backpropagation implementation
* Apply zeros initialization, random initialization, and He initialization
* Apply regularization to a deep learning model* 

## Useful links

[Symmetry Breaking versus Zero Initialization](https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)


## Notes:

### Weights initialization

* The weights 𝑊[𝑙] should be initialized randomly to break symmetry. However, it's okay to initialize the biases 𝑏[𝑙] to zeros. Symmetry is still broken so long as 𝑊[𝑙] is initialized randomly.
* Initializing weights to very large random values doesn't work well.
* Initializing with small random values should do better. The important question is, how small should be these random values be? Let's find out up next!




The main difference between Gaussian variable (numpy.random.randn()) and uniform random variable is the distribution of the generated random numbers:

numpy.random.rand() produces numbers in a uniform distribution and numpy.random.randn() produces numbers in a normal distribution.

When used for weight initialization, randn() helps most the weights to Avoid being close to the extremes, allocating most of them in the center of the range.

An intuitive way to see it is, for example, if you take the sigmoid() activation function.

You’ll remember that the slope near 0 or near 1 is extremely small, so the weights near those extremes will converge much more slowly to the solution, and having most of them near the center will speed the convergence.

### What is L2-regularization actually doing?:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

What you should remember: the implications of L2-regularization on:

    The cost computation:
        A regularization term is added to the cost.
    The backpropagation function:
        There are extra terms in the gradients with respect to weight matrices.
    Weights end up smaller ("weight decay"):
        Weights are pushed to smaller values.

### Dropout

When you shut some neurons down, you actually modify your model. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time. 


    A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
    Deep learning frameworks like TensorFlow, PaddlePaddle, Keras or caffe come with a dropout layer implementation.



What you should remember about dropout:

    Dropout is a regularization technique.
    You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
    Apply dropout both during forward and backward propagation.
    During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

### Gradient Checking

Gradient Checking is slow! Approximating the gradient with ∂𝐽∂𝜃≈𝐽(𝜃+𝜀)−𝐽(𝜃−𝜀)2𝜀

is computationally costly. For this reason, we don't run gradient checking at every iteration during training. Just a few times to check if the gradient is correct.
Gradient Checking, at least as we've presented it, doesn't work with dropout. You would usually run the gradient check algorithm without dropout to make sure your backprop is correct, then add dropout.


What you should remember from this notebook:

* Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
* Gradient checking is slow, so you don't want to run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.


