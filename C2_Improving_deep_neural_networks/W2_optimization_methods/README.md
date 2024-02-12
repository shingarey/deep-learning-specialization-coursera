## About

Develop your deep learning toolbox by adding more advanced optimizations, random minibatching, and learning rate decay scheduling to speed up your models. 

### Learning Objectives

* Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
* Use random minibatches to accelerate convergence and improve optimization
* Describe the benefits of learning rate decay and apply it to your optimization

## Useful links

[Adam paper](https://arxiv.org/pdf/1412.6980.pdf)

## Notes:

### Mini-Batch Gradient Descent

What you should remember:

* Shuffling and Partitioning are the two steps required to build mini-batches
* Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

### Momentum

Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations. 

Momentum takes into account the past gradients to smooth out the update. The 'direction' of the previous gradients is stored in the variable 𝑣. Formally, this will be the exponentially weighted average of the gradient on previous steps. You can also think of 𝑣 as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to the direction of the gradient/slope of the hill. 

How do you choose 𝛽?

The larger the momentum 𝛽 is, the smoother the update, because it takes the past gradients into account more. But if 𝛽 is too big, it could also smooth out the updates too much. Common values for 𝛽 range from 0.8 to 0.999. If you don't feel inclined to tune this, 𝛽=0.9 is often a reasonable default.
Tuning the optimal 𝛽 for your model might require trying several values to see what works best in terms of reducing the value of the cost function 𝐽. 

What you should remember:

* Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
* You have to tune a momentum hyperparameter 𝛽 and a learning rate 𝛼.

### Adam

Adam is one of the most effective optimization algorithms for training neural networks. It combines ideas from RMSProp (described in lecture) and Momentum.

How does Adam work?

1. It calculates an exponentially weighted average of past gradients, and stores it in variables 𝑣 (before bias correction) and 𝑣𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑 (with bias correction).
2. It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables 𝑠 (before bias correction) and 𝑠𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑒𝑑 (with bias correction).
3. It updates parameters in a direction based on combining information from "1" and "2".


Momentum usually helps, but given the small learning rate and the simplistic dataset, its impact is almost negligible.

On the other hand, Adam clearly outperforms mini-batch gradient descent and Momentum. If you run the model for more epochs on this simple dataset, all three methods will lead to very good results. However, you've seen that Adam converges a lot faster.

Some advantages of Adam include:

* Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
* Usually works well even with little tuning of hyperparameters (except 𝛼)

### Learning Rate Decay and Scheduling

Lastly, the learning rate is another hyperparameter that can help you speed up learning.

During the first part of training, your model can get away with taking large steps, but over time, using a fixed value for the learning rate alpha can cause your model to get stuck in a wide oscillation that never quite converges. But if you were to slowly reduce your learning rate alpha over time, you could then take smaller, slower steps that bring you closer to the minimum. This is the idea behind learning rate decay.

Learning rate decay can be achieved by using either adaptive methods or pre-defined learning rate schedules. 

Notice that if you set the decay to occur at every iteration, the learning rate goes to zero too quickly - even if you start with a higher learning rate. 

When you're training for a few epoch this doesn't cause a lot of troubles, but when the number of epochs is large the optimization algorithm will stop updating. One common fix to this issue is to decay the learning rate every few steps. This is called fixed interval scheduling.

You can help prevent the learning rate speeding to zero too quickly by scheduling the exponential learning rate decay at a fixed time interval, for example 1000. You can either number the intervals, or divide the epoch by the time interval, which is the size of window with the constant learning rate. 



