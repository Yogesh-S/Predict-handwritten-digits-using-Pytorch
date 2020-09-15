# Dataset
MNIST dataset consists of greyscale handwritten digits. Each image is 28x28 pixels

![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/mnist.png?raw=true)

# Training Neural Network
Neural networks with non-linear activations work like universal function approximators. There is some function that maps your input to the output. For example, images of handwritten digits to class probabilities. The power of neural networks is that we can train them to approximate this function, and basically any function given enough data and compute time.

![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/function_approx.png?raw=true)

At first the network is naive, it doesn't know the function mapping the inputs to the outputs. We train the network by showing it examples of real data, then adjusting the network parameters such that it approximates this function.

To find these parameters, we need to know how poorly the network is predicting the real outputs. For this we calculate a loss function (also called the cost), a measure of our prediction error. For example, the mean squared loss is often used in regression and binary classification problems

![](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/loss_formula.JPG?raw=true)
where  𝑛  is the number of training examples,  𝑦𝑖  are the true labels, and  𝑦̂𝑖  are the predicted labels.

By minimizing this loss with respect to the network parameters, we can find configurations where the loss is at a minimum and the network is able to predict the correct labels with high accuracy. We find this minimum using a process called gradient descent. The gradient is the slope of the loss function and points in the direction of fastest change. To get to the minimum in the least amount of time, we then want to follow the gradient (downwards). You can think of this like descending a mountain by following the steepest slope to the base.

![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/gradient_descent.png?raw=true)

# Backpropagation
For single layer networks, gradient descent is straightforward to implement. However, it's more complicated for deeper, multilayer neural networks like the one we've built. Complicated enough that it took about 30 years before researchers figured out how to train multilayer networks.

Training multilayer networks is done through backpropagation which is really just an application of the chain rule from calculus. It's easiest to understand if we convert a two layer network into a graph representation.

# Predict-handwritten-digits-using-Pytorch
