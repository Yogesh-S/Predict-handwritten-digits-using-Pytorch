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

![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/backprop_diagram.png?raw=true)
In the forward pass through the network, our data and operations go from bottom to top here. We pass the input  𝑥  through a linear transformation  𝐿1  with weights  𝑊1  and biases  𝑏1 . The output then goes through the sigmoid operation  𝑆  and another linear transformation  𝐿2 . Finally we calculate the loss  ℓ . We use the loss as a measure of how bad the network's predictions are. The goal then is to adjust the weights and biases to minimize the loss.

To train the weights with gradient descent, we propagate the gradient of the loss backwards through the network. Each operation has some gradient between the inputs and outputs. As we send the gradients backwards, we multiply the incoming gradient with the gradient for the operation. Mathematically, this is really just calculating the gradient of the loss with respect to the weights using the chain rule.
![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/chain_rule.JPG?raw=true)

Note: I'm glossing over a few details here that require some knowledge of vector calculus, but they aren't necessary to understand what's going on.

We update our weights using this gradient with some learning rate  𝛼 .

![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/weights_update.JPG?raw=true)

The learning rate  𝛼  is set such that the weight update steps are small enough that the iterative method settles in a minimum.

# Losses in PyTorch
Let's start by seeing how we calculate the loss with PyTorch. Through the nn module, PyTorch provides losses such as the cross-entropy loss (nn.CrossEntropyLoss). You'll usually see the loss assigned to criterion. As noted in the last part, with a classification problem such as MNIST, we're using the softmax function to predict class probabilities. With a softmax output, you want to use cross-entropy as the loss. To actually calculate the loss, you first define the criterion then pass in the output of your network and the correct labels.

Something really important to note here. Looking at the documentation for nn.CrossEntropyLoss,

This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

The input is expected to contain scores for each class.

This means we need to pass in the raw output of our network into the loss, not the output of the softmax function. This raw output is usually called the logits or scores. We use the logits because softmax gives you probabilities which will often be very close to zero or one but floating-point numbers can't accurately represent values near zero or one. It's usually best to avoid doing calculations with probabilities, typically we use log-probabilities.

It's more convenient to build the model with a log-softmax output using nn.LogSoftmax or F.log_softmax. Then you can get the actual probabilities by taking the exponential torch.exp(output). With a log-softmax output, you want to use the negative log likelihood loss, nn.NLLLoss.

# Neural Network Structure
![alt text](https://github.com/Yogesh-S/Predict-handwritten-digits-using-Pytorch/blob/master/assets/mlp_mnist.png?raw=true)

Network with 784 input units, a hidden layer1 with 128 units and a ReLU activation, then a hidden layer2 with 64 units and a ReLU activation, and finally an output layer with 10 units and a softmax activation as shown above.

# Predict-handwritten-digits-using-Pytorch
Our goal is to build a neural network that can take one of these images and predict the digit in the image.
