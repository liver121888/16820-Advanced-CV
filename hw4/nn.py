import numpy as np
from util import *

# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name=""):

    # Standard deviation
    init_range = np.sqrt(6/(in_size+out_size))
    W = np.random.uniform(-init_range, init_range, (in_size, out_size))
    b = np.zeros(out_size)

    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res


############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """

    # get the layer parameters
    W = params["W" + name]
    b = params["b" + name]

    pre_act = X@W + b

    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        c = -np.max(x_i)
        x_shifted = x_i + c
        res[i] = np.exp(x_shifted)/np.sum(np.exp(x_shifted))
    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = 0, 0
    examples, classes = y.shape

    loss = -np.sum(y*np.log(probs))
    for label, prob in zip(y, probs):
        if np.argmax(label) == np.argmax(prob):
            acc += 1
    acc = acc/examples
    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X

    # Examples, classes
    n, k = delta.shape
    derivative = activation_deriv(post_act)
    # Examples, k
    # print(derivative.shape)

    # [Examples x D]
    # print(X.shape)

    # D*n*n*k = D*k
    grad_W = np.dot(X.T, derivative*delta)

    # n*k*k*D = n*D
    grad_X = np.dot(derivative*delta, W.T)

    # 1*n*n*k = (k,)
    grad_b = np.dot(np.ones((1, n)), derivative*delta).flatten()

    assert grad_W.shape == W.shape
    assert grad_X.shape == X.shape
    assert grad_b.shape == b.shape

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    batches = []
    n, d = x.shape
    # print(n)
    # print(batch_size)
    indices = np.random.randint(0, n, size=(int(n/batch_size), batch_size))
    # print(indices)
    for idx in indices:
        x_batch = x[idx, :]
        y_batch = y[idx, :]
        batches.append((x_batch, y_batch))

    return batches
