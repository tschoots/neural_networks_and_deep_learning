import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
#from testCases_v2 import *
#from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (15.0, 14.0)  # set the default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer

    Returns
    :return:
    :param parameters : python dictionary containing your parameters:
                            W1 -- weight matrix of shape (n_h, n_x)
                            b1 -- bias vector of shape (n_h, 1)
                            W2 -- weight matrix of shape (n_y, n_h)
                            b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))


    assert(W1.shape == (n_h, n_x))*0.01
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))*0.01
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    :param layer_dims: python array (list) containing the dimensions of each layer in your network

    Returns
    :return: parameters : python dictionary containing your parameters "W1", "b1", .... , "WL", "bL":
                             Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1]
                             bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # because we are using RelU beter to user the following initialization
        # keeping close around 1 to prevent vanisching / exploding gradients
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1]) #*0.01
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layers's forward propagation

    :param A: activations from previous layer (or input data): (size_of_previous_layer, number of examples)
    :param W: weights matrix : numpy array of shape (size_of_current_layer, size_of _previous_layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)

    Returns
    :return:
        Z -- the input of the activation function , also caller pre-activation parameter
        cache == a python dictionary containing "A", "W" and "b"  stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations from previous layer (or input data): (size_of_previous_layer, number_of_examples)
    :param W: weights matrix: numpy array of shape(size_of_current_layer, size_of_previous_layer)
    :param b: bias vector : numpy array of shape (size_of_current_layer, 1)
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns
    :return:
    :param A: the output of the activation function, also called the post-activation value
    :param cache: a python dictionary containing "linear_cache" and "activation_cache"
                   stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    if activation == "relu":
        A, activation_cache = relu(Z)


    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELY]*(L-1)->LINEAR->SIGMOID computation

    Arguments
    :param X: data, numpy array of shape (input size, number of examples)
    :param parameters: output of initialize_parameters_deep()

    Returns
    :return:
    :param AL : last post-activation value
    :param caches : list of caches containing:
                        every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                        the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2   # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1. Add "cache" to the "caches" list
    for l in range(1,L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7)

    Arguments:
    :param AL: probability vector corresponding to your label predictions, shape(1, number of examples)
    :param Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape(1, number of examples)

    Returns
    :return:
    :param cost : cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y
    cost = -(np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))/m

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    :param dZ: Gradient of the cost with respect to the linear output (of current layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    :return:
    :param dA:_prev, W, b = cache
    :param dW: Gradient of the cost with respect to W (current layer l), same shape as W
    :param db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    :param dA: post-activation gradient for current layer l
    :param cache: tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    :param activation:  the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    :return:
    :param dA_prev : Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :param dW : Gradient of the cost with respect to W (current layer l), same shape as W
    :param db : Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)


    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev , dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    :param AL: probability vector, output of the forward propagation (L_model_forward())
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat)
    :param caches: list of cashes containing:
                     every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e. l = = 0..L-2
                     the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns
    :return:
    :param grads: A dictionary with the gradients
                   grads["dA" + str(l)] = ...
                   grads["dW" + str(l)] = ...
                   grads["db" + str(l)] = ...
    """
    grads={}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initialize the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_tmp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)], current_cache, "relu")
        grads["dA" + str(l+1)] = dA_prev_tmp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    :param parameters: python dictionary containing your updated parameters
    :param grads: python dictionary containing your gradients, output of L_model_backward
    :param learning_rate: the learning rate

    Returns:
    :return:
    :param parameters : python dictionary containing your updated parameters
                        parameters["W" + str(l)] = ...
                        parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    :param X: input data, of shape (n_x, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: dimensions of the layers (n_x, n_h, n_y)
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if set to True , this will print the cost every 100 iterations

    Returns:
    :return:
    :param parameters: a dictionary containing W1, W2, b1, b2
    """

    np.random.seed(1)
    grads = {}
    costs = []         # to keep track of the cost
    m = X.shape[1]     # number of examples
    (n_x, n_h, n_y) = layers_dims

    #initialize parameters dictionary, by calling one of the funciton you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # loop (gradient descent)
    for i in range(0, num_iterations):
        #forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        # set grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads , learning_rate)

        # Retrieve parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 ==0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments
    :param X: data, numpy array of shape (num_px * num_py * 3, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of exampls)
    :param layer_dims: list containing the input size and each layer size, of lenght (number of layers +1)
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps

    Returns
    :return:
    :param parameters: parameters learned by the model. They can then be used to predict
    """

    np.random.seed(1)
    costs = []        # keep track of cost

    # parameter initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters










def main():
    # load the data set
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Example of a picture
    # index = 10
    # print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
    # plt.imshow(train_x_orig[index])
    # plt.show()

    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

    # Reshape the training and test example
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    # hyper parameters
    n_x = 12288   # num_px * num_px * 3
    n_h = 7
    n_y =1
    layer_dims = (n_x, n_h, n_y)


    # train two layer network
    # parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True)
    #
    # # predict model , change the function in dnn_app_utils_v2 , vectorize with np.ceil
    # predictions_train = predict(train_x, train_y, parameters)
    #
    # predictions_test = predict(test_x, test_y, parameters)

    # now L-layer neural network
    #layers_dims = [12288, 20, 7, 5, 1] # 5 layer model
    layers_dims = [12288, 2000, 700, 500, 1]  # 5 layer model
    #parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

    print("train accuracy")
    pred_train = predict(train_x, train_y, parameters)

    print("test/dev accuracy")
    pred_test = predict(test_x, test_y, parameters)

    #print_mislabeled_images(classes, test_x, test_y, pred_test)




if __name__ == "__main__":
    main()