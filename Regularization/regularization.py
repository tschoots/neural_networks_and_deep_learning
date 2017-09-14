import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (14.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def visualize_input_data(train_X, train_Y):
    fig = plt.figure(1)
    fig.canvas.set_window_title("Input data visualization for regularization")
    fig.suptitle("Input data", fontsize=20)
    plt.scatter(train_X[0,:], train_X[1,:], c=train_Y, s=40, cmap=plt.cm.Spectral)

def model(X, Y, learning_rate=0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob =1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments
    :param X: input data, of shape (input size, number of examples)
    :param Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    :param learning_rate: learning rate of the optimization
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: If True, print the cost every 10000 itereations
    :param lambd: regularization hyperparameter, scalar
    :param keep_prob: probability of keeping a neuron active during drop-out, scalar

    Returns
    :return:
    :param parameters: parameters learned by the model. They can then be used to predict
    """

    grads = {}
    costs = []              # to keep track of the cost
    m = X.shape[1]          # number of examples
    layer_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary
    parameters = initialize_parameters(layer_dims)

    # loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # Backward propagation.
        assert(lambd==0 or keep_prob==1)        # it is possible to use both L2 regularization and dropout
                                                # but this assignment will only explore one at a time
        if lambd==0 and keep_prob==1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    fig2 = plt.figure(2)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations x 1.000')
    fig2.canvas.set_window_title('Cost')
    fig2.suptitle("Cost: Learning rate = " + str(learning_rate) + " lambda = " + str(lambd) + " keep_prob = " + str(keep_prob), fontsize=20)

    return parameters

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization.

    Arguments:
    :param A3:post-activation, output of forward propagation, of shape (output size, number of examples)
    :param Y: "true" labels vector, of shape (output size, number of examples)
    :param parameters: python dictionary containing parameters of the model
    :param lambd: the regularization hyper parameter lambda number

    Returns
    :return:
    :param cost: value of the regularized loss function
    """

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)  # this give the cross entropy part of the cost

    # sugestion take the range over the keys of the dictionary makes is parameterizable
    L2_regularization_cost = lambd*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))/(2*m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline modle to which we added an L2 regularization

    Arguments
    :param X:input dataset , of shape (input size, number of examples)
    :param Y:"true" labels vector, of shape (output size, number of examples)
    :param cache: chache output from forward_propagation()
    :param lambd: regularization hyperparameter, scalar

    Returns
    :return:
    :param gradients: A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd/m * W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
             "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
             "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return grads

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID

    Arguments:
    :param X: input dataset, of shape (x, number of examples
    :param parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                          W1 -- weight matrix of shape (20, 2)
                          b1 -- bias vector of shapen(20,1)
                          W2 -- weight matrix of shape (3, 20)
                          b2 -- bias vector of shape (3, 1)
                          W3 -- weight matrix of shape (1, 3)
                          b3 -- bias vector of shape (1,1)
    :param keep_prob: probability of keeping a neuron active during drop-out, scalar

    Returns
    :return:
    :param A3: last activation value , output of the forward propagation, of shape (1, number of samples)
    :param cache: tuple , information stored for computing the backward propagation

    remarks
    the keep_prob is the same for all hidden layers, let's make this more flexible and independent of the number of layers
    or per layer, maybe a keep_prob list per layer like the layers_dims, maybe layer_dims and this one should be in some
    sort of hyper parameters dictionary
    """

    np.random.seed(1)

    # do the following with range over the layers and str(l) construct

    #retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = (np.random.rand(A1.shape[0], A1.shape[1])< keep_prob)
    A1 = np.divide(np.multiply(A1, D1), keep_prob)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = (np.random.rand(A2.shape[0], A2.shape[1]) < keep_prob)
    A2 = np.divide(np.multiply(A2, D2), keep_prob)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prop):
    """
    Implements the backward propagation of our baseline model to which we added dropout.

    Arguments:
    :param X: input dataset, of shape (2, number of examples)
    :param Y: "true" labels vector, of shape (output size, number of examples)
    :param cache: cache output from forward_propagation_with_dropout()
    :param keep_prop: probability of keeping a neuron active during drop-out, scalar

    Returns:
    :return:
    :param gradients: A dictionary with the gradients with respect to each parameter, activation and pre-activation variable
    """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A3.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    # drop out also in backprop
    dA2 = np.divide(np.multiply(dA2, D2), keep_prop)
    #dA2 = np.multiply(dA2, D2)
    #dA2 = np.divide(dA2, keep_prop)

    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    # drop out also in backprop
    dA1 = np.divide(np.multiply(dA1, D1), keep_prop)

    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients



def main():
    print("hallo")
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    visualize_input_data(train_X, train_Y)

    # train the model
    #parameters = model(train_X, train_Y)
    #parameters = model(train_X, train_Y, lambd=0.7)
    parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)
    print("On the trainging set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    fig3 = plt.figure(3)
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda  x: predict_dec(parameters, x.T), train_X, train_Y)

    plt.show()



if __name__ == "__main__":
    main()