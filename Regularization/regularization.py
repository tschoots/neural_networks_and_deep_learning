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
            a3. cache = forward_propagation_with_dropout(X, parameters, keep_prob)

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
    :param X:
    :param Y:
    :param cache:
    :param lambd:
    :return:
    """


def main():
    print("hallo")
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    visualize_input_data(train_X, train_Y)

    # train the model
    parameters = model(train_X, train_Y)
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