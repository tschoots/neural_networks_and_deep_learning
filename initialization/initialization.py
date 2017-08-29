import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (14.0, 8.0)     # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def initialize_parameters_zeros(layers_dims):
    """
    Arguments
    :param layers_dims: python array (list) containing the size of eache layer

    Returns
    :return:
    :param parameters: python dictionary containing your parameters "W1", "b1", ... ,"WL", "bL":
                         W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                         b1 -- bias weight vector of shape (layers_dims[1], 1)
                         ...
                         WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                         bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims)   # number of layers in the network

    for l in range(1,L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def model(X,Y, learning_rate = 0.01, num_iterations = 15000, print_cost=True, initialization="he"):
    """
    Implements a three layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments
    :param X: input data of shape (x, number of examples
    :param Y: true "label" vector (containing 0 for red dots: 1 for blue dots), of shape (1, number of examples)
    :param learning_rate: learning rate of gradient descent
    :param num_iterations: number of iterations to run gradient descent
    :param print_cost: if true , print cost every 1000 iterations
    :param initialization: flag to choose which initialization to use ("zero", "random", "he")

    Returns
    :return:
    :param parameters: parameters learnt by the model
    """

    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)

        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i,cost))
            costs.append(cost)

        # plot the loss
        fig = plt.figure(2)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        fig.canvas.set_window_title("Cost with learning rate=" + str(learning_rate) + " and " + initialization + " initilization")

    return parameters

def initialize_parameters_random(layers_dims):
    """
    Arguments
    :param layers_dims: python array (list) containing the size of each layer

    Returns
    :return:
    :param parameters: python dictiionary containing your parameter "W1', "b1", ... , "WL", "bL":
                         W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                         b1 -- bias vector of shape (layers_dims[1], 1)
                         ...,
                         WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                         bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)    # This seed makes sure your "random" numbers will be as ours
    parameters = {}
    L = len(layers_dims) # integer representing the number of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*10
        # should be "He initialization" because it's RELU activation function, if tanh activation use "Xavier initialization
        #parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def initialize_parameters_he(layers_dims):
    """
    Arguments
    :param layers_dims: python array (list) containing the size of each layer

    Returns
    :return:
    :param parameters: python dictiionary containing your parameter "W1', "b1", ... , "WL", "bL":
                         W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                         b1 -- bias vector of shape (layers_dims[1], 1)
                         ...,
                         WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                         bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)    # This seed makes sure your "random" numbers will be as ours
    parameters = {}
    L = len(layers_dims) # integer representing the number of layers

    for l in range(1, L):
        # should be "He initialization" because it's RELU activation function, if tanh activation use "Xavier initialization
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def visualize_input_data(train_X, train_Y):
    colors = list()
    pallette = {0: "red", 1: "blue"}
    for c in np.nditer(train_Y):
        colors.append(pallette[int(c)])
    fig = plt.figure(1)
    fig.canvas.set_window_title('Input data initialization')
    fig.suptitle("Input data", fontsize=20)
    #plt.scatter(train_X[0, :], train_X[1, :], color=colors)
    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.grid()




def main():
    print("hallo")

    # load image dataset: blue/red dots in circles
    train_X, train_Y, test_X, test_Y = load_dataset()
    print("train_X %s" %(type(train_X)))
    print(train_X.shape)
    print(train_Y.shape)

    # let's have a look at the input data
    visualize_input_data(train_X, train_Y)

    #init_choice = 'zeros'
    #init_choice = 'random'
    init_choice = 'he'
    #parameters = model(train_X, train_Y, initialization='zeros')
    parameters = model(train_X, train_Y, initialization=init_choice)
    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    print("predictions_train = " + str(predictions_train))
    print("predictions_test  = " + str(predictions_test))

    fig3 = plt.figure(3)
    plt.title("Model with " + init_choice + " initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    plt.show()





if __name__ == "__main__":
    main()

