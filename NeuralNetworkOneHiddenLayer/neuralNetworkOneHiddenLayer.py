import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets



np.random.seed(1) # set a seed so that the results are consistent

def logistic_regression(X,Y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)

    #plt the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions)+np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) + '% ' + "(percentage of correctly labeled datapoints")
    plt.show()
    # conclusion the dataset is not linear seperable so logistic regression doesn't perform well


def layer_sizes(X,Y):
    """
    Arguments:
    :param X: input dataset of shape (input size, number of examples)
    :param Y: labels of shape (output size, number of examples)
    :return:
    :param n_x : the size of the input layer
    :param n_h : the size of the hidden layer
    :param n_y : the size of the output layer
    """

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """

    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer
    :return:
    :params : python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_y)
                        b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization of the weights is random

    W1 = np.random.rand(n_h, n_x)*0.01  # use factor 0.001 because of the tanh activation function keeping close to zero for bigger slope
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    """
    Argument
    :param X: input data of size (n_x, m)
    :param parameters: python dictionary containing your parameters (output of initialization function)
    :return:
    :param A2: the sigmoid output of the second activation
    :param cache: a dictionary containing "Z1", "A1", "Z2", and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # forward propagation
    Z1 = np.dot(W1, X) + b1  # b1 will be broadcasted from (n_x, 1) to (n_x, m)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    :param A2: The sigmoid output of the scond activation , of shape (1, number of examples)
    :param Y: "true" labels vector of shape (1, number of examples)
    :param parameters: python dictionairy containing your parameters W1, b1, W2, b2

    Returns
    :return:
    :param cost : cross-entropy cost given equation (13)
    """
    m = Y.shape[1] # number of examples
    cost = float(-(np.dot(Y, np.log(A2.T)) + np.dot((1 - Y), np.log(1 - A2.T))) / m)

    assert(isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
    :param parameters: python dictionary containing our parameters
    :param cache: a dictionary containing "Z1", "A1", "Z2" and "A2".
    :param X: input data of shape (2, number of examples)
    :param Y: "true" labels vector of shape (1, number of examples)

    Returns:
    :return:
    :param : grads : python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # backward propagation
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using gradient descent update rule given above

    Arguments:
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients
    :param learning_rate: learning rate

    Returns
    :return:
    :param : parameters : python dictionary containing updated parameters
    """
    # retrieve the current parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # retrieve the gradients
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # apply update rule theta = theta - alpha*derivative J/theta
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def nn_model(X, Y, n_h, num_iterations = 1000, print_cost=False):
    """
    Arguments:
    :param X: dataset of shape (2, number of examples)
    :param Y: labels of shape (1, number of examples)
    :param n_h: size of the hidden layer
    :param num_iterations: number of iterations in gradient descent loop
    :param print_cost: if True , print the cost every 1000 iterations

    Returns:

    :return:
    :param parameters : parameters learnt by the model. They can then be used to predict
    """

    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        # forward propagation
        A2, cache = forward_propagation(X, parameters)

        # Cost function
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update,
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:

    :param parameters: python dictionary containing learned parameters
    :param X: input data of size (n_x, m)

    Returns:
    :return:
    :param predictions
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 sing 0.5 as threshold
    A2, cache = forward_propagation(X, parameters)
    predictions = np.ceil(A2 - 0.5)

    return predictions


def visualize_other_datasets(X, Y):
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "guassian_quantiles": gaussian_quantiles}

    for ds in datasets.keys():
        dataset = ds

        X, Y = datasets[dataset]
        X, Y = X.T, Y.reshape(1, Y.shape[0])

        # make blobs binary
        if dataset == "blobs":
            Y = Y%2

        # Visualize the data
        plt.title(dataset)
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        plt.show()


def main():
    X,Y = load_planar_dataset()

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    # first see how logistic regression performs on this problem
    logistic_regression(X,Y)

    # this is just bogus code to check the funtion layer_sizes
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

    # Build a model with a n_h-dimesional hidden_layer
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decistion Boundary for hidden layer size " + str(n_h))
    plt.show()

    # Print accuracy
    predictions = predict(parameters, X)
    print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # Tuning hidden layer size
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [ 1, 2, 3, 4, 5, 20, 50 ]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5,2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        plt.show()

    visualize_other_datasets(X, Y)




if __name__ == "__main__":
    main()