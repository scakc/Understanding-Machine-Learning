'''..................................................... by Abhishek Kumar................................................. ####


## first nnmodel file developed only for single output................................................
## hidden layers uses relu and output uses sigmoid activations respecttively......................
'''


# imports-----------------------------------------------------------------------------------------------
import numpy as np
import h5py
import matplotlib.pyplot as plt


# defining activation functions and gradients w.r.t input: ---------------------------------------------
def relu(z): # in actual its a leaky relu performs better
    o = z*(z>0) - 0.001*(z<0)*z
    return o, z

def drelu(dA,z):# in actual its a diff leaky relu performs better
    o = (z>0)*1 - 0.001*(z<=0)
    return o*dA


def sigmoid(z):
    o = 1/(1+np.exp(-z))
    return o, z


def dsigmoid(dA,z):
    o = 1/(1+np.exp(-z))
    return (o*(1-o))*dA


# Initialization of parameters w's and b's ------------------------------------------------------
def initparams(layers):
    """
    Arguments:
    layers     -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers)  # length/ depth of the network

    # NOTE : Representing w's and b's with dimensions, rows -> (i)th layer size, columns -> (i-1)th layer size:
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l], layers[l - 1])
        parameters['b' + str(l)] = np.zeros((layers[l], 1))
        assert (parameters['W' + str(l)].shape == (layers[l], layers[l - 1]))
        assert (parameters['b' + str(l)].shape == (layers[l], 1))

    return parameters


# Linear Forward for calculating linear part of forward prop-----------------------------------------------
def linfwd(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# Linear Activation Forward for calculating activation part of forward prop------------------------------------
def linactfwd(A_prev, W, b, act):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if act == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linfwd(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif act == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linfwd(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)  # cache = ((input, W, b), (output, z)) for that layer
    return A, cache


# Complete forward propagation -------------------------------------------------------------------
def lmodelfwd(X, parameters):

    """
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network L = len(w's) + len(b's) = 2xdepth

    for l in range(1, L):
        A_prev = A
        A, cache = linactfwd(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = linactfwd(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, caches


# Cost function----------------------------------------------------------------------------------
def compcost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector , shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]  # training batch size
    # cross entropy cost

    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    #cost  = (1 /2/ m) * (np.dot((Y-AL),(Y-AL).T))

    cost = np.squeeze(cost)
    return cost


# Linear Backward computing the gradients for current node --------------------------------------------
def linback(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# Linear Activation backward computing prev layer grad along with update values------------------------
def linactback(dA, cache, act):
    """
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if act == "relu":
        dZ = drelu(dA, activation_cache)
        dA_prev, dW, db = linback(dZ, linear_cache)

    elif act == "sigmoid":
        dZ = dsigmoid(dA, activation_cache)
        dA_prev, dW, db = linback(dZ, linear_cache)

    return dA_prev, dW, db


# Complete backprop model-------------------------------------------------------------------
def lmodelback(AL, Y, caches):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    n = AL.shape[0]
    Y = Y.reshape(AL.shape)  
	# after this line, Y is the same shape as AL
    # Initializing the backpropagation
	# Zero divide avoided
    en = 10**-20 # a very small number

    try:
        dAL = (-np.divide(Y, AL) + np.divide(1 - Y, 1 - AL))/m
    except ZeroDivisionError:
        dAL = (-np.divide(Y, AL+np.sign(AL)*en) + np.divide(1 - Y, 1 - AL + np.sign(1-AL)*en))/m

    #dAL = -(1/m)*(Y-AL)


    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linactback(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linactback(grads["dA" + str(l + 2)], current_cache, "relu")

        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# Updating the parameters-----------------------------------------------------------------------------------------
def updateparams(parameters, grads, learning_rate):
    """
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# Model to train the network  ------------------------------------------------------------------------------------
def llayermodel(X, Y, layers, learning_rate=0.0007, num_epochs=10000, print_cost=True, printerval=1000):
    """
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initparams(layers)

    # Loop (gradient descent)
    for i in range(0, num_epochs):
        # Forward propagation
        a3, caches = lmodelfwd(X, parameters)

        # Compute cost
        cost = compcost(a3, Y)

        # Backward propagation
        grads = lmodelback(a3, Y, caches)

        # Update parameters
        parameters = updateparams(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % printerval == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if i % printerval == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations /' + str(printerval))
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# Function to predict the output----------------------------------------------------------------------------------
def predict(X, parameters):
    AL, caches = lmodelfwd(X, parameters)
    return AL