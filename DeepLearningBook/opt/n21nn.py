'''
!!!------------------------------------- Abhishek Kumar-------------------------------!!!
!!!---------------------------------------24/10/2017----------------------------------!!!
!!!--------------------------------Nueral network library V 2.0 ----------------------!!!
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

    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# Linear Activation Forward for calculating activation part of forward prop------------------------------------
def linactfwd(A_prev, W, b, act):

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

    m = Y.shape[1]  # training batch size
    # cross entropy cost

    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    #cost  = (1 /2/ m) * (np.dot((Y-AL),(Y-AL).T))

    cost = np.squeeze(cost)
    return cost


# Linear Backward computing the gradients for current node --------------------------------------------
def linback(dZ, cache):

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

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# Model to train the network  ------------------------------------------------------------------------------------
def llayermodel(X, Y, layers, learning_rate=0.0007, num_epochs=10000, print_cost=True, printerval=1000):

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


# defining new cost function
def compcost_reg(AL, Y, ld, parameters, reg):
    m = Y.shape[1]  # training batch size
    # cross entropy cost

    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)

    L = len(parameters) // 2  # number of layers in the neural network
    if (reg == 'l2reg'):
        for l in range(L):
            cost = cost + ld * np.sum(np.sum(parameters["W" + str(l + 1)] ** 2)) / 2 / m

    elif (reg == 'l1reg'):
        alpha = 10 ** -20
        for l in range(L):
            cost = cost + ld * np.sum(np.sum(np.sqrt(parameters["W" + str(l + 1)] ** 2 + alpha))) / m
    else:
        pass

    return cost


# defining new update rules:
def updateparams_reg(parameters, grads, learning_rate, ldbym, reg):
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        if (reg == 'l2reg'):
            parameters["W" + str(l + 1)] = (1 - ldbym) * parameters["W" + str(l + 1)] - learning_rate * grads[
                "dW" + str(l + 1)]
        elif (reg == 'l1reg'):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads[
                "dW" + str(l + 1)] - ldbym * np.sign(parameters["W" + str(l + 1)])
        else:
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]

        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# for implementation of dropout we need to modify the layer calculation rules:
def lmodelfwd_reg(X, parameters, streg='none', stld=0.5):

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network L = len(w's) + len(b's) = 2xdepth
    masks = {}
    for l in range(1, L):
        A_prev = A
        if (streg == 'dout'):
            A, cache = linactfwd(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            mask = np.random.binomial(1, stld, size=A.shape) / stld
            masks['m' + str(l)] = mask
        else:
            A, cache = linactfwd(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")

        caches.append(cache)

    AL, cache = linactfwd(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, caches, masks

def lmodelback_reg(AL, Y, caches, masks, streg= 'none'):

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
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = nn.linactback(dAL, current_cache, "sigmoid")
    grads["dA" + str(L)] = grads["dA" + str(L)]*masks['m' + str(L-1)]
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        if (streg=='dout' and l > 1):
            dA_prev_temp, dW_temp, db_temp = nn.linactback(grads["dA" + str(l + 2)], current_cache, "relu")
            dA_prev_temp = dA_prev_temp*masks['m'+str(l)]
        else:
            dA_prev_temp, dW_temp, db_temp = nn.linactback(grads["dA" + str(l + 2)], current_cache, "relu")

        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# defining new model with regularization parameter:........
def llayermodel_reg(X, Y, xval, yval, layers, learning_rate=0.0007, streg='none', stld=0.5, ld=0.001,
                  reg='none', estop=True, max_passes=100, num_epochs=10000, print_cost=True, printerval=1000,
                  plotcost=True):
    np.random.seed(1)
    costs = []  # keep track of cost

    [n, m] = X.shape
    # Parameters initialization.
    parameters = nn.initparams(layers)

    valcst = 10 ** 5
    passes = 0
    oparams = parameters

    # Loop (gradient descent)
    if (streg == 'dout'):
        if (reg == 'l2reg' or reg == 'l1reg'):
            for i in range(0, num_epochs):
                # Forward propagation
                a3, caches, masks = lmodelfwd_reg(X, parameters, streg, stld)

                # Compute cost
                if (plotcost):
                    cost = compcost_reg(a3, Y, ld, parameters, reg)

                # Backward propagation
                grads = lmodelback_reg(a3, Y, caches, masks, streg)

                # Update parameters
                parameters = updateparams_reg(parameters, grads, learning_rate, ld / m, reg)

                # Print the cost every 100 training example
                if (plotcost):
                    if print_cost and i % printerval == 0:
                        print("Cost after iteration %i: %f" % (i, cost))
                    if i % printerval == 0:
                        costs.append(cost)

                if (estop and i % 10 == 0):
                    yvald = nn.predict(xval, parameters)
                    vcost = compcost_reg(yvald, yval, ld, parameters, reg)
                    if (vcost < valcst):
                        passes = 0
                        valcst = vcost
                        oparams = parameters
                    else:
                        if (passes > max_passes):
                            parameters = oparams
                            print("breaking the loop........")
                            break
                        else:
                            passes = passes + 1




        else:
            for i in range(0, num_epochs):
                # Forward propagation
                a3, caches, masks = lmodelfwd_reg(X, parameters, streg, stld)

                # Compute cost
                if (plotcost):
                    cost = nn.compcost(a3, Y)

                # Backward propagation
                grads = lmodelback_reg(a3, Y, caches, masks, streg)

                # Update parameters
                parameters = nn.updateparams(parameters, grads, learning_rate)

                # Print the cost every 100 training example
                if (plotcost):
                    if print_cost and i % printerval == 0:
                        print("Cost after iteration %i: %f" % (i, cost))
                    if i % printerval == 0:
                        costs.append(cost)

                if (estop and i % 10 == 0):
                    yvald = nn.predict(xval, parameters)
                    vcost = nn.compcost(yvald, yval)
                    if (vcost < valcst):
                        passes = 0
                        valcst = vcost
                        oparams = parameters
                    else:
                        if (passes > max_passes):
                            parameters = oparams
                            print("breaking the loop........")
                            break
                        else:
                            passes = passes + 1
    else:
        if (reg == 'l2reg' or reg == 'l1reg'):
            for i in range(0, num_epochs):
                # Forward propagation
                a3, caches = nn.lmodelfwd(X, parameters)

                # Compute cost
                if (plotcost):
                    cost = compcost_reg(a3, Y, ld, parameters, reg)

                # Backward propagation
                grads = nn.lmodelback(a3, Y, caches)

                # Update parameters
                parameters = updateparams_reg(parameters, grads, learning_rate, ld / m, reg)

                # Print the cost every 100 training example
                if (plotcost):
                    if print_cost and i % printerval == 0:
                        print("Cost after iteration %i: %f" % (i, cost))
                    if i % printerval == 0:
                        costs.append(cost)

                if (estop and i % 10 == 0):
                    yvald = nn.predict(xval, parameters)
                    vcost = compcost_reg(yvald, yval, ld, parameters, reg)
                    if (vcost < valcst):
                        passes = 0
                        valcst = vcost
                        oparams = parameters
                    else:
                        if (passes > max_passes):
                            parameters = oparams
                            print("breaking the loop........")
                            break
                        else:
                            passes = passes + 1




        else:
            for i in range(0, num_epochs):
                # Forward propagation
                a3, caches = nn.lmodelfwd(X, parameters)

                # Compute cost
                if (plotcost):
                    cost = nn.compcost(a3, Y)

                # Backward propagation
                grads = nn.lmodelback(a3, Y, caches)

                # Update parameters
                parameters = nn.updateparams(parameters, grads, learning_rate)

                # Print the cost every 100 training example
                if (plotcost):
                    if print_cost and i % printerval == 0:
                        print("Cost after iteration %i: %f" % (i, cost))
                    if i % printerval == 0:
                        costs.append(cost)

                if (estop and i % 10 == 0):
                    yvald = nn.predict(xval, parameters)
                    vcost = nn.compcost(yvald, yval)
                    if (vcost < valcst):
                        passes = 0
                        valcst = vcost
                        oparams = parameters
                    else:
                        if (passes > max_passes):
                            parameters = oparams
                            print("breaking the loop........")
                            break
                        else:
                            passes = passes + 1

    # plot the cost
    if (plotcost):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations /' + str(printerval))
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


# defining bags
def make_bags(x, y, k):
    m = y.shape[1]
    np.random.seed(12)
    bags = {}
    for i in range(k):
        permutation = np.random.randint(0, m - 1, m) + (np.random.randn(1, m) > 0)
        bags['bag_x' + str(i)] = x[:, permutation]
        bags['bag_y' + str(i)] = y[:, permutation]

    return bags


def train_bagging(x_str, y_str, x_val, y_val, layers, k=5, learning_rate=0.9, streg='none', stld=0.5, ld=0.001, reg='l1reg', estop=False,
                  max_passes=100, num_epochs=15000, print_cost=False, printerval=1000):
    bags = make_bags(x_str, y_str, k)
    params = {}
    for i in range(k):
        print('Training bag no.' + str(i))
        x_trn = bags['bag_x'+str(i)].reshape(x_str.shape[0],x_str.shape[1])
        y_trn = bags['bag_y'+str(i)].reshape(y_str.shape[0],y_str.shape[1])

        params['p' + str(i)] = llayermodel_reg(x_trn, y_trn, x_val, y_val, layers, learning_rate=learning_rate, streg=streg, stld=stld, ld=ld,
                        reg=reg, estop=estop, max_passes=max_passes, num_epochs=num_epochs,
                        print_cost=print_cost, printerval=printerval, plotcost=False)

    return params


def predict_bagging(x, params):
    k = len(params)
    pY = predict(x, params['p0'])
    for i in range(1, k):
        pY = pY + predict(x, params['p' + str(i)])

    pY = pY / k

    return pY