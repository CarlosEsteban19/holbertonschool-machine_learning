#!/usr/bin/env python3
"""classification project"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""

    def __init__(self, nx, layers, activation='sig'):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """number of layers in the neural network"""
        return (self.__L)

    @property
    def cache(self):
        """holds all the intermediary values of the network"""
        return (self.__cache)

    @property
    def weights(self):
        """holds all the wrights and biases of the network"""
        return (self.__weights)

    @property
    def activation(self):
        """activation function for hidden layers"""
        return self.__activation

    def forward_prop(self, X):
        """calculates the forward propagation of the neuron"""
        self.__cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            z = np.matmul(W, Aprev) + b
            if i < self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache[Akey] = self.sigmoid(z)
                else:
                    self.__cache[Akey] = np.tanh(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        weights_c = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                if self.__activation == "sig":
                    g = A * (1 - A)
                    dz = (weights_c["W" + str(i + 1)].T @ dz) * g
                elif self.__activation == "tanh":
                    g = 1 - (A ** 2)
                    dz = (weights_c["W" + str(i + 1)].T @ dz) * g
            dw = (dz @ cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights["W" + str(i)] = self.__weights[
                    "W" + str(i)] - (alpha * dw)
            self.__weights["b" + str(i)] = self.__weights[
                    "b" + str(i)] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains the neuron and updates __weights and __cache"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """saves the instance object to a file in pickle format"""
        import pickle
        if type(filename) is not str:
            return
        if filename[-4:] != ".pkl":
            filename = filename[:] + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """loads a pickled DeepNeuralNetwork object from a file"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None

    def sigmoid(self, z):
        """Applies the sigmoid activation function"""
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def softmax(self, z):
        """Applies the softmax activation function"""
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)
