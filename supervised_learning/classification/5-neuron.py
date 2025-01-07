#!/usr/bin/env python3
"""classification project"""
import numpy as np


class Neuron:
    """Neuron class doc"""

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
        self.nx = nx

    @property
    def W(self):
        """weights vector for the neuron"""
        return (self.__W)

    @property
    def b(self):
        """bias for the neuron"""
        return (self.__b)

    @property
    def A(self):
        """activated output of the neuron"""
        return (self.__A)

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return (cost)

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        db = np.sum(dz) / m
        dw = np.matmul(X, dz.T) / m
        self.__W = self.W - (alpha * dw).T
        self.__b = self.b - alpha * db
        return (self.W, self.b)
