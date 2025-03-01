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
