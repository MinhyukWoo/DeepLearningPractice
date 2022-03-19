"""
Activation Function

Provides activation functions

What is Activation functions?
-----
The activation function of a node defines the output of that node given an input or set of inputs.
See with https://en.wikipedia.org/wiki/Activation_function

"""
import numpy as np


def logistic(x, l, k, x_0):
    """
    logistic_function(self, x, x0, k, l)
    create ndarray of logistic activation

    Parameter
    -----
    x: input ndarray
    l: the curve's maximum value
    k: the logistic growth rate
    x0: the x value of the sigmoid's midpoint

    Returns
    -----
    out: ndarray
    """
    return l / (1 + np.exp(-k * (x - x_0)))


def sigmoid(z):
    """
    sigmoid(z)
    create ndarray of sigmoid activation

    Parameter
    ----
    z: input ndarray

    Returns
    -----
    out: ndarray
    """
    return logistic(z, 1, 1, 0)


def relu(z):
    """
    relu(z)
    create ndarray of ReLU activation

    Parameter
    -----
    z: input ndarray

    Returns
    -----
    out: ndarray
    """
    return np.maximum(0, z)


def heaviside_step(z):
    """
    heaviside_step(z)
    create ndarray of heaviside_step activation

    Parameter
    -----
    z: input ndarray

    Returns
    -----
    out: ndarray
    """
    return np.heaviside(z, 0)

