import numpy as np
from typing import Callable


def _numerical_gradient_1d(func: Callable[[np.ndarray], float], x: np.ndarray):
    if x.ndim == 1:
        _h0 = 1e-7
        _grad = np.zeros_like(x)
        for i in range(x.size):
            _h = np.zeros_like(x)
            _h[i] = _h0
            _grad[i] = (func(x + _h) - func(x - _h)) / (2 * _h0)
        return _grad
    else:
        raise ValueError("x.ndim is not 1")


def numerical_gradient(func: Callable[[np.ndarray], float], x: np.ndarray):
    if x.ndim == 1:
        return _numerical_gradient_1d(func, x)
    elif x.ndim == 2:
        _grad_mat = np.zeros_like(x)
        for i in range(x.shape[0]):
            _grad_mat[i, :] = _numerical_gradient_1d(func, x[i, :])
        return _grad_mat
    else:
        raise ValueError("x.ndim is not 1 or 2")


def numerical_gradient_batch(
        func: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    if x.ndim == 2:
        _grad_mat = np.zeros_like(x)
        for j in range(x.shape[1]):
            _h0 = 1e-7
            _h = np.zeros(x.shape[1])
            _h[j] = _h0
            _grad_mat[:, j] = (func(x + _h) - func(x - _h)) / (2 * _h0)
        return _grad_mat
    else:
        raise ValueError


if __name__ == '__main__':
    sample_2d = np.array([
        [1, 3, 2, 4],
        [5, 7, 9, 11],
        [4, -3, 2, -1]
    ], dtype=float)
    print(numerical_gradient(lambda _x: np.sum(np.square(_x)), sample_2d))
    print(numerical_gradient_batch(
        lambda _x: np.sum(np.square(_x), axis=1), sample_2d))
