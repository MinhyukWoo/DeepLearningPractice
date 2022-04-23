import numpy as np


def identity(x: np.ndarray):
    return x


def heaviside_step(x: np.ndarray):
    return np.array(x > 0, dtype=np.int)


def relu(x: np.ndarray):
    return np.maximum(0, x)


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def soft_max(x: np.ndarray):
    _m = np.max(x, axis=1)
    _m = _m.reshape((_m.size, 1))
    _x_exp = np.exp(x - _m)
    if x.ndim == 1:
        return _x_exp / np.sum(_x_exp)
    elif x.ndim == 2:
        sum_x_exp = np.sum(_x_exp, axis=1)
        return _x_exp / sum_x_exp.reshape((sum_x_exp.size, 1))
    else:
        raise ValueError("x.ndim is not 1 or 2")


def mean_squared_error(y: np.ndarray, t: np.ndarray):
    return 0.5 * np.sum(np.square(y - t))


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    _batch_size = y.shape[0]
    _log = np.log(y[np.arange(_batch_size), t.argmax(axis=1)] + 1e-7)
    return -np.sum(_log) / _batch_size


if __name__ == '__main__':
    sample_2d = np.array([
        [1, 3, 2, 4],
        [5, 7, 9, 11],
        [4, -3, 2, -1]
    ])
    print(soft_max(sample_2d))
    print(mean_squared_error(sample_2d, np.zeros_like(sample_2d)))
    sample_3d = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]
    ])
