import numpy as np


def identity(x: np.ndarray) -> np.ndarray:
    """Return the identity of x
    Parameters
    ----------
    x

    Returns
    -------
    y :
        numpy.ndarray, y is x
    """
    return x


def heaviside_step(x: np.ndarray) -> np.ndarray:
    """Return the Heaviside step function of x
    The value of which is zero for negative arguments and one for positive ones.

    References: https://en.wikipedia.org/wiki/Heaviside_step_function

    Parameters
    ----------
    x

    Returns
    -------
    y :
        numpy.ndarray, y is 1 if x > 0, else y is 0
    """
    return np.array(x > 0, dtype=int)


def relu(x: np.ndarray) -> np.ndarray:
    """Return the Relu function of x

    References: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    Parameters
    ----------
    x

    Returns
    -------
    y :
        numpy.ndarray, y is x if x > 0, else y is 0
    """
    return np.maximum(0, x)


def relu_derived(y: np.ndarray) -> np.ndarray:
    """Return the Relu derived function of y.
    It is fast to calculate it because of Automatic differentiation.

    Parameters
    ----------
    y :
        numpy.ndarray, y is the result of relu function.

    Returns
    -------
    dx_y :
        numpy.ndarray, dx_y is 1 if dx_y > 0, else dx_y is 0

    """
    out = np.zeros_like(y)
    out[y > 0] = 1
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Return the sigmoid function of x.
    This sigmoid function is logistic function.

    References : https://en.wikipedia.org/wiki/Sigmoid_function

    Parameters
    ----------
    x

    Returns
    -------
    y :
        numpy.ndarray, y has a value within the open interval (0, 1).
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derived(y: np.ndarray) -> np.ndarray:
    """Return the sigmoid derived function of y.
    It is fast to calculate it because of Automatic differentiation.

    Parameters
    ----------
    y :
        numpy.ndarray, y is the result of sigmoid function.

    Returns
    -------
    dx_y :
        numpy.ndarray, dx_y has a value within the interval (0, 0.25].
        If y is 0.5 (it's mean if x is 0), dx_y is Maximum.
    """
    return y * (1 - y)


def softmax(x: np.ndarray) -> np.ndarray:
    """Return the softmax function of x.
    Softmax normalizes inputs into a probability distribution
    consisting of probabilities proportional to the exponential of the inputs.

    References : https://en.wikipedia.org/wiki/Softmax_function

    Parameters
    ----------
    x

    Returns
    -------
    y :
        numpy.ndarray, y has a value within the open interval (0, 1).
        Sum of y's row is 1.
    """
    # To protect overflow. Result is same because of reduction of a fraction.
    x = x - np.max(x, axis=1)[:, np.newaxis]
    _x_exp = np.exp(x)
    if x.ndim == 1:
        return _x_exp / np.sum(_x_exp)
    elif x.ndim == 2:
        sum_x_exp = np.sum(_x_exp, axis=1)
        return _x_exp / sum_x_exp.reshape((sum_x_exp.size, 1))
    else:
        raise ValueError("x.ndim is not 1 or 2")


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    """Return the mean squared error of y and t.
    This is sum of the average squared difference
    between the estimated values and the actual values.

    References : https://en.wikipedia.org/wiki/Mean_squared_error

    Parameters
    ----------
    y :
        numpy.ndarray, the model-estimated values
    t :
        numpy.ndarray, the actual values.

    Returns
    -------
    cost :
        float, Sum of the average squared difference. 0 is Minimum.
    """
    return 0.5 * np.sum(np.square(y - t))


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """Return the cross entropy error of y and t.

    References : https://en.wikipedia.org/wiki/Cross_entropy

    Parameters
    ----------
    y :
        numpy.ndarray, the model-estimated values
    t :
        numpy.ndarray, the actual values.

    Returns
    -------
    cost :
        float, 0 is Minimum.

    """
    _batch_size = y.shape[0]
    _log = np.log(y[np.arange(_batch_size), t.argmax(axis=1)] + 1e-7)
    return -np.sum(_log) / _batch_size


def soft_cross_onehot_derived(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return the cross_entropy(softmax(x)) derived function.
    t's row has only one 1; rest is 0. It's mean one-hot encoding.
    It is fast to calculate it because of Automatic differentiation.

    Parameters
    ----------
    y :
        numpy.ndarray, the result of softmax function.
    t :
        numpy.ndarray, the actual values.

    Returns
    -------
    dx_loss :
        numpy.ndarray, result of cross_entropy(softmax(x)) derived function.
    """
    return y - t
