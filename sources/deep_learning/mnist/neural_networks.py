import functions
import numpy as np


class TwoLayerNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layer_size: int = 1):
        np.random.seed(0)
        __init_std = 0.01
        n_sizes = [input_size]
        n_sizes.extend([hidden_size for _ in range(hidden_layer_size)])
        n_sizes.append(output_size)
        self.__w_arr = [__init_std * np.random.randn(n_sizes[_i-1], n_sizes[_i]) for _i in range(1, len(n_sizes))]
        self.__b_arr = [__init_std * np.random.randn(n_sizes[_i]) for _i in range(1, len(n_sizes))]
        self.__lr = 0.01
        self.__hidden_eval_f = functions.sigmoid
        self.__out_eval_f = functions.soft_max
        self.__loss_f = functions.cross_entropy_error

    def __forward(self, x: np.ndarray) -> list:
        x_list = [x]
        for _i in range(len(self.__w_arr) - 1):
            x = x @ self.__w_arr[_i] + self.__b_arr[_i]
            x = self.__hidden_eval_f(x)
            x_list.append(x)
        x = x @ self.__w_arr[-1] + self.__b_arr[-1]
        x = self.__out_eval_f(x)
        x_list.append(x)
        return x_list

    def __forward_fast(self, x: np.ndarray) -> float:
        for _i in range(len(self.__w_arr) - 1):
            x = x @ self.__w_arr[_i] + self.__b_arr[_i]
            x = self.__hidden_eval_f(x)
        x = x @ self.__w_arr[-1] + self.__b_arr[-1]
        return self.__out_eval_f(x)

    def __backward(self, x_list: list, t: np.ndarray) -> (list, list):
        _dy_loss = x_list[-1] - t
        _b_grad = []
        _w_grad = []
        for _i in reversed(range(len(self.__w_arr))):
            _b_grad.append(np.sum(_dy_loss, axis=0))
            _w_grad.append(np.transpose(x_list[_i]) @ _dy_loss)
            _dy_loss = _dy_loss @ np.transpose(self.__w_arr[_i])
            _dy_loss = _dy_loss * x_list[_i] * (1 - x_list[_i])
        _w_grad.reverse()
        _b_grad.reverse()
        return _w_grad, _b_grad

    def __optimize(self, w_grad: list, b_grad: list) -> None:
        for _i in range(len(w_grad)):
            self.__b_arr[_i] -= self.__lr * b_grad[_i]
            self.__w_arr[_i] -= self.__lr * w_grad[_i]

    def fit(self, x: np.ndarray, t: np.ndarray):
        x_list = self.__forward(x)
        w_grad, b_grad = self.__backward(x_list, t)
        self.__optimize(w_grad, b_grad)

    def predict_proba(self, x: np.ndarray):
        return self.__forward_fast(x)

    def score(self, x: np.ndarray, y: np.ndarray):
        _x_indices = np.argmax(self.predict_proba(x), axis=1)
        _y_indices = np.argmax(y, axis=1)
        return np.sum(_x_indices == _y_indices) / x.shape[0]


if __name__ == '__main__':
    n = TwoLayerNeuralNetwork(3, 5, 3)
    sample_2d = np.array([
        [1, 3, 2],
        [5, 7, 9],
        [4, -3, 2],
        [-1, -2, -3],
    ])
    sample_label = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    for i in range(1000):
        n.fit(sample_2d, sample_label)
    print(n.predict_proba(sample_2d))
    print(n.score(sample_2d, sample_label))
