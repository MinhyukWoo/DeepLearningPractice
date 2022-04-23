import functions
import numpy as np


class TwoLayerNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        np.random.seed(0)
        self.__init_std = 0.01
        self.__w_arr = [
            self.__init_std * np.random.randn(input_size, hidden_size),
            self.__init_std * np.random.randn(hidden_size, output_size)
        ]
        self.__b_arr = [
            self.__init_std * np.random.randn(hidden_size),
            self.__init_std * np.random.randn(output_size)
        ]
        self.__lr = 0.01
        self.__hidden_eval_f = functions.sigmoid
        self.__out_eval_f = functions.soft_max
        self.__loss_f = functions.cross_entropy_error

    def fit(self, x: np.ndarray, y: np.ndarray):
        # forward
        _x_arr = [x]
        for i in range(len(self.__w_arr) - 1):
            x = x @ self.__w_arr[i] + self.__b_arr[i]
            x = self.__hidden_eval_f(x)
            _x_arr.append(x)
        x = x @ self.__w_arr[-1] + self.__b_arr[-1]
        _p = self.__out_eval_f(x)

        # backward
        _dy_loss = _p - y
        _b_grad = [np.sum(_dy_loss, axis=0)]
        _w_grad = [np.transpose(_x_arr[-1]) @ _dy_loss]
        _dy_loss = _dy_loss @ np.transpose(self.__w_arr[-1])
        _dy_loss = _dy_loss * (_x_arr[-1] * (1 - _x_arr[-1]))
        for i in reversed(range(len(self.__w_arr) - 1)):
            _b_grad.append(np.sum(_dy_loss, axis=0))
            _w_grad.append(np.transpose(_x_arr[i]) @ _dy_loss)
            _dy_loss = _dy_loss @ np.transpose(self.__w_arr[i])
            _dy_loss = _x_arr[i] * (1 - _x_arr[i])
        _b_grad.reverse()
        _w_grad.reverse()

        # adjust
        for i in range(len(_w_grad)):
            self.__b_arr[i] -= self.__lr * _b_grad[i]
            self.__w_arr[i] -= self.__lr * _w_grad[i]

    def predict_proba(self, x: np.ndarray):
        for i in range(len(self.__w_arr) - 1):
            x = x @ self.__w_arr[i] + self.__b_arr[i]
            x = self.__hidden_eval_f(x)
        x = x @ self.__w_arr[-1] + self.__b_arr[-1]
        return self.__out_eval_f(x)

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
