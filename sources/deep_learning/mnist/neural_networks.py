import functions
import numpy as np


class NeuralNetworkClassifier:
    def __init__(self, input_node_size: int, hidden_node_size: int, output_node_size: int, hidden_layer_size: int = 1):
        np.random.seed(0)
        __init_std = 0.01
        n_sizes = [input_node_size]
        n_sizes.extend([hidden_node_size for _ in range(hidden_layer_size)])
        n_sizes.append(output_node_size)
        self.__w_list = [__init_std * np.random.randn(n_sizes[_i - 1], n_sizes[_i]) for _i in range(1, len(n_sizes))]
        self.__b_list = [__init_std * np.random.randn(n_sizes[_i]) for _i in range(1, len(n_sizes))]
        self.__lr = 0.01
        self.__hid_eval_f = functions.sigmoid
        self.__hid_derived_f = functions.sigmoid_derived
        self.__out_eval_f = functions.soft_max
        self.__loss_f = functions.cross_entropy_error
        self.__out_loss_derived_f = functions.soft_cross_onehot_derived

    def __forward(self, x: np.ndarray) -> list:
        x_list = [x]
        for _i in range(len(self.__w_list) - 1):
            x = x @ self.__w_list[_i] + self.__b_list[_i]
            x = self.__hid_eval_f(x)
            x_list.append(x)
        x = x @ self.__w_list[-1] + self.__b_list[-1]
        x = self.__out_eval_f(x)
        x_list.append(x)
        return x_list

    def __forward_fast(self, x: np.ndarray) -> float:
        for _i in range(len(self.__w_list) - 1):
            x = x @ self.__w_list[_i] + self.__b_list[_i]
            x = self.__hid_eval_f(x)
        x = x @ self.__w_list[-1] + self.__b_list[-1]
        return self.__out_eval_f(x)

    def __backward(self, x_list: list, t: np.ndarray) -> (list, list):
        _dy_loss = self.__out_loss_derived_f(x_list[-1], t)
        _w_grad_list = []
        _b_grad_list = []
        for _i in reversed(range(len(self.__w_list))):
            _b_grad_list.append(np.sum(_dy_loss, axis=0))
            _w_grad_list.append(np.transpose(x_list[_i]) @ _dy_loss)
            _dy_loss = _dy_loss @ np.transpose(self.__w_list[_i])
            _dy_loss = _dy_loss * self.__hid_derived_f(x_list[_i])
        _w_grad_list.reverse()
        _b_grad_list.reverse()
        return _w_grad_list, _b_grad_list

    def __optimize(self, w_grad_list: list, b_grad_list: list) -> None:
        for _i in range(len(w_grad_list)):
            self.__b_list[_i] -= self.__lr * b_grad_list[_i]
            self.__w_list[_i] -= self.__lr * w_grad_list[_i]

    def fit(self, x: np.ndarray, t: np.ndarray):
        x_list = self.__forward(x)
        w_grad_list, b_grad_list = self.__backward(x_list, t)
        self.__optimize(w_grad_list, b_grad_list)

    def predict_proba(self, x: np.ndarray):
        return self.__forward_fast(x)

    def score(self, x: np.ndarray, t: np.ndarray):
        _x_indices = np.argmax(self.predict_proba(x), axis=1)
        _t_indices = np.argmax(t, axis=1)
        return np.sum(_x_indices == _t_indices) / x.shape[0]


if __name__ == '__main__':
    n = NeuralNetworkClassifier(3, 5, 3)
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
