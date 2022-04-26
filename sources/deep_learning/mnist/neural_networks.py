import functions
import numpy as np


class NeuralNetworkClassifier:
    def __init__(
            self, input_node_size: int, hidden_node_size: int,
            output_node_size: int, hidden_layer_size: int = 1
    ):
        # Restriction to protect side effect.
        np.random.seed(0)

        # =====Local variable Declaration=====
        # Declare random array's deviation.
        __init_std = 0.01
        # Declare each node size list.
        n_sizes = [input_node_size]
        n_sizes.extend([hidden_node_size for _ in range(hidden_layer_size)])
        n_sizes.append(output_node_size)
        # =====End of Local variable Declaration=====

        # =====Attribute declaration.======
        # Declare weight list.
        self.__w_list = [
            __init_std * np.random.randn(n_sizes[_i - 1], n_sizes[_i])
            for _i in range(1, len(n_sizes))
        ]
        # Declare bias list.
        self.__b_list = [
            __init_std * np.random.randn(n_sizes[_i])
            for _i in range(1, len(n_sizes))
        ]
        # Declare hidden layer's evaluation function.
        self.__hid_eval_f = functions.sigmoid
        # Declare output layer's evaluation function.
        self.__out_eval_f = functions.softmax
        # Declare loss function.
        self.__loss_f = functions.cross_entropy_error
        # Declare some derived function to use backpropagation.
        self.__hid_derived_f = functions.sigmoid_derived
        self.__out_loss_derived_f = functions.soft_cross_onehot_derived
        # Declare learning rate.
        self.__lr = 0.01
        # =====End of Attribute declaration=====

    def __forward(self, x: np.ndarray) -> list:
        """forward pass to get list of layer node's inputs.

        Parameters
        ----------
        x :
            numpy.ndarray

        Returns
        -------
        x_list :
            list of numpy.ndarray, x_list is list of each node's inputs.
            First one is input layer node's input.
            Last one is output layer node's evaluated output.
        """
        # Declare list of each layer node's inputs.
        x_list = [x]
        # Append the hidden layer node's inputs.
        for _i in range(len(self.__w_list) - 1):
            x = x @ self.__w_list[_i] + self.__b_list[_i]
            x = self.__hid_eval_f(x)
            x_list.append(x)
        # Append the output layer node's evaluated outputs.
        x = x @ self.__w_list[-1] + self.__b_list[-1]
        x = self.__out_eval_f(x)
        x_list.append(x)
        return x_list

    def __forward_fast(self, x: np.ndarray) -> float:
        """forward pass to just calculate the output value.

        Parameters
        ----------
        x :
            numpy.ndarray

        Returns
        -------
        y :
            numpy.ndarray, y is output layer's evaluated output.
        """
        # Forward pass in hidden layers.
        for _i in range(len(self.__w_list) - 1):
            # It means 'Y = XW + B'
            x = x @ self.__w_list[_i] + self.__b_list[_i]
            # Evaluate the hidden layer's output.
            x = self.__hid_eval_f(x)
        # Forward pass in output layers.
        x = x @ self.__w_list[-1] + self.__b_list[-1]
        return self.__out_eval_f(x)

    def __backward(self, x_list: list, t: np.ndarray) -> (list, list):
        """backward pass to get list of weight and bias gradients.
        This is also called backpropagation.

        Parameters
        ----------
        x_list :
            list of numpy.ndarray, x_list is list of each node's inputs.
            First one must be input layer node's input.
            Last one must be output layer node's evaluated output.
        t :
            numpy.ndarray, label of x.

        Returns
        -------
        w_grad_list, b_grad_list :
            tuple of two list, first one is list of weight gradients.
            second one is list of bias gradients.
        """
        # Declare list of gradients.
        _w_grad_list = []
        _b_grad_list = []
        # Declare the loss gradient of output layer's output.
        _dy_loss = self.__out_loss_derived_f(x_list[-1], t)
        # Append the loss gradient of hidden and input layer's output.
        for _i in reversed(range(len(self.__w_list))):
            # Append the loss gradient of bias.
            _b_grad_list.append(np.sum(_dy_loss, axis=0))
            # Append the loss gradient of weight.
            _w_grad_list.append(np.transpose(x_list[_i]) @ _dy_loss)
            # Set it to the loss gradient of layer's input.
            _dy_loss = _dy_loss @ np.transpose(self.__w_list[_i])
            # Set it to the loss gradient of previous layer's output.
            _dy_loss = _dy_loss * self.__hid_derived_f(x_list[_i])
        # Reverse the list of gradient. Because of direction.
        _w_grad_list.reverse()
        _b_grad_list.reverse()
        return _w_grad_list, _b_grad_list

    def __optimize(self, w_grad_list: list, b_grad_list: list) -> None:
        """adjust weight and bias using gradient, learning rate and so on.

        Parameters
        ----------
        w_grad_list :
            list of numpy.ndarray, list of weight gradients.
        b_grad_list :
            list of numpy.ndarray, list of bias gradients.
        """
        for _i in range(len(w_grad_list)):
            # Simple gradient descent.
            self.__b_list[_i] -= self.__lr * b_grad_list[_i]
            self.__w_list[_i] -= self.__lr * w_grad_list[_i]

    def fit(self, x: np.ndarray, t: np.ndarray):
        """train the model using training data and labels.

        Parameters
        ----------
        x :
            numpy.ndarray, training data.
        t :
            numpy.ndarray, training labels.
        """
        # Forward pass to get node's input list.
        x_list = self.__forward(x)
        # Backward pass to get gradient list.
        w_grad_list, b_grad_list = self.__backward(x_list, t)
        # Adjust the weight and bias to better prediction.
        self.__optimize(w_grad_list, b_grad_list)

    def predict_proba(self, x: np.ndarray):
        """get prediction by probability.

        Parameters
        ----------
        x :
            numpy.ndarray, data.

        Returns
        -------
        p :
            numpy.ndarray, prediction probability.
        """
        return self.__forward_fast(x)

    def score(self, x: np.ndarray, t: np.ndarray):
        """get model accuracy.

        Parameters
        ----------
        x :
            numpy.ndarray, test data.
        t :
            numpy.ndarray, test labels.

        Returns
        -------
        cost :
            float, score of model accuracy.
        """
        # Declare Indices of row max probability.
        _x_indices = np.argmax(self.predict_proba(x), axis=1)
        # Declare Indices of row labels which is 1.
        _t_indices = np.argmax(t, axis=1)
        # Return average count of same index.
        return np.sum(_x_indices == _t_indices) / x.shape[0]
