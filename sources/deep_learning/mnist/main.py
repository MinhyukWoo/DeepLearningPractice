import matplotlib.pyplot as plt

import mnist as mn
from neural_networks import NeuralNetworkClassifier
import numpy as np


def fit_and_score_model(
        model, train_data, train_labels, test_data, test_labels, epoch_size, batch_size) -> (list, list):
    _train_acc_list = [model.score(train_data, train_labels)]
    _test_acc_list = [model.score(test_data, test_labels)]
    _train_data_size = len(train_data)
    for epoch in range(epoch_size):
        print("epoch:", epoch + 1)
        start = 0
        for start in range(0, _train_data_size - batch_size, batch_size):
            batch_data = train_data[start:start + batch_size]
            batch_labels = train_labels[start:start + batch_size]
            model.fit(batch_data, batch_labels)
        last_batch_data = train_data[start:]
        last_batch_labels = train_labels[start:]
        model.fit(last_batch_data, last_batch_labels)
        _train_acc_list.append(model.score(train_data, train_labels))
        _test_acc_list.append(model.score(test_data, test_labels))
    return _train_acc_list, _test_acc_list


def plt_show_acc_list(train_acc_list, test_acc_list):
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label="train_acc_list")
    plt.plot(x, test_acc_list, '--', label="test_acc_list")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='upper left')
    plt.show()


def main():
    (train_data, train_labels), (test_data, test_labels) = mn.load_mnist(
        normalize=True, one_hot_label=True)
    model = NeuralNetworkClassifier(784, 50, 10, 2)
    epoch_size = 20
    batch_size = 32
    train_acc_list, test_acc_list = fit_and_score_model(
        model, train_data, train_labels, test_data, test_labels,
        epoch_size, batch_size
    )
    print("model accuracy:", test_acc_list[-1])
    plt_show_acc_list(train_acc_list, test_acc_list)


if __name__ == '__main__':
    main()
