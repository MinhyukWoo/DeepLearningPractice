import matplotlib.pyplot as plt

import mnist as mn
from neural_networks import NeuralNetworkClassifier
import numpy as np


if __name__ == '__main__':
    (train_data, train_label), (test_data, test_label) = mn.load_mnist(
        normalize=True, one_hot_label=True)
    network = NeuralNetworkClassifier(784, 50, 10)
    repeat_size = 10000
    data_len = train_data.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_acc_list = []
    test_acc_list = []
    epoch_size = max(1, repeat_size // batch_size)
    epoch_num = 1
    for i in range(repeat_size):
        batch_mask = np.random.choice(data_len, batch_size)
        batch_data = train_data[batch_mask]
        batch_label = train_label[batch_mask]
        network.fit(batch_data, batch_label)
        if i % epoch_size == 0:
            print("epoch_num: ", epoch_num)
            epoch_num += 1
            network.fit(batch_data, batch_label)
            train_acc_list.append(network.score(train_data, train_label))
            test_acc_list.append(network.score(test_data, test_label))
    np.set_printoptions(precision=2, linewidth=143)
    print(test_data[0])
    d = test_data[0]
    d = d[np.newaxis, :]
    y: np.ndarray = network.predict_proba(d)
    y = y.flatten()
    print("predict:", y.argmax())
    print("target:", test_label[0].argmax())
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label="train_acc_list")
    plt.plot(x, test_acc_list, '--', label="test_acc_list")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='upper left')
    plt.show()
