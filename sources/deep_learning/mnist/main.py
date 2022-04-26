import matplotlib.pyplot as plt
import numpy as np
import mnist as mn
from neural_networks import NeuralNetworkClassifier


def fit_and_score_model(
        model, train_data, train_labels, test_data, test_labels,
        epoch_size, batch_size) -> (list, list):
    """Update model with train data and score the train and test set accuracy.

    Parameters
    ----------
    model
    train_data
    train_labels
    test_data
    test_labels
    epoch_size :
        int, the number of times to train whole training data.
    batch_size :
        int, the size of data to be trained at the same time.

    Returns
    -------
    train_acc_list, test_acc_list :
        tuple of two list, first one is training set accuracy list and
        second one is testing set accuracy list. Both are indexed by epoch
    """
    # Declare function's output. Init score before the training.
    train_acc_list = [model.score(train_data, train_labels)]
    test_acc_list = [model.score(test_data, test_labels)]
    # Declare the size of training data
    train_data_size = len(train_data)
    # Train whole training data for epoch_size times by using batch.
    for epoch in range(epoch_size):
        # Print the epoch number to notice the progress.
        print("epoch:", epoch + 1)
        # Declare the start index.
        start = 0
        # Train batch size of training data from first to last.
        for start in range(0, train_data_size - batch_size, batch_size):
            batch_data = train_data[start:start + batch_size]
            batch_labels = train_labels[start:start + batch_size]
            model.fit(batch_data, batch_labels)
        # Train rest of training data.
        rest_batch_data = train_data[start:]
        rest_batch_labels = train_labels[start:]
        model.fit(rest_batch_data, rest_batch_labels)
        # Record the model accuracy to list.
        train_acc_list.append(model.score(train_data, train_labels))
        test_acc_list.append(model.score(test_data, test_labels))
    # End of Training
    # Return the model accuracy list
    return train_acc_list, test_acc_list


def show_model_accuracy_graph(train_acc_list: list, test_acc_list: list) -> None:
    """Draw the graph of model accuracy and show it using matplotlib

    Parameters
    ----------
    train_acc_list :
        list, model accuracy of training set. It is indexed by epoch.
    test_acc_list :
        list, model accuracy of testing set. It is indexed by epoch.

    Returns
    -------
    None
    """
    # Declare the x. x is epoch
    x = np.arange(len(train_acc_list))
    # Draw graph of model accuracy of training data
    plt.plot(x, train_acc_list, label="train_acc_list")
    # Draw graph of model accuracy of testing data using dotted line.
    plt.plot(x, test_acc_list, '--', label="test_acc_list")
    # Show labels of graph
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc='lower left')
    # Restrict the y range.
    plt.ylim(0, 1.0)
    # Show the graph at new window.
    plt.show()


def main():
    # Load the mnist data and split data into training set and testing set
    (train_data, train_labels), (test_data, test_labels) = mn.load_mnist(
        normalize=True, one_hot_label=True
    )
    # Generate NeuralNetworkClassifier for mnist data
    model = NeuralNetworkClassifier(784, 50, 10, hidden_layer_size=3)
    # Declare the size of epoch and batch
    epoch_size = 20
    batch_size = 32
    # Update model for epoch_size times
    #   and record the model accuracy of each epoch.
    train_acc_list, test_acc_list = fit_and_score_model(
        model, train_data, train_labels, test_data, test_labels,
        epoch_size, batch_size
    )
    # Print the model accuracy of testing data.
    print("model accuracy:", test_acc_list[-1])
    # Show graph of model accuracy.
    show_model_accuracy_graph(train_acc_list, test_acc_list)


if __name__ == '__main__':
    # Execute main function
    main()
