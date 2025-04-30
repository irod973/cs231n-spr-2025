from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy import typing as npt


class Data(NamedTuple):
    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.uint8]


# Constants

N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes


def generate_data() -> Data:
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype="uint8")
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return Data(X=X, y=y)


def visualize_data(data: Data) -> None:
    # lets visualize the data:
    plt.scatter(data.X[:, 0], data.X[:, 1], c=data.y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def plot_linear_classifier(
    data: Data, W: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> None:
    # Some magic function from the example, don't ask me to recreate it
    h = 0.02
    x_min, x_max = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
    y_min, y_max = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    # fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(data.X[:, 0], data.X[:, 1], c=data.y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def plot_twolayer_net(
    data: Data,
    W1: npt.NDArray[np.float64],
    b1: npt.NDArray[np.float64],
    W2: npt.NDArray[np.float64],
    b2: npt.NDArray[np.float64],
) -> None:
    # Some magic function from the example, don't ask me to recreate it
    h = 0.02
    x_min, x_max = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
    y_min, y_max = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1) + b1), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    # fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(data.X[:, 0], data.X[:, 1], c=data.y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def preprocess_data(data: Data) -> Data:
    # Normally we would want to preprocess the dataset so that each feature has zero mean and unit standard deviation,
    # but in this case the features are already in a nice range from -1 to 1, so we skip this step.
    return data


def init_params(
    dim: int, num_classes: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    W = 0.01 * np.random.randn(dim, num_classes)
    b = np.zeros((1, num_classes))
    return W, b


def train_linear_classifier(
    data: Data,
    dim: int = D,
    num_classes: int = K,
    num_iterations: int = 200,
    step_size: float = 1e-0,
    reg: float = 1e-3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    W, b = init_params(dim=dim, num_classes=num_classes)
    logger.debug(f"{W.shape=} {W.dtype=} {b.shape=} {b.dtype=}")
    num_examples = data.X.shape[0]
    logger.info(f"Training linear classifier with {num_iterations=}")
    for i in range(num_iterations):
        # Compute the scores
        scores = np.dot(data.X, W) + b

        # Compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_examples), data.y])

        # Compute the loss
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W)
        loss = data_loss + reg_loss
        if i % 10 == 0:
            logger.info(f"iteration {i}: {loss=}")

        # Do backprop - I simply copied the gradients as given from the notes
        # See derivation: https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
        dscores = probs
        dscores[range(num_examples), data.y] -= 1
        dscores /= num_examples

        dW = np.dot(data.X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        dW += reg * W  # the regularization gradient

        # Parameter update
        W += -step_size * dW
        b += -step_size * db

    logger.info("Done training. Evaluating training set accuracy")
    scores = np.dot(data.X, W) + b
    predicted_class = np.argmax(scores, axis=1)
    logger.info(f"Training accuracy: {np.mean(predicted_class == data.y)}")

    return W, b


def train_twolayer_network(
    data: Data,
    hidden_layer_size: int,
    dim: int = D,
    num_classes: int = K,
    num_iterations: int = 200,
    step_size: float = 1e-0,
    reg: float = 1e-3,
) -> list[npt.NDArray[np.float64]]:
    # Initialize params
    W1 = 0.01 * np.random.randn(dim, hidden_layer_size)
    b1 = np.zeros((1, hidden_layer_size))
    W2 = 0.01 * np.random.randn(hidden_layer_size, num_classes)
    b2 = np.zeros((1, num_classes))

    num_examples = data.X.shape[0]

    for i in range(num_iterations):
        # Compute the scores - using relu activation for hidden layer
        hidden_layer = np.maximum(0, np.dot(data.X, W1) + b1)  # elementwise max
        scores = np.dot(hidden_layer, W2) + b2  # [N x K]
        logger.debug(f"{scores.shape == (num_examples, num_classes)=}")

        # Compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        logger.debug(f"{probs.shape == (num_examples, num_classes)=}")

        # Compute the loss
        correct_logprobs = -np.log(probs[range(num_examples), data.y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            logger.info(f"iteration {i}: {loss=}")

        # Do backprop
        # Again, gradients are assumed given from the notes
        dscores = probs
        dscores[range(num_examples), data.y] -= 1
        dscores /= num_examples
        # Backprop on output/classifier layer
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # Backprop on hidden layer
        dhidden = np.dot(dscores, W2.T)
        # Backprop the ReLU activation
        dhidden[hidden_layer <= 0] = 0
        # Backprop on hidden layer params
        dW1 = np.dot(data.X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        # Add regularization gradients
        dW2 += reg * W2
        dW1 += reg * W1

        # Parameter update
        W1 += -step_size * dW1
        b1 += -step_size * db1
        W2 += -step_size * dW2
        b2 += -step_size * db2

    logger.info("Done training. Evaluating training set accuracy")
    hidden_layer = np.maximum(0, np.dot(data.X, W1) + b1)  # elementwise max
    scores = np.dot(hidden_layer, W2) + b2  # [N x K]
    predicted_class = np.argmax(scores, axis=1)
    logger.info(f"Training accuracy: {np.mean(predicted_class == data.y)}")
    return W1, b1, W2, b2


def main() -> None:
    data: Data = generate_data()
    # visualize_data(data)

    data = preprocess_data(data)
    logger.debug(f"{data.X.shape=} {data.X.dtype=} {data.y.shape=} {data.y.dtype=}")

    # Linear classifier
    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    W, b = train_linear_classifier(data=data, dim=D, num_classes=K, step_size=step_size, reg=reg)
    # plot_linear_classifier(data, W, b)

    # The linear classifier is not great, let's try a neural network instead
    hidden_layer_size = 100
    W1, b1, W2, b2 = train_twolayer_network(
        data=data,
        dim=D,
        num_classes=K,
        step_size=step_size,
        reg=reg,
        num_iterations=10000,
        hidden_layer_size=hidden_layer_size,
    )

    plot_twolayer_net(data, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
