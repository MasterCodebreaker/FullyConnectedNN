import numpy as np


def batch_loader(X, Y, batch_size, stochastic=False):

    number_of_batches = X.shape[0] // batch_size
    indices = [i for i in range(X.shape[0])]
    if stochastic:
        temp = np.random.permutation(X.shape[0])
        Y = Y[temp]
        X = X[temp]
    for i in range(number_of_batches):
        # select a set of indices for each batch of samples
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        x = X[batch_indices]
        y = Y[batch_indices]
        yield (x, y)


def cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    # loss_n = np.sum(targets * np.log(outputs), axis=1)
    # loss = np.sum(-loss_n) / targets.shape[0]
    loss = (
        -np.einsum(
            "ij -> ", targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs)
        )
        / targets.shape[0]
    )
    return loss


def d_cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> np.array:
    return (outputs - targets) / outputs.shape[0]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    I = np.eye(num_classes)
    return np.array([I[int(y[0])] for y in Y])


def softmax(z):
    """
    z is a #batch times #outputs matrix
    """
    exp_matrix = np.exp(z)
    # Sum rows and keep dimensions to make division easy
    exp_sum = np.sum(exp_matrix, axis=1, keepdims=True)
    y = exp_matrix / exp_sum
    return y


def d_softmax(X):
    f = softmax(X)
    return f * (1 - f)


def relu(z):
    return z * (z > 0)


def d_relu(z):
    return 1 * (z > 0)


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig


def d_sigmoid(z):
    sigma = sigmoid(z)
    return sigma * (1 - sigma)


def difference(outputs, targets) -> np.array:
    loss = np.einsum("ij ->", outputs - targets) / outputs.shape[0]
    return loss


def d_difference(outputs, targets):
    return np.eye(N=outputs.shape[0], M=outputs.shape[1]) / outputs.shape[0]


def mse(outputs, targets) -> float:
    """
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    """
    loss = np.einsum("ij ->", (outputs - targets) ** 2) / outputs.shape[0]
    return loss


def d_mse(outputs, targets) -> np.array:
    return 2 * (outputs - targets) / outputs.shape[0]


def identity(z):
    return z


def d_identity(z):
    return np.eye(z.shape[0], z.shape[1])
