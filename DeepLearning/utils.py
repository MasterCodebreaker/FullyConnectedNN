import numpy as np


def count_correct_pred(predictions: np.array, targets: np.array) -> np.array:
    """
    Arg:  predictions [#batch, #outputs]    targets [#batch, #outputs]
    Turns prediction vectors in to one_hot vectors. I.e [1,15,2,-19] becomes [0,1,0,0], the max coordinate in each collumn gets a 1
    and every other coordinate gets a zero. We then count how many times we makes a correct prediction, i.e., # of times prediciton = targets.

    """
    int_pred = np.zeros_like(predictions)  # Zero matrix
    int_pred[
        np.arange(predictions.shape[0]), predictions.argmax(1)
    ] = 1  # every row gets a one in the collumn correspoining to the max of pred
    count = np.sum(
        (int_pred == targets).all(1)
    )  # Count how many times we predicted correct in the batch
    return count


def batch_loader(X: np.array, Y: np.array, batch_size: int, stochastic: bool = False):
    """
    Imputs X and Y and yields a mini_batch of X and Y. If stochastic, then we shuffle X and Y between every epoch.
    Arg: X, Y
    Yields: X_mini_batch Y_mini_batch
    """
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
    Args: targets: labels/targets of each image of shape: [batch size, num_classes]
          outputs: outputs of model of shape: [batch size, num_classes]
    Ret: Cross entropy error (float)
    """
    loss = (
        -np.einsum(
            "ij -> ", targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs)
        )
        / targets.shape[0]
    )
    return loss


def d_cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> np.array:
    """
    Args: targets: labels/targets of each image of shape: [batch size, num_classes]
          outputs: outputs of model of shape: [batch size, num_classes]
    Ret:
    """
    return (outputs - targets) / outputs.shape[0]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [#batch, 1]
        num_classes: #output layer
    Returns:
        Y: shape [#batch, num classes]
    """
    I = np.eye(num_classes)
    return np.array([I[int(y[0])] for y in Y])


def softmax(z):
    exp_matrix = np.exp(z)
    # Sum rows and keep dimensions to make division easy
    exp_sum = np.sum(exp_matrix, axis=1, keepdims=True)
    y = exp_matrix / exp_sum
    return y


def d_softmax(X):
    f = softmax(X)
    return f * (1 - f)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    fz = tanh(z)
    return 1 - fz ** 2


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
    loss = np.einsum("ij ->", (outputs - targets) ** 2) / outputs.shape[0]
    return loss


def d_mse(outputs, targets) -> np.array:
    return 2 * (outputs - targets) / outputs.shape[0]


def identity(z):
    return z


def d_identity(z):
    return np.eye(z.shape[0], z.shape[1])
