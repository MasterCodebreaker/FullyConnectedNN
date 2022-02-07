import numpy as np
from utils import (
    sigmoid,
    d_sigmoid,
    mse,
    d_mse,
    batch_loader,
    d_difference,
    difference,
    d_cross_entropy_loss,
    cross_entropy_loss,
    relu,
    d_relu,
    softmax,
    d_softmax,
    identity,
    d_identity,
)

# from network import Network


class Layer:
    def __init__(
        self,
        loss_function: str,
        n_prev_layer: int,
        n_this_layer: int,
        actfunc: str = "sigmoid",
        output_layer: bool = False,
        softmax: bool = False,
    ) -> None:
        self.softmax = softmax
        self.loss_function = loss_function
        self.output_layer = output_layer
        self.activations = None
        self.d_activations = None
        self.z = None
        self.error = None
        self.d_error = None
        self.input = None
        self.actfunc = actfunc
        # self.loss_function = loss_function
        if softmax * output_layer:
            self.weights = np.eye(n_prev_layer - 1, n_prev_layer - 1)
            # Add zero row to kll "Bias trick"
            self.weights = np.r_[self.weights, np.zeros((1, self.weights.shape[1]))]
            # print("We make a softmax layer")
            self.actfunc = "softmax"
        else:
            self.weights = np.random.rand(n_prev_layer, n_this_layer)
            self.actfunc = actfunc
            self.delta = np.zeros_like(self.weights)

    # TODO CLEAN
    def activation_function(self, z):
        if self.actfunc == "sigmoid":
            return sigmoid(z)
        elif self.actfunc == "ReLU":
            return relu(z)
        elif self.actfunc == "softmax":
            return softmax(z)
        elif self.actfunc == "identity":
            return identity(z)

    def d_activation_function(self, z):
        if self.actfunc == "sigmoid":
            return d_sigmoid(z)
        elif self.actfunc == "ReLU":
            return d_relu(z)
        elif self.actfunc == "softmax":
            return d_softmax(z)
        elif self.actfunc == "identity":
            return d_identity(z)

    def loss_fun(self, outputs, targets) -> float:
        if self.loss_function == "cross_entropy_loss":
            return cross_entropy_loss(outputs, targets)
        elif self.loss_function == "MSE":
            return mse(outputs, targets)
        elif self.loss_function == "difference":
            return difference(outputs, targets)
        else:
            print("ERROR no loss function")

    def d_loss_fun(self, outputs, targets):
        if self.loss_function == "cross_entropy_loss":
            return d_cross_entropy_loss(outputs, targets)
        elif self.loss_function == "MSE":
            return d_mse(outputs, targets)
        elif self.loss_function == "difference":
            return d_difference(outputs, targets)

    def update(self, learning_rate):
        if self.softmax * self.output_layer:
            # No update for softmax layer output layer
            pass
        else:
            grad = np.matmul(self.input.T, self.delta) / self.input.shape[0]
            self.weights -= learning_rate * grad

    def forward(
        self, X_batch: np.array, targets: np.array, prediction: bool = False
    ) -> np.array:
        """
        Args:
            X_batch: Activation of previous layer per batch_size
            [m=#batch, n=#neurons in layer]
            self.weights = [n,p] matrix where p = # neurons in next layer
        Returns:
            y: Activations of next layer
            [m,p]
        """

        self.input = X_batch
        self.z = np.matmul(self.input, self.weights)
        self.activations = self.activation_function(self.z)
        # print(self.actfunc)
        self.d_activations = self.d_activation_function(self.z)

        if prediction:
            return self.activations

        if self.output_layer:
            self.error = self.loss_fun(self.activations, targets)
            self.d_error = self.d_loss_fun(self.activations, targets)

        return self.activations

    def backward(self, prev_layer=None) -> np.array:
        """
        Computes the gradient and saves it to the variable self.delta
        """
        if self.output_layer:
            self.delta = self.d_error * self.d_activations
        else:
            self.delta = (
                np.matmul(prev_layer.delta, prev_layer.weights.T) * self.d_activations
            )
