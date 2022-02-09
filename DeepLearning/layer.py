import numpy as np
from utils import (
    sigmoid,
    d_sigmoid,
    d_mse,
    d_cross_entropy_loss,
    relu,
    d_relu,
    softmax,
    d_softmax,
    identity,
    d_identity,
    tanh,
    d_tanh,
)


class Layer:
    def __init__(
        self,
        reg_const: float,
        reg: str,
        which_loss_function: str,
        n_prev_layer: int,
        n_this_layer: int,
        learning_rate: float = 0,
        which_activation_function: str = "sigmoid",
        output_layer: bool = False,
        softmax: bool = False,
        weight_range=(0, 1),
    ) -> None:
        self.learning_rate = learning_rate
        self.reg_const = reg_const
        self.reg = reg
        self.softmax = softmax
        self.which_loss_function = which_loss_function
        self.output_layer = output_layer
        self.activations = None
        self.d_activations = None
        self.z = None
        self.error = None
        self.d_error = None
        self.input = None
        self.which_activation_function = which_activation_function
        # self.loss_function = loss_function
        if softmax * output_layer:
            self.weights = np.eye(n_prev_layer - 1, n_prev_layer - 1)
            # Add zero row to kll "Bias trick"
            self.weights = np.r_[self.weights, np.zeros((1, self.weights.shape[1]))]
            self.which_activation_function = "softmax"
        else:
            self.weights = np.random.uniform(
                weight_range[0], weight_range[1], (n_prev_layer, n_this_layer)
            )
            self.which_activation_function = which_activation_function
            self.delta = np.zeros_like(self.weights)

    def activation_function(self, z: np.array):
        """
        z = Xw
        Arg: z [m=#batch, n=# neurons in output]
        Ret: Activation function applied to z
        """
        if self.which_activation_function == "sigmoid":
            return sigmoid(z)
        elif self.which_activation_function == "ReLU":
            return relu(z)
        elif self.which_activation_function == "softmax":
            return softmax(z)
        elif self.which_activation_function == "identity":
            return identity(z)
        elif self.which_activation_function == "tanh":
            return tanh(z)

    def d_activation_function(self, z):
        """
        Arg: z [m=#batch, n=# neurons in output]
        Ret: Derivative of the activation function applied to z
        """
        if self.which_activation_function == "sigmoid":
            return d_sigmoid(z)
        elif self.which_activation_function == "ReLU":
            return d_relu(z)
        elif self.which_activation_function == "softmax":
            return d_softmax(z)
        elif self.which_activation_function == "identity":
            return d_identity(z)
        elif self.which_activation_function == "tanh":
            return d_tanh(z)

    def d_loss_function(self, outputs, targets):
        """
        Arg: outputs [m=#batch, n=# neurons in output]
             targets [m=#batch, n=# neurons in output]
        Ret: Derivative of the loss function, w.r.t outputs, applied to outputs and targets
        """
        if self.which_loss_function == "cross_entropy_loss":
            return d_cross_entropy_loss(outputs, targets)
        elif self.which_loss_function == "MSE":
            return d_mse(outputs, targets)

    def update(self):
        """
        Updates weights in the layer as in gradient descent
        """
        if self.softmax * self.output_layer:
            # No update for the softmax layer output layer
            pass
        else:
            # regularization
            dr = 0
            if self.reg_const != 0:
                if self.reg == "L1":
                    temp = self.weights.copy()
                    temp[temp > 0] = 1
                    temp[temp < 0] = -1
                    dr = self.reg_const * temp
                elif self.reg == "L2":
                    dr = self.reg_const * self.weights
            # Update weights
            grad = (
                np.matmul(self.input.T, self.delta) / self.input.shape[0]
                + dr / self.input.shape[0]
            )
            self.weights -= self.learning_rate * grad

    def forward(
        self, X_batch: np.array, targets: np.array, prediction: bool = False
    ) -> np.array:
        """
        Preforms the forward, i.e, a(xw), we also calculate and store the derivative of the
        activation function applied to z, for gradient descent. And the derivative of the loss function
        applied to activations for the output layer, for gradient descent. If we forward an X_batch from test or validating set,
        then prediction = True, and we dont do this.
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
        if prediction:
            return self.activations

        self.d_activations = self.d_activation_function(self.z)

        if self.output_layer:
            self.d_error = self.d_loss_function(self.activations, targets)

        return self.activations

    def backward(self, prev_layer=None) -> np.array:
        """
        #arg: Layer, previous layer
        Computes the (local) gradient and saves it to the variable self.delta
        gradient of a none outputs is used to calculate the current gradient via the chain rule.
        """
        if self.output_layer:
            self.delta = self.d_error * self.d_activations
        else:
            self.delta = (
                np.matmul(prev_layer.delta, prev_layer.weights.T) * self.d_activations
            )
