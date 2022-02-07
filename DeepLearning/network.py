import numpy as np

from layer import Layer

# from trainer import Trainer
from datagenerator import DataGenerator
from utils import (
    sigmoid,
    d_sigmoid,
    mse,
    d_mse,
    batch_loader,
    one_hot_encode,
    d_difference,
    difference,
    cross_entropy_loss,
    d_cross_entropy_loss,
)

# PLotting for Testing
import matplotlib.pyplot as plt

# np.random.seed(0)


class Network:
    def __init__(
        self,
        networkshape: list,
        softmax: bool = False,
        actfunc: str = "sigmoid",
        loss_function: str = "cross_entropy_loss",
    ) -> None:

        # Probably move down
        nhiddenl = len(networkshape) - 1
        inputdim = networkshape[0]
        outputdim = networkshape[-1]
        """
        assert (
            nhiddenl <= 4 and nhiddenl > -1
        ), f"Number of hidden layersmust be between 0 and 5."

        assert (
            inputdim <= 50 * 50 and inputdim >= 4
        ), f"The dimension of input vector must be between 10 and 50. "
        """
        self.end_weights = None
        self.loss_function = loss_function
        self.networkshape = networkshape
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.nhiddenl = nhiddenl
        self.softmax = softmax
        self.layers = []
        self.actfunc = actfunc
        # Just to store end weights for fun
        self.endweights = []

        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # Make list of layers
        for i in range(1, len(self.networkshape) + self.softmax):
            # print(f"making layer {i}")
            output_layer = False
            if i == len(self.networkshape) + self.softmax - 1:
                output_layer = True
            n_prev_layer = self.networkshape[i - 1] + 1
            n_this_layer = (
                self.networkshape[i - self.softmax * output_layer] + 1 - output_layer
            )  # Bias trick
            layer = Layer(
                softmax=self.softmax,
                n_prev_layer=n_prev_layer,
                n_this_layer=n_this_layer,
                output_layer=output_layer,
                loss_function=self.loss_function,
                actfunc=self.actfunc,
            )
            self.layers.append(layer)

    # Loss loss_function
    def loss_fun(self, outputs, targets) -> float:
        if self.loss_function == "cross_entropy_loss":
            return cross_entropy_loss(outputs, targets)
        elif self.loss_function == "MSE":
            return mse(outputs, targets)
        elif self.loss_function == "difference":
            return difference(outputs, targets)

    def d_loss_fun(self, outputs, targets):
        if self.loss_function == "cross_entropy_loss":
            return d_cross_entropy_loss(outputs, targets)
        elif self.loss_function == "MSE":
            return d_mse(outputs, targets)
        elif self.loss_function == "difference":
            return d_difference(outputs, targets)

    def forward(
        self, X_batch: np.array, targets: np.array, prediction: bool = False
    ) -> None:
        """
        X_batch: Activation of previous layer per batch_size
        [m=#batch, n=#neurons in layer]
        return activations of last layer
        """
        # takes input X_batch and pushes it through the layers
        activation = X_batch
        for m in self.layers:
            # bias trick
            activation = m.forward(activation, targets)
        # print(activation)
        return activation

    # Assume forward is use before backward
    def backward(self):
        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i].output_layer:
                self.layers[i].backward()
            else:
                self.layers[i].backward(self.layers[i + 1])

    def validation_step(self, X_val, Y_val):
        X_val = np.c_[X_val, np.ones((X_val.shape[0], 1))]
        outputs = self.forward(X_val, Y_val, prediction=True)

        return self.loss_fun(outputs, Y_val)

    def train_step(self, X_train, Y_train, learning_rate):
        """
        forward and backward and uptdate
        """
        outputs = self.forward(X_train, Y_train)
        self.backward()
        for layer in self.layers:
            layer.update(learning_rate)
        return self.loss_fun(outputs, Y_train)

    def train(
        self,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        stochastic: bool,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
    ) -> None:
        loss_val = []
        loss_train = []
        for i in range(epochs):
            # select minibatch
            for X_batch, Y_batch in iter(
                batch_loader(X_train, Y_train, batch_size, stochastic=stochastic)
            ):
                # print(X_batch.shape, "XshapeY", Y_batch.shape)
                loss_train.append(
                    self.train_step(
                        np.c_[X_batch, np.ones((X_batch.shape[0], 1))],
                        Y_batch,
                        learning_rate,
                    )
                )
                loss_val.append(self.validation_step(X_val, Y_val))
        # print(loss)
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot()
        plt.xlabel("Number of Training Steps")
        plt.ylabel("Loss")
        ax = plt.gca()  # Get Current Axis
        # For better Zoom
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([0, 5])

        plt.plot(np.array(loss_train), color="b", label="Training")
        plt.plot(np.array(loss_val), color="r", label="Validating")
        plt.legend(loc="best")
        plt.show()

    def getweights(self):
        for layer in self.layers:
            self.endweights.append(layer.weights)

    def predict(
        self, X_val: np.array, Y_val, X_train, Y_train, X_test, Y_test
    ) -> np.array:
        # Bias trick
        # Score val
        X_val = np.c_[X_val, np.ones((X_val.shape[0], 1))]
        outputs = self.forward(X_val, Y_val, prediction=True)
        int_pred = np.zeros_like(outputs)
        int_pred[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1
        count = np.sum((int_pred == Y_val).all(1))
        print(
            f"We have {count} out of {Y_val.shape[0]} correct prediction for val, that is {np.round(count/Y_val.shape[0] *100,2)}% accurate."
        )

        # Score training
        X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))]
        outputs = self.forward(X_train, Y_train, prediction=True)
        int_pred = np.zeros_like(outputs)
        int_pred[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1
        count = np.sum((int_pred == Y_train).all(1))
        print(
            f"We have {count} out of {Y_train.shape[0]} correct prediction for train, that is {np.round(count/Y_train.shape[0] *100,2)}% accurate."
        )

        # Score training
        X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
        outputs = self.forward(X_test, Y_test, prediction=True)
        int_pred = np.zeros_like(outputs)
        int_pred[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1
        count = np.sum((int_pred == Y_test).all(1))
        print(
            f"We have {count} out of {Y_test.shape[0]} correct prediction for X_test, that is {np.round(count/Y_test.shape[0] *100,2)}% accurate."
        )

        return outputs.round(1)
