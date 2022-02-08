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
        reg_const: float = 0,
        reg: str = "L1",
        softmax: bool = False,
        loss_function: str = "cross_entropy_loss",
    ) -> None:

        self.reg_const = reg_const
        self.reg = reg
        self.end_weights = None
        self.loss_function = loss_function
        self.networkshape = networkshape
        self.softmax = softmax
        self.layers = []
        self.loss_val_history = []
        # Just to store end weights for fun
        self.endweights = []

        # Make list of layers
        counter = 0
        for k, v in self.networkshape.items():
            if counter == 0:
                n_prev_layer = k + 1  # Bias trick
                counter += 1
                # We dont want to make the first layer
                continue
            if (counter == len(networkshape) - 1) and (not softmax):
                # Then this is the output layer
                output_layer = True
                print("Output, no softmax")
                n_this_layer = k
            else:
                output_layer = False
                n_this_layer = k + 1  # Bias trick
            layer = Layer(
                softmax=self.softmax,
                n_prev_layer=n_prev_layer,
                n_this_layer=n_this_layer,
                output_layer=output_layer,
                loss_function=self.loss_function,
                actfunc=v[0],
                weight_range=v[1],
                reg_const=self.reg_const,
                reg=self.reg,
            )
            self.layers.append(layer)
            n_prev_layer = k + 1
            counter += 1
        if self.softmax:
            # Make another layer for softmax
            output_layer = True
            layer = Layer(
                softmax=self.softmax,
                n_prev_layer=n_prev_layer,
                n_this_layer=n_prev_layer,
                output_layer=output_layer,
                loss_function=self.loss_function,
                reg_const=self.reg_const,
                reg=self.reg,
            )
            self.layers.append(layer)

    # Loss loss_function
    def loss_fun(self, outputs, targets) -> float:
        r = 0
        if self.reg_const != 0:
            if self.reg == "L1":
                for layer in self.layers:
                    r += layer.reg_const * np.einsum("ij ->", np.abs(layer.weights))
            elif self.reg == "L2":
                for layer in self.layers:
                    r += (
                        layer.reg_const
                        * np.einsum("ij ->", (1 / 2) * layer.weights ** 2)
                        * (1 / 2)
                    )

        if self.loss_function == "cross_entropy_loss":
            return cross_entropy_loss(outputs, targets) + r
        elif self.loss_function == "MSE":
            return mse(outputs, targets) + r
        elif self.loss_function == "difference":
            return difference(outputs, targets) + r

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
        return (self.loss_fun(outputs, Y_val), outputs)

    def train_step(self, X_train, Y_train, learning_rate):
        """
        forward and backward and uptdate
        """
        outputs = self.forward(X_train, Y_train)
        self.backward()
        for layer in self.layers:
            layer.update(learning_rate)
        return (self.loss_fun(outputs, Y_train), outputs)

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
        early_stop: int = False,
    ) -> None:
        loss_val = []
        acc_val = []
        loss_train = []
        acc_train = []
        Y_batches = []
        # Init a low value
        min_val = 1000
        counter = early_stop
        for i in range(epochs):
            # select minibatch
            for X_batch, Y_batch in iter(
                batch_loader(X_train, Y_train, batch_size, stochastic=stochastic)
            ):
                (l_t, a_t) = self.train_step(
                    np.c_[X_batch, np.ones((X_batch.shape[0], 1))],
                    Y_batch,
                    learning_rate,
                )
                loss_train.append(l_t)
                # Make into a function
                int_pred = np.zeros_like(a_t)
                int_pred[np.arange(a_t.shape[0]), a_t.argmax(1)] = 1
                count = np.sum((int_pred == Y_batch).all(1))
                acc_train.append(count / Y_batch.shape[0])

            (l_v, a_v) = self.validation_step(X_val, Y_val)
            loss_val.append(l_v)
            # Early stopping
            if early_stop:
                if l_v > min_val:
                    counter -= 1
                else:
                    min_val = l_v

                if counter == 0:
                    print(
                        "We did an early stop at epoch ",
                        i,
                        "with loss = ",
                        np.round(l_v, 2),
                        ".",
                    )
                    break

            # self.loss_val_history.append(l_v)
            # Make into a function TODO
            int_pred = np.zeros_like(a_v)
            int_pred[np.arange(a_v.shape[0]), a_v.argmax(1)] = 1
            count = np.sum((int_pred == Y_val).all(1))
            acc_val.append(count / Y_val.shape[0])
            ################################
        ###Plotting loss, accuracy for val and train###
        fig, axs = plt.subplots(2)
        loss_val = np.array(loss_val)
        loss_train = np.array(loss_train)
        axs[0].plot(
            np.arange(0, loss_train.shape[0], 1),
            np.array(loss_train),
            color="b",
            label="Training",
        )
        axs[0].plot(
            np.arange(0, loss_val.shape[0], 1) * (X_train.shape[0] // batch_size),
            loss_val,
            color="r",
            label="Validating",
        )
        axs[0].legend(loc="best")

        acc_val = np.array(acc_val)
        acc_train = np.array(acc_train)
        # Label for axis
        axs.flat[0].set(xlabel="Mini-batches", ylabel="Loss")
        axs.flat[1].set(xlabel="Mini-batches", ylabel="Accuracy")

        axs[1].plot(
            np.arange(0, acc_train.shape[0], 1), acc_train, color="b", label="Training",
        )
        axs[1].plot(
            np.arange(0, acc_val.shape[0], 1) * (X_train.shape[0] // batch_size),
            acc_val,
            color="r",
            label="Validating",
        )
        axs[1].legend(loc="best")
        fig.tight_layout()
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
            f"We have {count} out of {Y_val.shape[0]} correct prediction for val, that is, {np.round(count/Y_val.shape[0] *100,2)}% accurate."
        )

        # Score training
        X_train = np.c_[X_train, np.ones((X_train.shape[0], 1))]
        outputs = self.forward(X_train, Y_train, prediction=True)
        int_pred = np.zeros_like(outputs)
        int_pred[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1
        count = np.sum((int_pred == Y_train).all(1))
        print(
            f"We have {count} out of {Y_train.shape[0]} correct prediction for train, that is, {np.round(count/Y_train.shape[0] *100,2)}% accurate."
        )

        # Score training
        X_test = np.c_[X_test, np.ones((X_test.shape[0], 1))]
        outputs = self.forward(X_test, Y_test, prediction=True)
        int_pred = np.zeros_like(outputs)
        int_pred[np.arange(outputs.shape[0]), outputs.argmax(1)] = 1
        count = np.sum((int_pred == Y_test).all(1))
        print(
            f"We have {count} out of {Y_test.shape[0]} correct prediction for X_test, that is, {np.round(count/Y_test.shape[0] *100,2)}% accurate."
        )

        return outputs.round(1)
