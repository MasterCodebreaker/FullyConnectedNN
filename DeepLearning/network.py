import numpy as np
import matplotlib.pyplot as plt

from layer import Layer
from utils import (
    mse,
    cross_entropy_loss,
    count_correct_pred,
    batch_loader,
)


class Network:
    def __init__(
        self,
        networkshape: dict,
        reg_const: float = 0,
        reg: str = "L1",
        softmax: bool = False,
        which_loss_function: str = "cross_entropy_loss",
    ) -> None:
        self.reg_const = reg_const
        self.reg = reg
        self.which_loss_function = which_loss_function
        self.networkshape = networkshape
        self.softmax = softmax
        self.layers = []
        # Just to store end weights for fun if you want to visualize them
        self.endweights = []

        # Make list of layers
        # Counter just to keep track of output layer
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
                n_this_layer = k
            else:
                output_layer = False
                n_this_layer = k + 1  # Bias trick
            # Make layer object
            layer = Layer(
                learning_rate=v[2],
                softmax=self.softmax,
                n_prev_layer=n_prev_layer,
                n_this_layer=n_this_layer,
                output_layer=output_layer,
                which_loss_function=self.which_loss_function,
                which_activation_function=v[0],
                weight_range=v[1],
                reg_const=self.reg_const,
                reg=self.reg,
            )
            # Add it to network
            self.layers.append(layer)
            n_prev_layer = k + 1  # Bias tick
            counter += 1
        if self.softmax:
            # Make another layer for softmax and add it to network
            output_layer = True
            layer = Layer(
                softmax=self.softmax,
                n_prev_layer=n_prev_layer,
                n_this_layer=n_prev_layer,
                output_layer=output_layer,
                which_loss_function=self.which_loss_function,
                reg_const=self.reg_const,
                reg=self.reg,
            )
            self.layers.append(layer)

    def loss_function(self, outputs: np.array, targets: np.array) -> float:
        """
        Arg : outputs and targets have the same dimension: [batch_size, output_layer_dim]
        Ret: loss
        """
        # r is the regularization constant
        r = 0
        # Check if we use regularization
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
        # Output the correct loss function
        if self.which_loss_function == "cross_entropy_loss":
            return cross_entropy_loss(outputs, targets) + r
        elif self.which_loss_function == "MSE":
            return mse(outputs, targets) + r

    def forward(
        self, X_batch: np.array, targets: np.array, prediction: bool = False
    ) -> None:
        """
        Arg :   X_batch: [m=#batch, n=#neurons in first layer]
                targets: [m=#batch, p=#neurons in outputlayer]
                prediction: Bool if we use forward to calculate prediction, we can skip a calculation
        Ret: activations of last layer [m=#batch, p=#neurons in last layer]
        """
        # takes input X_batch and pushes it through the layers
        activation = X_batch
        for m in self.layers:
            activation = m.forward(activation, targets, prediction)
        return activation

    def backward(self) -> None:
        """
        Calculate the derivative of the loss function w.r.t the weights.
        Starts with the output layer and goes backward, towards the input layer taking use of the chain rule
        """
        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i].output_layer:
                self.layers[i].backward()
            else:
                self.layers[i].backward(self.layers[i + 1])

    def validation_step(self, X_val: np.array, Y_val: np.array) -> (float, np.array):
        """
        Calculates loss for validation set, also returns outputs to be used for calculating the accuracy.
        Arg: X_val: [m=#batch, n=#neurons in first layer]
             Y_val: [m=#batch, p=#neurons in outputlayer]
        Ret: (float, outputs: [m=#batch, p=#neurons in outputlayer])
        """
        X_val = np.c_[X_val, np.ones((X_val.shape[0], 1))]
        outputs = self.forward(X_val, Y_val, prediction=True)
        return (self.loss_function(outputs, Y_val), outputs)

    def train_step(self, X_train, Y_train) -> (float, np.array):
        """
        Preforms one step of gradient descent:forward and backward and update weights in every layer.
        We also return loss and outputs to be used for calculating the accuracy.
        Arg: X_train: [m=#batch, n=#neurons in first layer]
             Y_train: [m=#batch, p=#neurons in outputlayer]
        Ret: (float, outputs: [m=#batch, p=#neurons in outputlayer])
        """
        outputs = self.forward(X_train, Y_train)
        self.backward()
        for layer in self.layers:
            layer.update()
        return (self.loss_function(outputs, Y_train), outputs)

    def train(
        self,
        epochs: int,
        batch_size: int,
        stochastic: bool,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        early_stop: int = False,
    ) -> tuple:
        """
        Loops through the number of epochs, and every mini-batch in every epoch
        we preform gradient descent for the training set. Moreover we record and return
        the loss and accuracy history for the training and validation data.
        Arg: X,Y train, X,Y val, epochs = #of epoch, batch_size = size of every mini-batch
        stochastic: If we want stochastic gradient descent.
        Early stop (In experimental phase): if we want to to an early stop if loss of validation increases early_stop # of times.
        """

        loss_val = []
        acc_val = []
        loss_train = []
        acc_train = []
        Y_batches = []
        # Init a high value
        min_val = 1000
        counter = early_stop
        for i in range(epochs):
            if i % 15 == 0:
                print(f"We are at epoch {i}")
            # select minibatch
            for X_batch, Y_batch in iter(
                batch_loader(X_train, Y_train, batch_size, stochastic=stochastic)
            ):  # Preform train step
                (l_t, a_t) = self.train_step(
                    np.c_[X_batch, np.ones((X_batch.shape[0], 1))], Y_batch,
                )
                # Record train history
                loss_train.append(l_t)
                acc_train.append(count_correct_pred(a_t, Y_batch) / Y_batch.shape[0])
            # Record validation history
            (l_v, a_v) = self.validation_step(X_val, Y_val)
            acc_val.append(count_correct_pred(a_v, Y_val) / Y_val.shape[0])
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
        # Make into np.arrays
        loss_val = np.array(loss_val)
        loss_train = np.array(loss_train)
        acc_val = np.array(acc_val)
        acc_train = np.array(acc_train)
        # Return history
        return (
            (
                np.arange(1, loss_val.shape[0] + 1, 1)
                * (X_train.shape[0] // batch_size),
                loss_val,
            ),
            (np.arange(0, loss_train.shape[0], 1), loss_train),
            (
                np.arange(1, acc_val.shape[0] + 1, 1)
                * (X_train.shape[0] // batch_size),
                acc_val,
            ),
            (np.arange(0, acc_train.shape[0], 1), acc_train),
        )

    def getweights(self):
        """
        Puts weights into self.endweights
        """
        for layer in self.layers:
            self.endweights.append(layer.weights)

    def predict(
        self,
        X_val: np.array,
        Y_val: np.array,
        X_train: np.array,
        Y_train: np.array,
        X_test: np.array,
        Y_test: np.array,
        loss_and_acc_history: tuple,
    ) -> np.array:
        """
        We calculate accuracy for the test, train and validation set and makes plots for accuracy and loss
        Arg: X,Y val, X,Y train, X,Y test, loss and accuracy history for training and validation
        Ret: Outputs for the test data

        """
        # Score val
        (loss_val, loss_train, acc_val, acc_train) = loss_and_acc_history

        dict = {
            "training": (X_train, Y_train),  #
            "validation": (X_val, Y_val),
            "testing": (X_test, Y_test),
        }
        for k, v in dict.items():
            # Bias trick
            X = np.c_[v[0], np.ones((v[0].shape[0], 1))]
            # Compute outputs
            outputs = self.forward(X, v[1], prediction=True)
            # Count how many predictions that where correct
            count = count_correct_pred(outputs, v[1])
            if k == "testing":
                # We need to additionally compute accuracy and loss for the testing set,
                # as we have not done so before.
                test_acc_score = np.round(count / v[1].shape[0], 2)
                test_loss_score = self.loss_function(outputs, v[1])
            print(
                f"We have {count} out of {v[1].shape[0]} correct prediction for {k}, that is, {np.round(count/v[1].shape[0] *100,2)}% accurate."
            )
        # To plot test acc/val at the right spot
        max_index = acc_train[0][-1]
        steps = np.int(acc_train[0].shape[0] / 15)  # Lenght of test graph
        index_vector = np.arange(max_index, max_index + steps, 1)

        # Plot
        fig, axs = plt.subplots(2)
        axs[0].plot(
            loss_train[0], loss_train[1], color="b", label="Training",
        )
        axs[0].plot(
            loss_val[0], loss_val[1], color="r", label="Validating",
        )
        axs[0].plot(
            index_vector,
            np.array([test_loss_score] * steps),
            color="g",
            label="Testing",
        )
        axs[0].legend(loc="best")

        # Label for axis
        axs.flat[0].set(xlabel="Mini-batches", ylabel="Loss")
        axs.flat[1].set(xlabel="Mini-batches", ylabel="Accuracy")

        axs[1].plot(
            acc_train[0], acc_train[1], color="b", label="Training",
        )
        axs[1].plot(
            acc_val[0], acc_val[1], color="r", label="Validating",
        )

        axs[1].plot(
            index_vector,
            np.array([test_acc_score] * steps),
            color="g",
            label="Testing",
        )
        axs[1].legend(loc="best")
        fig.tight_layout()
        plt.show()

        return outputs.round(1)
