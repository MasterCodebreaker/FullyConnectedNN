from network import Network
from datagenerator import DataGenerator
from utils import one_hot_encode
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np

parser = ConfigParser()

# Dimension of picture

# Set parameters
parser.read("globals.ini")
# Data parameters
dimension = parser.getint("datagenerator", "dimension")
size = parser.getint("datagenerator", "size")
noise = parser.getint("datagenerator", "noise")
center = parser.getboolean("datagenerator", "center")
# Network parameters
loss_function = parser.get("network", "loss_function")
learning_rate = parser.getfloat("network", "learning_rate")
softmax = parser.getboolean("network", "softmax")
actfunc = parser.get("network", "actfunc")
networkshape = eval(parser.get("network", "networkshape"))
# Training variables
epochs = parser.getint("training", "epochs")
batch_size = parser.getint("training", "batch_size")
stochastic = parser.getboolean("training", "stochastic")

print(loss_function)

if __name__ == "__main__":
    data = DataGenerator(dimension=dimension, size=size, noise=noise, center=center)
    data.generate()
    X_train = data.X_train
    Y_train = one_hot_encode(data.Y_train, 4)
    X_val = data.X_val
    Y_val = one_hot_encode(data.Y_val, 4)
    X_test = data.X_test
    Y_test = one_hot_encode(data.Y_test, 4)
    print(
        f"Generating Data\nTraining data X = {X_train.shape}, Y = {Y_train.shape} \nValidating data X = {X_val.shape}, Y = {Y_val.shape}."
    )
    print(
        f"Making network: networkshape = {networkshape}, softmax = {softmax}, actfunc = {actfunc}, loss_function = {loss_function}."
    )
    net = Network(
        networkshape=networkshape,
        softmax=softmax,
        actfunc=actfunc,
        loss_function=loss_function,
    )
    # Train it
    print(
        f"Training network: learning_rate = {learning_rate}, epochs = {epochs}, batch_size = {batch_size}, stochastic = {stochastic}"
    )
    net.train(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        stochastic=stochastic,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
    )
    print(
        "Example prediction",
        net.predict(X_val, Y_val, X_train, Y_train, X_test, Y_test)[0],
    )
    # Plot some random images:
    d = {0: "rectangle", 1: "cross", 2: "circle", 3: "triangle"}
    Y = data.Y
    unique, counts = np.unique(Y, return_counts=True)
    print(dict(zip(d.values(), counts)))
    # Plotter
    NUMBER_OF_INTS = 10
    fig = plt.figure(figsize=(15, 10))
    randomints = np.random.randint(0, Y.shape[0], size=NUMBER_OF_INTS)
    for i in range(NUMBER_OF_INTS):
        index = randomints[i]
        title = d[int(Y[index])]
        matrix = data.Xmatrix[index]
        fig.add_subplot(2, 5, i + 1)
        plt.title(title)
        plt.imshow(matrix, cmap=plt.cm.gray)
    plt.show(block=True)