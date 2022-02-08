from network import Network
from datagenerator import DataGenerator
from utils import one_hot_encode
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np

parser = ConfigParser()

# Dimension of picture

# Set parameters input your config file
parser.read("tryforearly.ini")
# Data parameters
dimension = parser.getint("datagenerator", "dimension")
size = parser.getint("datagenerator", "size")
noise = parser.getint("datagenerator", "noise")
center = parser.getboolean("datagenerator", "center")
# Networkout parameters
loss_function = parser.get("network", "loss_function")
learning_rate = parser.getfloat("network", "learning_rate")
softmax = parser.getboolean("network", "softmax")
networkshape = eval(parser.get("network", "networkshape"))
reg = parser.get("network", "reg")
reg_const = parser.getfloat("network", "reg_const")
# Training variables
epochs = parser.getint("training", "epochs")
batch_size = parser.getint("training", "batch_size")
stochastic = parser.getboolean("training", "stochastic")
early_stop = parser.getint("training", "early_stop")

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
        f"Making network: networkshape = {networkshape}\nsoftmax = {softmax}, loss_function = {loss_function}, regularization = {reg} with lamda = {reg_const}."
    )
    net = Network(
        networkshape=networkshape,
        softmax=softmax,
        loss_function=loss_function,
        reg_const=reg_const,
        reg=reg,
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
        early_stop=early_stop,
    )
    print(
        "Example prediction",
        net.predict(X_val, Y_val, X_train, Y_train, X_test, Y_test)[0],
    )
    # Plot some random images:
    d = {0: "# rectangles", 1: "# cross", 2: "# circles", 3: "# triangles"}
    Y = data.Y
    unique, counts = np.unique(Y, return_counts=True)
    print(dict(zip(d.values(), counts)))
    # Plotter
    """
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
    """
