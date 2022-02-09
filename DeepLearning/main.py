from network import Network
from datagenerator import DataGenerator
from utils import one_hot_encode
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np

parser = ConfigParser()

# Get parameters from input our config file
parser.read("configfile1.ini")
# Data parameters
dimension = parser.getint("datagenerator", "dimension")
size = parser.getint("datagenerator", "size")
noise = parser.getint("datagenerator", "noise")
center = parser.getboolean("datagenerator", "center")
training = parser.getfloat("datagenerator", "training")
validating = parser.getfloat("datagenerator", "validating")

# Networkout parameters
which_loss_function = parser.get("network", "which_loss_function")
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
    data = DataGenerator(
        dimension=dimension,
        size=size,
        noise=noise,
        center=center,
        training=training,
        validating=validating,
    )
    # use one_hot_encode to make 3 to [0,0,0,1] etc.
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
        f"Making network: loss_function = {which_loss_function}, regularization = {reg} with lamda = {reg_const}"
    )
    layer_counter = 0
    for k, v in networkshape.items():
        if v[0] == "Initialize_weights":
            init_dim = v[0]  # Store if we want to visualize the weights
            print(f"Input layer with {k} nodes.")
            continue
        layer_counter += 1
        if layer_counter == len(networkshape) - 1 and not softmax:
            print(
                f"Output layer, number of nodes = {k} activation function = {v[0]}, learning rate = {v[2]}, initial weights = {v[1]}."
            )
        else:
            print(
                f"Layer number {layer_counter}, number of nodes = {k} activation function = {v[0]}, learning rate = {v[2]}, initial weights = {v[1]}."
            )
        if layer_counter == len(networkshape) - 1 and softmax:
            print("Softmax output layer.")
    net = Network(
        networkshape=networkshape,
        softmax=softmax,
        which_loss_function=which_loss_function,
        reg_const=reg_const,
        reg=reg,
    )
    # Train it
    print(
        f"Training network: epochs = {epochs}, batch_size = {batch_size}, stochastic = {stochastic}, early_stop = {early_stop}"
    )
    loss_and_acc_history = net.train(
        epochs=epochs,
        batch_size=batch_size,
        stochastic=stochastic,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        early_stop=early_stop,
    )
    # Prints how an example of an output looks like
    print(
        "Example prediction",
        net.predict(
            X_val,
            Y_val,
            X_train,
            Y_train,
            X_test,
            Y_test,
            loss_and_acc_history=loss_and_acc_history,
        )[0],
    )
    # Count how many unique images there are in each class
    print("A count of how many shapes there are in each class in our data:")
    d = {0: "rectangle", 1: "cross", 2: "circles", 3: "triangles"}
    Y = data.Y
    unique, counts = np.unique(Y, return_counts=True)
    print(dict(zip(d.values(), counts)))
    # Plot some random images:
    NUMBER_OF_IMAGES = 10
    fig = plt.figure(figsize=(15, 10))
    randomints = np.random.randint(0, Y.shape[0], size=NUMBER_OF_IMAGES)
    for i in range(NUMBER_OF_IMAGES):
        index = randomints[i]
        title = d[int(Y[index])]
        matrix = data.Xmatrix[index]
        fig.add_subplot(2, 5, i + 1)
        plt.title(title)
        plt.imshow(matrix, cmap=plt.cm.gray)
    plt.show(block=True)
    """
    # Bonus if you want to print weights, only works for two (one) layer network
    net.getweights()
    gg = net.endweights[0]
    ggt = gg[:-1, 1]  # Second index represents which shape you want
    ggt = ggt * (1 / np.max(ggt))  # Normalize, did not make a huge difference
    ggt = np.reshape(ggt, (init_dim, init_dim))
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(ggt, cmap=plt.cm.gray)
    plt.show()
    """
