from configparser import ConfigParser

parser = ConfigParser()
n = 10
parser.read("globals.ini")

# size_of_picture

loss_function = parser.get("loss_function", "loss_function")
learning_rate = parser.get("learning_rate", "learning_rate")
epochs = parser.get("epochs", "epochs")
batch_size = parser.get("batch_size", "batch_size")
stochastic = parser.get("stochastic", "stochastic")
softmax = parser.get("softmax", "softmax")
actfunc = parser.get("actfunc", "actfunc")
networkshape = parser.get("networkshape", "networkshape")


# "cross_entropy_loss" "MSE" "difference"
loss_function = "MSE"
# make network
learning_rate = 1
epochs = 80
batch_size = 10
stochastic = True
softmax = True
networkshape = [n ** 2, 4]
# "sigmoid" "ReLU" "identity"
actfunc = "ReLU"
