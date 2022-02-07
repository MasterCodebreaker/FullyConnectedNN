from configparser import ConfigParser

config = ConfigParser()
N = 50
# DataGenerator variables
config["datagenerator"] = {"noise": 0, "size": 500, "dimension": N, "center": True}
# Network variables
config["network"] = {
    "loss_function": "cross_entropy_loss",
    "learning_rate": 100,
    "softmax": True,
    "networkshape": [N ** 2, 8, 4],
    "actfunc": "sigmoid",
}
# Training variables
config["training"] = {"epochs": 100, "batch_size": 10, "stochastic": True}
with open("./globals.ini", "w") as f:
    config.write(f)
