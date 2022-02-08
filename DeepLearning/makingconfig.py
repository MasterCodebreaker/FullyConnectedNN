from configparser import ConfigParser

config = ConfigParser()
N = 30
# DataGenerator variables
config["datagenerator"] = {"noise": 10, "size": 5000, "dimension": N, "center": False}
# Network variables
config["network"] = {
    "loss_function": "cross_entropy_loss",
    "learning_rate": 100,
    "softmax": True,
    "networkshape": {
        N ** 2: "No input",
        50: ("ReLU", (-1, 1)),
        32: ("sigmoid", (-0.5, 0.5)),
        16: ("ReLU", (-1, 1)),
        8: ("ReLU", (-1, 1)),
        4: ("sigmoid", (-0.5, 0.5)),
    },
    "reg": "L2",
    "reg_const": 0.001,
}
# Training variables
config["training"] = {
    "epochs": 100,
    "batch_size": 10,
    "stochastic": True,
    "early_stop": 30,
}
with open("./tryforearly.ini", "w") as f:
    config.write(f)
