from configparser import ConfigParser

"""
Make your own config file.
"""
config = ConfigParser()
N = 30
GLOBAL_LEARNING_RATE = 10
# DataGenerator variables
config["datagenerator"] = {
    "noise": 0,
    "size": 2000,
    "dimension": N,
    "center": True,
    "training": 0.7,
    "validating": 0.2,
}
# Network variables
"""
Possible loss functions are:
"MSE" --  "cross_entropy_loss"
Possible activation functions are:
"ReLU"  -- "sigmoid" -- "identity" aka linear -- "tanh"  --  "softmax"
Possible regularization are:
"L1" -- "L2"
"none" To use none or set "reg_const" to 0.
"""
config["network"] = {
    "which_loss_function": "cross_entropy_loss",
    "softmax": True,
    "networkshape": {
        N ** 2: ("Initialize_weights", "learning_rate"),  # Input layer
        64: ("ReLU", (-1, 1), GLOBAL_LEARNING_RATE * 100),
        32: ("tanh", (-0.8, 0.8), GLOBAL_LEARNING_RATE * 10),
        16: ("ReLU", (-0.8, 0.8), GLOBAL_LEARNING_RATE * 10),
        10: ("ReLU", (-0.8, 0.8), GLOBAL_LEARNING_RATE),
        8: ("sigmoid", (-0.8, 0.8), GLOBAL_LEARNING_RATE),
        4: ("sigmoid", (-0.8, 0.8), GLOBAL_LEARNING_RATE),  # Output layer
    },
    "reg": "L2",
    "reg_const": 0.0001,
}
# Training variables
config["training"] = {
    "epochs": 60,
    "batch_size": 12,
    "stochastic": True,
    # 0 for no early stop, otherwise specify epoch
    "early_stop": 0,
}
with open("./configfile4.ini", "w") as f:
    config.write(f)
