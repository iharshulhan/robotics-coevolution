import numpy as np

from initial_weights import weights_dense1_w, weights_dense2_w, weights_dense1_b, weights_dense2_b, weights_final_w, \
    weights_final_b


class SmallReactivePolicy:
    """Simple multi-layer perceptron policy, no internal state"""

    def __init__(self):
        self.weights_dense1_w = weights_dense1_w.copy()
        self.weights_dense2_w = weights_dense2_w.copy()
        self.weights_dense1_b = weights_dense1_b.copy()
        self.weights_dense2_b = weights_dense2_b.copy()
        self.weights_final_w = weights_final_w.copy()
        self.weights_final_b = weights_final_b.copy()

    def act(self, ob):
        x = ob
        x = relu(np.dot(x, self.weights_dense1_w) + self.weights_dense1_b)
        x = relu(np.dot(x, self.weights_dense2_w) + self.weights_dense2_b)
        x = np.dot(x, self.weights_final_w) + self.weights_final_b
        return x


def relu(x):
    return np.maximum(x, 0)
