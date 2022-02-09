import numpy as np
import random
from itertools import combinations
from utils import one_hot_encode

# To plot:
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(
        self,
        dimension: int,
        size: int,
        noise: int,
        center: bool = True,
        training: float = 0.7,
        validating: float = 0.2,
    ) -> None:
        self.training = int(training * size)
        self.validating = int(validating * size)
        self.testing = size - self.training - self.validating
        self.noise = noise  # Number of noisepoints
        self.center = center
        self.dim = dimension
        self.size = size
        # All
        self.X = np.zeros((size, dimension ** 2))
        self.Xmatrix = np.zeros((size, dimension, dimension))
        self.Y = np.zeros((size, 1))
        # Train
        self.X_train = self.X[0 : self.training]
        self.Xmatrix_train = self.Xmatrix[0 : self.training]
        self.Y_train = self.Y[0 : self.training]
        # Validating
        self.X_val = self.X[self.training : self.training + self.validating]
        self.Xmatrix_val = self.Xmatrix[self.training : self.training + self.validating]
        self.Y_val = self.Y[self.training : self.training + self.validating]
        # Testing
        self.X_test = self.X[
            self.training
            + self.validating : self.training
            + self.validating
            + self.testing
        ]
        self.Xmatrix_test = self.Xmatrix[
            self.training
            + self.validating : self.training
            + self.validating
            + self.testing
        ]
        self.Y_test = self.Y[
            self.training
            + self.validating : self.training
            + self.validating
            + self.testing
        ]

    def rectangle(self, a: np.array, n: int) -> np.array:
        # First point
        if self.center:
            (x1, x2) = np.random.randint(0, (n - 1) // 2, 2)
            (y1, y2) = (n - 1 - x1, n - 1 - x2)
        else:
            (x1, x2) = np.random.randint(0, n - 2, 2)
            # Makes sure that second point is not equal for first
            x = random.randint(0, 1)
            # Second point
            (y1, y2) = (
                random.randint(x1 + 1 * x, n - 1),
                random.randint(x2 + 1 * (1 - x), n - 1),
            )

        a[x1 : y1 + 1, y2] = 1
        a[x1 : y1 + 1, x2] = 1
        a[x1, x2:y2] = 1
        a[y1, x2:y2] = 1
        return a

    def cross(self, a: np.array, n: int) -> np.array:
        # Center point
        if self.center:
            (x1, x2) = ((n - 1) // 2, (n - 1) // 2)
        else:
            (x1, x2) = np.random.randint(1, n - 2, 2)
        # Find min lenght in x1 and x2 direction
        (x1_min, x2_min) = (min(x1, n - 1 - x1), min(x2, n - 1 - x2))

        # Find random length in x1 and x2 direction
        (x1_dir, x2_dir) = (
            random.randint(1, x1_min),
            random.randint(1, x2_min),
        )

        a[x1, x2 - x2_dir : x2 + x2_dir + 1] = 1
        a[x1 - x1_dir : x1 + x1_dir + 1, x2] = 1
        return a

    def circle(self, a: np.array, n: int) -> np.array:
        if self.center:
            (x1, x2) = ((n - 1) // 2, (n - 1) // 2)
        else:
            (x1, x2) = np.random.randint(2, n - 2, 2)
        # Radius, max just to make sure the circle fits the frame
        max_r = min(x1, n - x1, x2, n - x2)
        r = max(
            np.round(np.random.random() * max_r), 3
        )  # But we dont want to make it too small
        # Create index arrays to a. I corresponds to distance in x-direction and J in y-direction
        I, J = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[0]))
        # calculate distance of all points to centre
        dist = np.round(np.sqrt((I - x1) ** 2 + (J - x2) ** 2))
        # Plot points
        a[np.where(dist == r)] = 1
        return a

    def triangle(self, a: np.array, n: int) -> np.array:
        # make a circle
        circle = self.circle(np.zeros_like(a), n)
        # Now we pick three points on circle, that defines the triangle
        circle = circle.flatten()
        # Make array for index
        arang = np.arange(n ** 2)
        index_array = circle * arang
        # Pick three points at random, they define points in R^2
        (x1, x2, x3) = np.random.choice(index_array[index_array >= 1], 3, replace=False)
        # print(index_array)
        x1 = np.array([x1 // n, x1 % n])
        x2 = np.array([x2 // n, x2 % n])
        x3 = np.array([x3 // n, x3 % n])
        x = np.c_[x1, x2, x3].astype(int)
        # Draw lines between the points

        mat0 = a.copy()
        for i in combinations([0, 1, 2], 2):
            # Initialize matrix
            mat = mat0
            (x0, y0) = x[:, i[0]]
            (x1, y1) = x[:, i[1]]

            if (x0, y0) == (x1, y1):
                mat[x0, y0] = 1
            else:
                # Swap axes if Y slope is smaller than X slope
                if abs(x1 - x0) < abs(y1 - y0):
                    mat = mat.T
                    x0, y0, x1, y1 = y0, x0, y1, x1
                # Swap line direction if necessary
                if x0 > x1:
                    x0, y0, x1, y1 = x1, y1, x0, y0
                # Endpoints
                mat[x0, y0] = 1
                mat[x1, y1] = 1
                # Find indexes that should be 1
                cx = np.arange(x0 + 1, x1)
                cy = np.round(((y1 - y0) / (x1 - x0)) * (cx - x0) + y0).astype(cx.dtype)
                # Make them 1
                mat[cx, cy] = 1
                a += mat0.copy()

        # We  might add same point several times
        a[np.where(a > 0)] = 1

        return a

    def generate(self) -> None:
        """
        shape 0 = rectangle
        shape 1 = cross
        shape 2 = circle
        shape 3 = triangle
        """
        n = self.dim
        shapes = np.random.randint(0, 4, self.size)

        for j in range(self.size):
            shape = shapes[j]
            # Initialize array
            a = np.zeros((n, n))
            # Make noise matrix
            noise = np.zeros(n * n)
            noisevector = np.random.randint(0, n * n - 1, self.noise)
            np.put(noise, noisevector, -1)
            noise = np.reshape(noise, (n, n))

            if shape == 0:
                a = self.rectangle(a, n)

            elif shape == 1:
                a = self.cross(a, n)

            elif shape == 2:
                a = self.circle(a, n)

            elif shape == 3:
                a = self.triangle(a, n)
            # Add points that are noise i.e -1
            a += noise
            a[np.where(a < 0)] = 1
            # Add image and label to self as matrix and vector
            self.Xmatrix[j] = a
            self.Y[j] = shape
            self.X[j] = np.reshape(a, (1, n ** 2))
        self.Xmatrix = np.reshape(self.Xmatrix, (self.size, n, n))
        self.X = np.reshape(self.X, (self.size, n ** 2))
        self.Y = np.reshape(self.Y, (self.size, 1))


if __name__ == "__main__":
    d = {0: "rectangle", 1: "cross", 2: "circle", 3: "triangle"}
    n = 20
    gen = DataGenerator(dimension=n, size=10, noise=0, center=False)
    gen.generate()
    y = gen.Y
    unique, counts = np.unique(y, return_counts=True)
    print("Number of shapes")
    print(dict(zip(d.values(), counts)))
    # Plotter
    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        title = d[int(y[i])]
        a = gen.Xmatrix[i]
        fig.add_subplot(2, 5, 1 + i)
        plt.title(title)
        plt.imshow(a, cmap=plt.cm.gray)
    plt.show(block=True)
