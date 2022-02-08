# from PIL import Image
import numpy as np
import random
from itertools import combinations

# random.seed(1)

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
        self.X_val = self.X[0 : self.validating]
        self.Xmatrix_val = self.Xmatrix[0 : self.validating]
        self.Y_val = self.Y[0 : self.validating]
        # Testing
        self.X_test = self.X[0 : self.testing]
        self.Xmatrix_test = self.Xmatrix[0 : self.testing]
        self.Y_test = self.Y[0 : self.testing]

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
            # Make empty array with noise
            # Initialize array
            a = np.zeros(n * n)  # * 255
            # Add noise
            noisevector = np.random.randint(0, n * n - 1, self.noise)
            np.put(a, noisevector, -1)
            # Make a matrix
            a = np.reshape(a, (n, n))
            if shape == 0:
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

                a[x1 : y1 + 1, y2] += 1
                a[x1 : y1 + 1, x2] += 1
                a[x1, x2:y2] += 1
                a[y1, x2:y2] += 1

            elif shape == 1:
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

                a[x1, x2 - x2_dir : x2 + x2_dir + 1] += 1
                a[x1 - x1_dir : x1 + x1_dir + 1, x2] += 1

            elif shape == 2:
                # Center point
                if self.center:
                    (x1, x2) = ((n - 1) // 2, (n - 1) // 2)
                else:
                    (x1, x2) = np.random.randint(2, n - 2, 2)
                # Radius
                max_r = min(x1, n - x1, x2, n - x2)

                r = max(np.round(np.random.random() * max_r), 2)
                # Create index arrays to a
                I, J = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]))

                # calculate distance of all points to centre
                dist = np.round(np.sqrt((I - x1) ** 2 + (J - x2) ** 2))

                a[np.where(dist == r)] = +1

            elif shape == 3:

                # make new circle without noise
                zero = np.zeros_like(a)
                if self.center:
                    (x1, x2) = ((n - 1) // 2, (n - 1) // 2)
                else:
                    (x1, x2) = np.random.randint(2, n - 2, 2)
                # Radius
                max_r = min(x1, n - x1, x2, n - x2)

                r = max(np.round(np.random.random() * max_r), 2)
                # Create index arrays to a
                I, J = np.meshgrid(np.arange(zero.shape[0]), np.arange(zero.shape[1]))

                # calculate distance of all points to centre
                dist = np.round(np.sqrt((I - x1) ** 2 + (J - x2) ** 2))
                zero[np.where(dist == r)] = +1
                # Now we pick three points on circle, that defines the triangle
                zero = np.reshape(zero, (zero.shape[0] ** 2, 1))
                # Make array for index
                arang = np.arange(0, zero.shape[0])
                index_array = zero * arang
                # Pick three points at random, they define points in R^2
                (x1, x2, x3) = np.random.choice(
                    index_array[index_array > 0], 3, replace=False
                )
                x1 = np.array([x1 // n, x1 % n])
                x2 = np.array([x2 // n, x2 % n])
                x3 = np.array([x3 // n, x3 % n])
                x = np.c_[x1, x2, x3].astype(int)
                # Draw lines
                mat0 = -a.copy()
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
                        cy = np.round(((y1 - y0) / (x1 - x0)) * (cx - x0) + y0).astype(
                            cx.dtype
                        )
                        # Write intermediate coordinates
                        mat[cx, cy] += 1
                        a += mat0.copy()  # .copy()
                        a[np.where(a > 1)] = 1
            # Erase points that are noise i.e -1
            a[np.where(a != 0)] = 1
            # Add image to self
            self.Xmatrix[j] = a
            self.Y[j] = shape
            self.X[j] = np.reshape(a, (1, n ** 2))
        self.Xmatrix = np.reshape(self.Xmatrix, (self.size, n, n))
        self.X = np.reshape(self.X, (self.size, n ** 2))
        self.Y = np.reshape(self.Y, (self.size, 1))


if __name__ == "__main__":
    d = {0: "rectangle", 1: "cross", 2: "circle", 3: "triangle"}
    n = 50
    gen = DataGenerator(n, 5000, 5, False)
    gen.generate()
    y = gen.Y
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(d.values(), counts)))
    # PLotter
    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        title = d[int(y[i])]
        a = gen.Xmatrix[i]
        fig.add_subplot(2, 5, 1 + i)
        plt.title(title)
        plt.imshow(a, cmap=plt.cm.gray)
    plt.show(block=True)
    # plt.show()
