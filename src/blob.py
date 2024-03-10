#!/usr/bin/env python3
# import libraries
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import sys

sizes = {
        # "1k": 1000,
        # "10k": 10_000,
        # "100k": 100_000,
        # "1M": 1_000_000,
        "10M": 10_000_000,
}

for name, size in sizes.items():
    filename= "linear" + name + ".data"
    print(filename)
    with open(filename, "w") as sys.stdout:
        # generate a 2-class classification problem with 1,000 data points,
        # where each data point is a 2-D feature vector
        (X, Y) = make_blobs(n_samples=size, n_features=3, centers=2, 
                        cluster_std=1.5, random_state=1)

        for x, y in zip(X, Y):
            for v in x:
                print(str(v) + ";", end='')
            print(y)

# y = Y.reshape((Y.shape[0], 1))
# # plot the (testing) classification data
# plt.style.use("ggplot")
# # plt.figure()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.title("Data")
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y[:, 0], s=30)
# plt.savefig("matplotlib.png")
