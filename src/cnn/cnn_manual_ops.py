import torch
import torch.nn as nn



def corr2d(X, K):

    h, w = K.shape

    Y = torch.zeros(
        (X.shape[0] - h + 1,
         X.shape[1] - w + 1)
    )

    for i in range(Y.shape[0]):

        for j in range(Y.shape[1]):

            Y[i, j] = (
                X[i:i+h, j:j+w] * K
            ).sum()

    return Y

# X = torch.tensor([
#     [1., 1., 1., 0., 0., 0.],
#     [1., 1., 1., 0., 0., 0.],
#     [1., 1., 1., 0., 0., 0.],
#     [1., 1., 1., 0., 0., 0.],
#     [1., 1., 1., 0., 0., 0.],
#     [1., 1., 1., 0., 0., 0.]
# ])

# K = torch.tensor([
#     [1., -1.]
# ])

# Y = corr2d(X, K)

# print(Y)

def max_pool2d(X, pool_size):

    p_h, p_w = pool_size

    Y = torch.zeros(
        (
            X.shape[0] - p_h + 1,
            X.shape[1] - p_w + 1
        )
    )

    for i in range(Y.shape[0]):

        for j in range(Y.shape[1]):

            Y[i, j] = X[
                i:i+p_h,
                j:j+p_w
            ].max()

    return Y

def avg_pool2d(X, pool_size):

    p_h, p_w = pool_size

    Y = torch.zeros(
        (
            X.shape[0] - p_h + 1,
            X.shape[1] - p_w + 1
        )
    )

    for i in range(Y.shape[0]):

        for j in range(Y.shape[1]):

            Y[i, j] = X[
                i:i+p_h,
                j:j+p_w
            ].mean()

    return Y

Y = torch.tensor([
    [0., 1., 2.],
    [3., 4., 5.],
    [6., 7., 8.]
])

print(max_pool2d(Y, (2, 2)))
print(avg_pool2d(Y, (2, 2)))


pool = nn.MaxPool2d(2)

X = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

print(X)

print(pool(X))