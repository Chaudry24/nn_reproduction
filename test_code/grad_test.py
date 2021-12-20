import keras

import some_file_1
import numpy as np
import tensorflow as tf
from scipy.special import gamma, kv


# SPATIAL GRID
x = np.linspace(1, 16, 16)
y = x
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
spatial_grid = np.array([x, y]).T

# COMPUTE DISTANCE MATRIX
spatial_distance = some_file_1.Spatial.compute_distance(spatial_grid[:, 0], spatial_grid[:, 1])

true_val = tf.Variable(1.2)
guess = tf.Variable(1.4)

y = tf.exp(-spatial_distance / true_val)


def loss(yhat):
    return tf.norm(y-yhat)


optimizer = tf.keras.optimizers.Adam()

model = tf.keras.Model(inputs=keras.Input(shape=1), outputs=tf.keras.layers.Dense(1))

for i in range(10):
    # compute yhat
    with tf.GradientTape(persistent=True) as tape:
        yhat = tf.exp(-spatial_distance / guess)
        # compute loss
        l = loss(yhat)
    # compute derivative
    dl_dguess = tape.gradient(l, guess)
    # update guess
    optimizer.apply_gradients(zip(dl_dguess, guess))

# dy_db
# dy_db = tape.jacobian(y, b)
1+1

# fin_dif = (np.exp(-spatial_distance / (1.2 + 1e-3)) - np.exp(-spatial_distance / 1.2)) * 1e-3
1+1
2+2
