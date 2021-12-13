import numpy as np
import tensorflow as tf
import some_file_1

# SPATIAL GRID
x = np.linspace(1, 16, 16)
y = x
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
spatial_grid = np.array([x, y]).T

# set number of epochs
n_epochs = 1000

# open train cholesky matrices
with open("../npy/chol_mats_train_tf_stat.npy", mode="rb") as file:
    chol_mats_train = np.load(file)

# open parameter space for training
with open("../npy/training_201_200_y.npy", mode="rb") as file:
    training_parameter_space = np.load(file)

# open test data
with open("../npy/test30_semivariogram_tf_stat.npy", mode="rb") as file:
    semivariogram_test_30 = np.load(file)

# NN architecture
model_NV30 = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3000, activation="relu"),
    tf.keras.layers.Dense(units=1000),
    tf.keras.layers.Dense(units=2)
])

# compile NN
model_NV30.compile(optimizer=tf.optimizers.Adam(),
                   loss=tf.losses.MeanAbsoluteError(),
                   metrics=[tf.metrics.RootMeanSquaredError()])

# empty list to store losses
loss_NV30 = []

# ------- TRAIN DIFFERENT NNs ------- #

for i in range(n_epochs):

    # print start of epoch
    print(f"starting iteration {i}")

    # generate data at every 5th iteration
    if i % 41 == 0:
        print(f"generating data for {i}th time (mod 5)")

        # GENERATE OBSERVATIONS FOR TRAINING FOR THIRTY REALIZATIONS
        observations_train_30 = (chol_mats_train @ np.random.randn(training_parameter_space.shape[0], 256, 30)).reshape(
            training_parameter_space.shape[0], 16, 16, 30)
        semi_variogram_train_30 = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                                      observations_train_30[i, :, :,
                                                                                      j].reshape(256, -1)).ravel()
                                            for i in range(training_parameter_space.shape[0]) for j in
                                            range(30)]).reshape(training_parameter_space.shape[0], -1)

    history_NV30 = model_NV30.fit(x=tf.convert_to_tensor(semi_variogram_train_30),
                                  y=tf.convert_to_tensor(training_parameter_space), batch_size=16,
                                  epochs=20)

    # store losses for each "epoch"
    loss_NV30.append(history_NV30.history["loss"])

    # print end of epoch
    print(f"iteration {i} ended")

# ------- SAVE TRAINING LOSS FOR NN ------- #

loss_NV30 = np.array([nums for lists in loss_NV30 for nums in lists])

with open("NV30/training_loss_NV30.npy", mode="wb") as file:
    np.save(file, loss_NV30)

# ------- GET PREDICTIONS FOR NN ------- #

preds_NV30 = model_NV30.predict(x=tf.convert_to_tensor(semivariogram_test_30))

# ------- SAVE PREDICTIONS FOR NN ------- #

with open("NV30/preds_NV30.npy", mode="wb") as file:
    np.save(file, preds_NV30)

# ------- SAVE TRAINED NN ------- #

model_NV30.save(filepath="./NV30")