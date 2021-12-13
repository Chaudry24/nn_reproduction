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

# COMPUTE DISTANCE MATRIX
spatial_distance = some_file_1.Spatial.compute_distance(spatial_grid[:, 0], spatial_grid[:, 1])

# set number of epochs
n_epochs = 1000

# # open train cholesky matrices
# with open("../npy/chol_mats_train_my_idea.npy", mode="rb") as file:
#     chol_mats_train = np.load(file)

# open parameter space for training
with open("../npy/training_my_idea.npy", mode="rb") as file:
    training_parameter_space = np.load(file)

# GENERATE COVARIANCE MATRICES FOR TRAINING SET
# compute the covariance matrices
cov_mats_train = np.array([some_file_1.Spatial.compute_covariance
                          (covariance_type="matern", distance_matrix=spatial_distance,
                           variance=1.0, smoothness=1.0,
                           spatial_range=training_parameter_space[i, 1],
                           nugget=np.exp(training_parameter_space[i, 0]))
                          for i in range(training_parameter_space.shape[0])])

# compute cholesky matrices for training
chol_mats_train = np.linalg.cholesky(cov_mats_train)

# open test data
with open("../npy/test_semivariogram_my_idea.npy", mode="rb") as file:
    semivariogram_test = np.load(file)

# NN architecture
model_NV = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3000, activation="relu"),
    tf.keras.layers.Dense(units=1000),
    tf.keras.layers.Dense(units=2)
])

# compile NN
model_NV.compile(optimizer=tf.optimizers.Adam(),
                 loss=tf.losses.MeanAbsoluteError(),
                 metrics=[tf.metrics.RootMeanSquaredError()])

# empty list to store losses
loss_NV = []

# ------- TRAIN DIFFERENT NNs ------- #

for i in range(n_epochs):

    # print start of epoch
    print(f"starting iteration {i}")

    # generate data at every 5th iteration
    if i % 41 == 0:
        print(f"generating data for {i}th time (mod 5)")

        # GENERATE OBSERVATIONS FOR TRAINING FOR A SINGLE REALIZATION
        observations_train = (chol_mats_train @ np.random.randn(training_parameter_space.shape[0], 256, 1)).reshape(
            training_parameter_space.shape[0], 16, 16, 1)
        semi_variogram_train = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                                   observations_train[i, :].reshape(256,
                                                                                                                    1),
                                                                                   realizations=1, bins=10).ravel() for
                                         i in
                                         range(training_parameter_space.shape[0])])

    history_NV = model_NV.fit(x=tf.convert_to_tensor(semi_variogram_train),
                              y=tf.convert_to_tensor(training_parameter_space), batch_size=16,
                              epochs=20)

    # store losses for each "epoch"
    loss_NV.append(history_NV.history["loss"])

    # print end of epoch
    print(f"iteration {i} ended")

# ------- SAVE TRAINING LOSS FOR NN ------- #

loss_NV = np.array([nums for lists in loss_NV for nums in lists])

with open("NV/training_loss_NV.npy", mode="wb") as file:
    np.save(file, loss_NV)

# ------- GET PREDICTIONS FOR NN ------- #

preds_NV = model_NV.predict(x=tf.convert_to_tensor(semivariogram_test))

# ------- SAVE PREDICTIONS FOR NN ------- #

with open("NV/preds_NV.npy", mode="wb") as file:
    np.save(file, preds_NV)

# ------- SAVE TRAINED NN ------- #

model_NV.save(filepath="./NV")
