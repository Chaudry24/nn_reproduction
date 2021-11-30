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

# LOAD PARAMETER SPACE FOR TRAINING
with open("npy/training_201_200_y.npy", mode="rb") as file:
    training_parameter_space = np.load(file)

# LOAD PARAMETER SPACE FOR TESTING
with open("npy/test_y.npy", mode="rb") as file:
    testing_parameter_space = np.load(file)

# GENERATE COVARIANCE MATRICES FOR TRAINING SET
cov_mats_train = np.empty([256, 256, training_parameter_space.shape[0]])
for i in range(training_parameter_space.shape[0]):
    print(f"generating training covariance matrix for {i}th value")
    cov_mats_train[:, :, i] = some_file_1.Spatial.compute_covariance(covariance_type="matern",
                                                                     distance_matrix=spatial_distance,
                                                                     variance=1.0, smoothness=1.0,
                                                                     spatial_range=training_parameter_space[i, 1],
                                                                     nugget=np.exp(training_parameter_space[i, 0]))

# GENERATE COVARIANCE MATRICES FOR TESTING SET
cov_mats_test = np.empty([256, 256, testing_parameter_space.shape[0]])
for i in range(testing_parameter_space.shape[0]):
    print(f"generating testing covariance matrix for {i}th value")
    cov_mats_test[:, :, i] = some_file_1.Spatial.compute_covariance(covariance_type="matern",
                                                                    distance_matrix=spatial_distance,
                                                                    variance=1.0, smoothness=1.0,
                                                                    spatial_range=testing_parameter_space[i, 1],
                                                                    nugget=np.exp(testing_parameter_space[i, 0]))


# GENERATE OBSERVATIONS FOR TRAINING FOR A SINGLE REALIZATION
observations_train = np.empty([training_parameter_space.shape[0], 16, 16, 1])
semi_variogram_train = np.empty([training_parameter_space.shape[0], 10])
for i in range(training_parameter_space.shape[0]):
    print(f"generating training data for the {i}th covariance matrix")
    tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i])
    tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1, bins=10)
    semi_variogram_train[i, :] = tmp_var
    observations_train[i, :, :, :] = tmp_array.reshape(16, 16)

# GENERATE OBSERVATIONS FOR TESTING FOR A SINGLE REALIZATION
observations_test = np.empty([testing_parameter_space.shape[0], 16, 16, 1])
semi_variogram_test = np.empty([testing_parameter_space[0], 10])
for i in range(testing_parameter_space.shape[0]):
    print(f"generating testing data for the {i}th covariance matrix")
    tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i])
    tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1, bins=10)
    observations_test[i, :, :, :] = tmp_array.reshape(16, 16)

# GENERATE OBSERVATIONS FOR TRAINING FOR THIRTY REALIZATIONS
observations_train_30 = np.empty([training_parameter_space.shape[0], 16, 16, 30])
semi_variogram_train_30 = np.empty([training_parameter_space[0], 10, 30])
for i in range(training_parameter_space.shape[0]):
    print(f"generating training data for the {i}th covariance matrix")
    for j in range(30):
        print(f"getting {j}th training realization from {i}th covariance matrix")
        tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i])
        tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1)
        semi_variogram_train_30[i, :, j] = tmp_var
        observations_train_30[i, :, :, j] = tmp_array.reshape(16, 16)

# GENERATE OBSERVATIONS FOR TESTING FOR THIRTY REALIZATIONS
observations_test_30 = np.empty([testing_parameter_space.shape[0], 16, 16, 30])
semi_variogram_test_30 = np.empty([testing_parameter_space.shape[0], 10, 30])
for i in range(testing_parameter_space.shape[0]):
    print(f"generating testing data for the {i}th covariance matrix")
    for j in range(30):
        print(f"getting {j}th testing realization from {i}th covariance matrix")
        tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i])
        tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1)
        semi_variogram_test_30[i, :, j] = tmp_var
        observations_test_30[i, :, :, j] = tmp_array.reshape(16, 16)


def negative_log_likelihood(variance, spatial_range, smoothness, nugget,
                            covariance_mat, observations, n_points=256):

    # compute lower cholesky
    lower_cholesky = np.linalg.cholesky(covariance_mat)
    # first term of negative log likelihood function
    first_term = n_points / 2.0 * np.log(2.0 * np.pi)
    # compute the log determinant term
    log_determinant_term = 2.0 * np.trace(np.log(lower_cholesky))
    # the second term of the negative log likelihood function
    second_term = 0.5 * log_determinant_term
    # the third term of the negative log likelihood function
    third_term = float(0.5 * observations.T @ np.linalg.inv(covariance_mat)
                       @ observations)
    return first_term + second_term + third_term


# ------- DIFFERENT ARCHITECUTRE FOR DIFFERENT NNs ------- #

model_NF = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=10, input_shape=(16, 16, 1), activation="relu",
                           data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation="relu"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=500, activation="relu"),
    tf.keras.layers.Dense(units=2)
])

model_NF30 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=10, input_shape=(16, 16, 30), activation="relu",
                           data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation="relu"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=500, activation="relu"),
    tf.keras.layers.Dense(units=2)
])

model_NV = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3000, activation="relu"),
    tf.keras.layers.Dense(units=1000),
    tf.keras.layers.Dense(units=2)
])

model_NV30 = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3000, activation="relu"),
    tf.keras.layers.Dense(units=1000),
    tf.keras.layers.Dense(units=2)
])

# ------- COMPILE DIFFERENT NNs ------- #

model_NF.compile(optimizer=tf.optimizers.Adam(),
                 loss=tf.losses.MeanAbsoluteError(),
                 metrics=[tf.metrics.RootMeanSquaredError()])

model_NF30.compile(optimizer=tf.optimizers.Adam(),
                   loss=tf.losses.MeanAbsoluteError(),
                   metrics=[tf.metrics.RootMeanSquaredError()])

model_NV.compile(optimizer=tf.optimizers.Adam(),
                 loss=tf.losses.MeanAbsoluteError(),
                 metrics=[tf.metrics.RootMeanSquaredError()])

model_NV30.compile(optimizer=tf.optimizers.Adam(),
                   loss=tf.losses.MeanAbsoluteError(),
                   metrics=[tf.metrics.RootMeanSquaredError()])

# ------- TRAIN DIFFERENT NNs ------- #

history_NF = model_NF.fit(x=training_parameter_space, y=observations_train, batch_size=16,
                          epochs=2)

history_NF30 = model_NF.fit(x=training_parameter_space, y=observations_train_30, batch_size=16,
                            epochs=2)

history_NV = model_NF.fit(x=training_parameter_space, y=semi_variogram_train, batch_size=16,
                          epochs=2)

history_NV30 = model_NF.fit(x=training_parameter_space, y=semi_variogram_train_30, batch_size=16,
                            epochs=2)

# ------- SAVE TRAINING LOSS FOR EACH NN ------- #

with open("./tf_stat_reproduction/NF/training_loss_NF.npy", mode="wb") as file:
    np.save(file, history_NF.history["loss"])

with open("./tf_stat_reproduction/NF30/training_loss_NF30.npy", mode="wb") as file:
    np.save(file, history_NF30.history["loss"])

with open("./tf_stat_reproduction/NV/training_loss_NV.npy", mode="wb") as file:
    np.save(file, history_NV.history["loss"])

with open("./tf_stat_reproduction/NV30/training_loss_NV30.npy", mode="wb") as file:
    np.save(file, history_NV30.history["loss"])

# ------- GET PREDICTIONS FOR EACH NN ------- #

preds_NF = model_NF.predict(x=observations_test)

preds_NF30 = model_NF.predict(x=observations_test_30)

preds_NV = model_NF.predict(x=semi_variogram_test)

preds_NV30 = model_NF.predict(x=semi_variogram_test_30)

# ------- SAVE PREDICTIONS FOR EACH NN ------- #

with open("./tf_stat_reproduction/NF/preds_NF.npy", mode="wb") as file:
    np.save(file, preds_NF)

with open("./tf_stat_reproduction/NF30/preds_NF30.npy", mode="wb") as file:
    np.save(file, preds_NF30)

with open("./tf_stat_reproduction/NF/preds_NV.npy", mode="wb") as file:
    np.save(file, preds_NV)

with open("./tf_stat_reproduction/NF/preds_NV30.npy", mode="wb") as file:
    np.save(file, preds_NV30)

# ------- SAVE TRAINED NNs ------- #

model_NF.save(filepath="./tf_stat_reproduction/NF")
model_NF30.save(filepath="./tf_stat_reproduction/NF30")
model_NV.save(filepath="./tf_stat_reproduction/NV")
model_NV30.save(filepath="./tf_stat_reproduction/NV30")

# ------- COMPUTE MLE FOR A SINGLE REALIZATION ------- #

mle_estimates = np.empty([testing_parameter_space.shape[0]])
for i in range(testing_parameter_space.shape[0]):
    mle_estimates[i] = negative_log_likelihood(variance=1.0, spatial_range=testing_parameter_space[i, 0],
                                               smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 1]),
                                               covariance_mat=cov_mats_test[:, :, i],
                                               observations=observations_test[i, :].reshape(256))
mle_pred = observations_test[np.argmin(mle_estimates), :]

# ------- SAVE MLE PRED FOR A SINGLE REALIZATION ------- #

with open("./tf_stat_reproduction/ML/preds_MLE.npy", mode="wb") as file:
    np.save(file, mle_pred)

# ------- COMPUTE MLE FOR THIRTY REALIZATIONS ------- #
# TODO FIGURE THIS OUT
mle_estimates_30 = np.empty([testing_parameter_space.shape[0]])
tmp_array = np.empty([testing_parameter_space.shape[0], 30])
for i in range(testing_parameter_space.shape[0]):
    for j in range(30):
        tmp_array[i, j] = negative_log_likelihood(variance=1.0, spatial_range=testing_parameter_space[i, 0],
                                                  smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 1]),
                                                  covariance_mat=cov_mats_test[:, :, i],
                                                  observations=observations_test_30[i, :, :, j].reshape(256))
    mle_estimates_30
    mle_estimates[i] = negative_log_likelihood(variance=1.0, spatial_range=testing_parameter_space[i, 0],
                                               smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 1]),
                                               covariance_mat=cov_mats_test[:, :, i],
                                               observations=observations_test[i, :].reshape(256))
mle_pred = observations_test[np.argman(mle_estimates), :]

# ------- SAVE MLE PRED FOR THIRTY REALIZATIONS ------- #

with open("./tf_stat_reproduction/ML/preds_MLE.npy", mode="wb") as file:
    np.save(file, mle_pred)
