import numpy as np
import tensorflow as tf
import some_file_1
import dask
import cupy as cp
# import dask.distributed
# import graphviz
# import matplotlib.pyplot as plt

# SPATIAL GRID
x = np.linspace(1, 16, 16)
y = x
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
spatial_grid = np.array([x, y]).T

# set number of epochs
n_epochs = 100

# COMPUTE DISTANCE MATRIX
spatial_distance = cp.array(some_file_1.Spatial.compute_distance(spatial_grid[:, 0], spatial_grid[:, 1]))

# LOAD PARAMETER SPACE FOR TRAINING
with open("npy/training_201_200_y.npy", mode="rb") as file:
    training_parameter_space = cp.array(np.load(file)[0:3, :])

# LOAD PARAMETER SPACE FOR TESTING
with open("npy/test_y.npy", mode="rb") as file:
    n_test_samples = 2
    testing_parameter_space = np.load(file)
    test_sample_indices = np.random.choice(testing_parameter_space.shape[0], n_test_samples, replace=False)
    testing_parameter_space = testing_parameter_space[test_sample_indices]

# SAVE TESTING PARAMETER SUBSET
with open("npy/test_subset.npy", mode="wb") as file:
    np.save(file, testing_parameter_space)

# CONVERT TESTING PARAMETER SPACE INTO CUPY ARRAY
testing_parameter_space = cp.array(testing_parameter_space)

# GENERATE COVARIANCE MATRICES FOR TRAINING SET
# send computations to the scheduler
tmp_array = np.array([dask.delayed(some_file_1.Spatial.compute_covariance)
                      (covariance_type="matern", distance_matrix=spatial_distance,
                       variance=1.0, smoothness=1.0,
                       spatial_range=training_parameter_space[i, 1],
                       nugget=cp.exp(training_parameter_space[i, 0])).persist()
                      for i in range(training_parameter_space.shape[0])])
# compute the covariance matrices
cov_mats_train = cp.array([computations.compute() for computations in tmp_array])

# delete tmp_array before next use
del tmp_array

# GENERATE COVARIANCE MATRICES FOR TESTING SET
# send computations to the scheduler
tmp_array = np.array([dask.delayed(some_file_1.Spatial.compute_covariance)
                      (covariance_type="matern", distance_matrix=spatial_distance,
                       variance=1.0, smoothness=1.0,
                       spatial_range=training_parameter_space[i, 1],
                       nugget=np.exp(training_parameter_space[i, 0])).persist()
                      for i in range(testing_parameter_space.shape[0])])
# compute the covariance matrices
cov_mats_test = cp.array([computations.compute() for computations in tmp_array])

# delete tmp_array before next use
del tmp_array

# GENERATE OBSERVATIONS FOR TESTING FOR A SINGLE REALIZATION
tmp_array = np.array(
    [dask.delayed(some_file_1.Spatial.observations)(realizations=1, covariance=cov_mats_train[i, :, :]).persist()
     for i in range(testing_parameter_space.shape[0])])
observations_test = cp.array([computations.compute().reshape(16, 16, 1) for computations in tmp_array])
# for i in range(testing_parameter_space.shape[0]):
#     print(f"spatial grid: {spatial_grid}")
#     print(f"observation test .get {observations_test[i, :].reshape(256, 1).get()}")
#     print(f"observation test {observations_test[i, :].reshape(256, 1)}")
#     print(f"variogram {some_file_1.Spatial.compute_semivariogram(spatial_grid, observations_test[i, :].reshape(256, 1).get(), realizations = 1, bins = 10)}")
semi_variogram_test = cp.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                          observations_test[i, :].reshape(256, 1).get(),
                                                                          realizations=1, bins=10).ravel() for i in
                                range(testing_parameter_space.shape[0])])

# delete tmp_array before next use
del tmp_array

# GENERATE OBSERVATIONS FOR TESTING FOR THIRTY REALIZATIONS
observations_test_30 = cp.empty([testing_parameter_space.shape[0], 16, 16, 30])
semi_variogram_test_30 = cp.empty([testing_parameter_space.shape[0], 10, 30])
tmp_array = np.array(
    [dask.delayed(some_file_1.Spatial.observations)(realizations=1, covariance=cov_mats_train[i, :, :]).persist()
     for i in range(testing_parameter_space.shape[0]) for j in range(30)])
tmp1 = cp.array([tmp_array[i].compute().reshape(16, 16) for i in range(30 * testing_parameter_space.shape[0])])
tmp2 = cp.array(
    [some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp1[i, :].reshape(256, -1).get(), 1).ravel() for i in
     range(30 * testing_parameter_space.shape[0])])

print(f"\n{tmp2.shape}\n")
for i in range(testing_parameter_space.shape[0]):
    for j in range(30):
        print(tmp2[j + 30 * i, :])
        print(tmp2[j + 30 * i, :].shape)
        observations_test_30[i, :, :, j] = tmp1[j, :]
        semi_variogram_test_30[i, :, j] = tmp2[j + 30 * i, :]


# delete temp arrays before next use
del tmp1
del tmp2
del tmp_array


def negative_log_likelihood(variance, spatial_range, smoothness, nugget,
                            covariance_mat, observations, n_points=256):

    # ravel observations
    observations = observations.ravel()
    # compute lower cholesky
    lower_cholesky = cp.linalg.cholesky(covariance_mat)
    # first term of negative log likelihood function
    first_term = n_points / 2.0 * cp.log(2.0 * cp.pi)
    # compute the log determinant term
    log_determinant_term = 2.0 * cp.trace(cp.log(lower_cholesky))
    # the second term of the negative log likelihood function
    second_term = 0.5 * log_determinant_term
    # the third term of the negative log likelihood function
    third_term = float(0.5 * observations.T @ cp.linalg.inv(covariance_mat)
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

# empty lists to store losses

loss_NF = []
loss_NF30 = []
loss_NV = []
loss_NV30 = []


# ------- TRAIN DIFFERENT NNs ------- #

for i in range(n_epochs):

    # print start of epoch
    print(f"starting epoch {i}")

    # GENERATE OBSERVATIONS FOR TRAINING FOR A SINGLE REALIZATION
    tmp_array = np.array(
        [dask.delayed(some_file_1.Spatial.observations)(realizations=1, covariance=cov_mats_train[i, :, :]).persist()
         for i in range(training_parameter_space.shape[0])])
    observations_train = cp.array([computations.compute().reshape(16, 16, 1) for computations in tmp_array])
    semi_variogram_train = cp.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                               observations_train[i, :].reshape(256, 1).get(),
                                                                               realizations=1, bins=10).ravel() for i in
                                     range(training_parameter_space.shape[0])])

    # delete tmp_array before next use
    del tmp_array

    # GENERATE OBSERVATIONS FOR TRAINING FOR THIRTY REALIZATIONS
    observations_train_30 = cp.empty([training_parameter_space.shape[0], 16, 16, 30])
    semi_variogram_train_30 = cp.empty([training_parameter_space.shape[0], 10, 30])
    tmp_array = np.array(
        [dask.delayed(some_file_1.Spatial.observations)(realizations=1, covariance=cov_mats_train[i, :, :]).persist()
         for i in range(training_parameter_space.shape[0]) for j in range(30)])
    tmp1 = cp.array([tmp_array[i].compute().reshape(16, 16) for i in range(30 * training_parameter_space.shape[0])])
    tmp2 = cp.array([some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp1[i, :].reshape(256, -1).get(), 1).ravel() for i in
                     range(30 * training_parameter_space.shape[0])])

    for i in range(training_parameter_space.shape[0]):
        for j in range(30):
            print(tmp2[j + 30 * i, :])
            print(tmp2[j + 30 * i, :].shape)
            observations_train_30[i, :, :, j] = tmp1[j, :]
            semi_variogram_train_30[i, :, j] = tmp2[j + 30 * i, :]

    # delete temp arrays before next use
    del tmp1
    del tmp2
    del tmp_array

    print(f"\n{observations_train.shape} and {training_parameter_space.shape}\n")
    history_NF = model_NF.fit(x=observations_train.get(), y=training_parameter_space.get(), batch_size=16,
                              epochs=2)

    print(f"\n{observations_train_30.shape} and {training_parameter_space.shape}\n")
    history_NF30 = model_NF30.fit(x=observations_train_30.get(), y=training_parameter_space.get(), batch_size=16,
                                  epochs=2)

    print(f"\n{semi_variogram_train.shape} and {training_parameter_space.shape}\n")
    history_NV = model_NV.fit(x=semi_variogram_train.get(), y=training_parameter_space.get(), batch_size=16,
                              epochs=2)

    print(f"\n{semi_variogram_train_30.shape} and {training_parameter_space.shape}\n")
    history_NV30 = model_NV30.fit(x=semi_variogram_train_30.get(), y=training_parameter_space.get(), batch_size=16,
                                  epochs=2)

    # store losses for each "epoch"
    loss_NF.append(history_NF.history["loss"])
    loss_NF30.append(history_NF30.history["loss"])
    loss_NV30.append(history_NV.history["loss"])
    loss_NV30.append(history_NV30.history["loss"])

# ------- SAVE TRAINING LOSS FOR EACH NN ------- #

loss_NF = np.array([nums for lists in loss_NF for nums in lists])
loss_NF30 = np.array([nums for lists in loss_NF30 for nums in lists])
loss_NV = np.array([nums for lists in loss_NV for nums in lists])
loss_NV30 = np.array([nums for lists in loss_NV30 for nums in lists])

with open("./tf_stat_reproduction/NF/training_loss_NF.npy", mode="wb") as file:
    np.save(file, loss_NF)

with open("./tf_stat_reproduction/NF30/training_loss_NF30.npy", mode="wb") as file:
    np.save(file, loss_NF30)

with open("./tf_stat_reproduction/NV/training_loss_NV.npy", mode="wb") as file:
    np.save(file, loss_NV)

with open("./tf_stat_reproduction/NV30/training_loss_NV30.npy", mode="wb") as file:
    np.save(file, loss_NV30)

# ------- GET PREDICTIONS FOR EACH NN ------- #

preds_NF = model_NF.predict(x=observations_test.get())

preds_NF30 = model_NF30.predict(x=observations_test_30.get())

preds_NV = model_NV.predict(x=semi_variogram_test.get())

preds_NV30 = model_NV30.predict(x=semi_variogram_test_30.get())

# ------- SAVE PREDICTIONS FOR EACH NN ------- #

with open("./tf_stat_reproduction/NF/preds_NF.npy", mode="wb") as file:
    np.save(file, preds_NF)

with open("./tf_stat_reproduction/NF30/preds_NF30.npy", mode="wb") as file:
    np.save(file, preds_NF30)

with open("./tf_stat_reproduction/NV/preds_NV.npy", mode="wb") as file:
    np.save(file, preds_NV)

with open("./tf_stat_reproduction/NV30/preds_NV30.npy", mode="wb") as file:
    np.save(file, preds_NV30)

# ------- SAVE TRAINED NNs ------- #

model_NF.save(filepath="./tf_stat_reproduction/NF")
model_NF30.save(filepath="./tf_stat_reproduction/NF30")
model_NV.save(filepath="./tf_stat_reproduction/NV")
model_NV30.save(filepath="./tf_stat_reproduction/NV30")

# ------- COMPUTE MLE FOR A SINGLE REALIZATION ------- #


mle_estimates = np.empty([testing_parameter_space.shape[0], 2])
tmp_array = np.array([dask.delayed(negative_log_likelihood)
                      (variance=1.0, spatial_range=testing_parameter_space[j, 1],
                       smoothness=1.0, nugget=np.exp(testing_parameter_space[j, 0]),
                       covariance_mat=cov_mats_test[j, :, :],
                       observations=observations_test[i, :].reshape(256)).persist() for i in range(observations_test.shape[0])
                      for j in range(testing_parameter_space.shape[0])])
for i in range(observations_test.shape[0]):
    tmp1 = np.array([tmp_array[j].compute() for j in range(i * testing_parameter_space.shape[0], (i + 1) * testing_parameter_space.shape[0])])
    mle_estimates[i, 0] = testing_parameter_space[np.argmin(tmp1), 0]
    mle_estimates[i, 1] = testing_parameter_space[np.argmin(tmp1), 1]

# ------- SAVE MLE PRED FOR A SINGLE REALIZATION ------- #

with open("./tf_stat_reproduction/ML/preds_MLE.npy", mode="wb") as file:
    np.save(file, mle_estimates)

# ------- COMPUTE MLE FOR THIRTY REALIZATIONS ------- #

# delete tmp_array before next use
del tmp_array
del tmp1

mle_estimates_30 = np.empty([testing_parameter_space.shape[0], 2])
tmp_array1 = cp.empty([30, testing_parameter_space.shape[0]])
tmp_array2 = cp.empty([30, 2])

for l in range(observations_test_30.shape[0]):
    tmp1 = np.array([dask.delayed(negative_log_likelihood)
                     (variance=1.0, spatial_range=testing_parameter_space[i, 1],
                      smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 0]),
                      covariance_mat=cov_mats_test[i, :, :],
                      observations=observations_test_30[l, :, :, j].reshape(256))
                     for i in range(testing_parameter_space.shape[0])
                     for j in range(30)])
    tmp2 = cp.array([tmp1[i].compute() for i in range(30 * testing_parameter_space.shape[0])])
    # tmp3 gives the index of point of interest for each realization
    tmp3 = cp.array([cp.argmin(tmp2[i + j]) for i in range(testing_parameter_space.shape[0]) for j in range(30)])
    # store the value at the point of interest for each realization
    tmp4 = testing_parameter_space[tmp3.get(), :]
    # tmp4 = np.array([testing_parameter_space[np.argmin(tmp3[i]), :] for i in range(testing_parameter_space.shape[0])])
    # save the averages
    mle_estimates_30[l, 0] = np.average(tmp4[:, 0])
    mle_estimates_30[l, 1] = np.average(tmp4[:, 1])

# for loop for each sample
# for l in range(observations_test_30.shape[0]):
#     # for loop to save min parameters for each realization
#     for k in range(30):
#         # for loop for each parameter value
#         for i in range(testing_parameter_space.shape[0]):
#             # for loop for each realization
#             for j in range(30):
#                 print(f"\nStarting MLE for {l}th sample {j}th realization using {i}th parameters\n")
#                 tmp_array1[j, i] = negative_log_likelihood(variance=1.0, spatial_range=testing_parameter_space[i, 1],
#                                                            smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 0]),
#                                                            covariance_mat=cov_mats_test[i, :, :],
#                                                            observations=observations_test_30[l, :, :, j].reshape(256))
#                 print(f"\nThe MLE for {l}th sample {j}th realization using {i}th parameters is {tmp_array1[j, i]}\n")
#                 print(f"\nEnded MLE for {l}th sample {j}th realization using {i}th parameters\n")
#         print(f"\nFinding average maximum estimates for the {k}th realization\n")
#         point_of_interest = np.argmin(tmp_array1[k, :])
#         tmp_array2[k, 0] = testing_parameter_space[point_of_interest, 0]
#         tmp_array2[k, 1] = testing_parameter_space[point_of_interest, 1]
#         print(f"\nFound average maximum estimates for the {k}th realization\n")
#     print(f"\nSaving average maximum estimates for the {l}th sample\n")
#     mle_estimates_30[l, 0] = np.average(tmp_array2[:, 0])
#     mle_estimates_30[l, 1] = np.average(tmp_array2[:, 1])
#     print(f"\nSaved average maximum estimates for the {l}th sample\n")

# ------- SAVE MLE PRED FOR THIRTY REALIZATIONS ------- #

with open("./tf_stat_reproduction/ML30/preds_ML30.npy", mode="wb") as file:
    np.save(file, mle_estimates_30)

print("\nScript has successfully ended")

