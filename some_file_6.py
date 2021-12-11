import numpy as np
import tensorflow as tf
import some_file_1
# import dask
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
spatial_distance = some_file_1.Spatial.compute_distance(spatial_grid[:, 0], spatial_grid[:, 1])

# GENERATE PARAMETER SPACE FOR TRAINING
n_train = 40200
training_log_nugget_dom = (-14.159006997232309, 5.541257432749877)
training_spatial_range_dom = (0.2, 50.0)
training_log_nugget_vals = np.random.uniform(training_log_nugget_dom[0], training_log_nugget_dom[1], n_train)
training_spatial_range_vals = np.random.uniform(training_spatial_range_dom[0], training_spatial_range_dom[1], n_train)
training_parameter_space = np.column_stack((training_log_nugget_vals, training_spatial_range_vals))
with open("npy/training_my_idea.npy", mode="wb") as file:
    np.save(file, training_parameter_space)

# GENERATE PARAMETER SPACE FOR TESTING
n_test = 50
testing_log_nugget_dom = (-8.90021914537373, 0.5544872141687432)
testing_spatial_range_dom = (2.0, 25.0)
testing_log_nugget_vals = np.random.uniform(testing_log_nugget_dom[0], testing_log_nugget_dom[1], n_test)
testing_spatial_range_vals = np.random.uniform(testing_spatial_range_dom[0], testing_spatial_range_dom[1], n_test)
testing_parameter_space = np.column_stack((testing_log_nugget_vals, testing_spatial_range_vals))
with open("npy/test_my_idea.npy", mode="wb") as file:
    np.save(file, testing_parameter_space)

# GENERATE COVARIANCE MATRICES FOR TRAINING SET
# compute the covariance matrices
cov_mats_train = np.array([some_file_1.Spatial.compute_covariance
                          (covariance_type="matern", distance_matrix=spatial_distance,
                           variance=1.0, smoothness=1.0,
                           spatial_range=training_parameter_space[i, 1],
                           nugget=np.exp(training_parameter_space[i, 0]))
                          for i in range(training_parameter_space.shape[0])])

# GENERATE COVARIANCE MATRICES FOR TESTING SET
# compute the covariance matrices
cov_mats_test = np.array([some_file_1.Spatial.compute_covariance
                         (covariance_type="matern", distance_matrix=spatial_distance,
                          variance=1.0, smoothness=1.0,
                          spatial_range=testing_parameter_space[i, 1],
                          nugget=np.exp(testing_parameter_space[i, 0]))
                         for i in range(testing_parameter_space.shape[0])])

# GENERATE OBSERVATIONS FOR TESTING FOR A SINGLE REALIZATION
observations_test = (cov_mats_test @ np.random.randn(cov_mats_test.shape[0], cov_mats_test.shape[1], 1)).reshape(cov_mats_test.shape[0], 16, 16, 1)
semi_variogram_test = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                          observations_test[i, :].reshape(256, 1),
                                                                          realizations=1, bins=10).ravel() for i in
                                range(testing_parameter_space.shape[0])])


# GENERATE OBSERVATIONS FOR TESTING FOR THIRTY REALIZATIONS
observations_test_30 = (cov_mats_test @ np.random.randn(testing_parameter_space.shape[0], 256, 30)).reshape(testing_parameter_space.shape[0], 16, 16, 30)
semi_variogram_test_30 = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid, observations_test_30[i, :, :, j].reshape(256, -1)).ravel()
                                   for i in range(testing_parameter_space.shape[0])
                                   for j in range(30)]).reshape(testing_parameter_space.shape[0], -1)


def negative_log_likelihood(variance, spatial_range, smoothness, nugget,
                            covariance_mat, observations, n_points=256):

    # ravel observations
    observations = observations.ravel()
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

# empty lists to store losses

loss_NF = []
loss_NF30 = []
loss_NV = []
loss_NV30 = []


# ------- TRAIN DIFFERENT NNs ------- #

for i in range(n_epochs):

    # print start of epoch
    print(f"starting iteration {i}")

    # generate data at every 5th iteration
    if i % 5 == 0:
        print(f"generating data for {i}th time (mod 5)")
        # GENERATE OBSERVATIONS FOR TRAINING FOR A SINGLE REALIZATION
        observations_train = (cov_mats_train @ np.random.randn(cov_mats_train.shape[0], cov_mats_train.shape[1], 1)).reshape(cov_mats_train.shape[0], 16, 16, 1)
        semi_variogram_train = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                                   observations_train[i, :].reshape(256, 1),
                                                                                   realizations=1, bins=10).ravel() for i in
                                         range(training_parameter_space.shape[0])])

        # GENERATE OBSERVATIONS FOR TRAINING FOR THIRTY REALIZATIONS
        observations_train_30 = (cov_mats_train @ np.random.randn(training_parameter_space.shape[0], 256, 30)).reshape(
            training_parameter_space.shape[0], 16, 16, 30)
        semi_variogram_train_30 = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                                      observations_train_30[i, :, :, j].reshape(256, -1)).ravel()
                                           for i in range(training_parameter_space.shape[0]) for j in
                                           range(30)]).reshape(training_parameter_space.shape[0], -1)

    print(f"fitting NF model for {i}th time")
    history_NF = model_NF.fit(x=tf.convert_to_tensor(observations_train),
                              y=tf.convert_to_tensor(training_parameter_space), batch_size=16,
                              epochs=3)

    print(f"fitting NF30 model for {i}th time")
    history_NF30 = model_NF30.fit(x=tf.convert_to_tensor(observations_train_30),
                                  y=tf.convert_to_tensor(training_parameter_space), batch_size=16,
                                  epochs=3)

    print(f"fitting NV model for {i}th time")
    history_NV = model_NV.fit(x=tf.convert_to_tensor(semi_variogram_train),
                              y=tf.convert_to_tensor(training_parameter_space), batch_size=16,
                              epochs=3)

    print(f"fitting NV30 model for {i}th time")
    history_NV30 = model_NV30.fit(x=tf.convert_to_tensor(semi_variogram_train_30),
                                  y=tf.convert_to_tensor(training_parameter_space), batch_size=16,
                                  epochs=3)

    # store losses for each "epoch"
    loss_NF.append(history_NF.history["loss"])
    loss_NF30.append(history_NF30.history["loss"])
    loss_NV30.append(history_NV.history["loss"])
    loss_NV30.append(history_NV30.history["loss"])

    # print end of epoch
    print(f"iteration {i} ended")

# ------- SAVE TRAINING LOSS FOR EACH NN ------- #

loss_NF = np.array([nums for lists in loss_NF for nums in lists])
loss_NF30 = np.array([nums for lists in loss_NF30 for nums in lists])
loss_NV = np.array([nums for lists in loss_NV for nums in lists])
loss_NV30 = np.array([nums for lists in loss_NV30 for nums in lists])

with open("./my_idea/NF/training_loss_NF.npy", mode="wb") as file:
    np.save(file, loss_NF)

with open("./my_idea/NF30/training_loss_NF30.npy", mode="wb") as file:
    np.save(file, loss_NF30)

with open("./my_idea/NV/training_loss_NV.npy", mode="wb") as file:
    np.save(file, loss_NV)

with open("./my_idea/NV30/training_loss_NV30.npy", mode="wb") as file:
    np.save(file, loss_NV30)

# ------- GET PREDICTIONS FOR EACH NN ------- #

preds_NF = model_NF.predict(x=tf.convert_to_tensor(observations_test))

preds_NF30 = model_NF30.predict(x=tf.convert_to_tensor(observations_test_30))

preds_NV = model_NV.predict(x=tf.convert_to_tensor(semi_variogram_test))

preds_NV30 = model_NV30.predict(x=tf.convert_to_tensor(semi_variogram_test_30))

# ------- SAVE PREDICTIONS FOR EACH NN ------- #

with open("./my_idea/NF/preds_NF.npy", mode="wb") as file:
    np.save(file, preds_NF)

with open("./my_idea/NF30/preds_NF30.npy", mode="wb") as file:
    np.save(file, preds_NF30)

with open("./my_idea/NV/preds_NV.npy", mode="wb") as file:
    np.save(file, preds_NV)

with open("./my_idea/NV30/preds_NV30.npy", mode="wb") as file:
    np.save(file, preds_NV30)

# ------- SAVE TRAINED NNs ------- #

model_NF.save(filepath="./my_idea/NF")
model_NF30.save(filepath="./my_idea/NF30")
model_NV.save(filepath="./my_idea/NV")
model_NV30.save(filepath="./my_idea/NV30")

# ------- COMPUTE MLE FOR A SINGLE REALIZATION ------- #


mle_estimates = np.empty([testing_parameter_space.shape[0], 2])
tmp_array = np.array([negative_log_likelihood
                      (variance=1.0, spatial_range=testing_parameter_space[j, 1],
                       smoothness=1.0, nugget=np.exp(testing_parameter_space[j, 0]),
                       covariance_mat=cov_mats_test[j, :, :],
                       observations=observations_test[i, :].reshape(256)) for i in range(observations_test.shape[0])
                      for j in range(testing_parameter_space.shape[0])])
for i in range(observations_test.shape[0]):
    tmp1 = np.array([tmp_array[j] for j in range(i * testing_parameter_space.shape[0], (i + 1) * testing_parameter_space.shape[0])])
    mle_estimates[i, 0] = testing_parameter_space[np.argmin(tmp1), 0]
    mle_estimates[i, 1] = testing_parameter_space[np.argmin(tmp1), 1]

# ------- SAVE MLE PRED FOR A SINGLE REALIZATION ------- #

with open("./my_idea/ML/preds_MLE.npy", mode="wb") as file:
    np.save(file, mle_estimates)

# ------- COMPUTE MLE FOR THIRTY REALIZATIONS ------- #

# delete tmp_array before next use
del tmp_array
del tmp1

mle_estimates_30 = np.empty([testing_parameter_space.shape[0], 2])
tmp_array1 = np.empty([30, testing_parameter_space.shape[0]])
tmp_array2 = np.empty([30, 2])

for l in range(observations_test_30.shape[0]):
    # compute MLE
    tmp1 = np.array([negative_log_likelihood
                     (variance=1.0, spatial_range=testing_parameter_space[i, 1],
                      smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 0]),
                      covariance_mat=cov_mats_test[i, :, :],
                      observations=observations_test_30[l, :, :, j].reshape(256))
                     for i in range(testing_parameter_space.shape[0])
                     for j in range(30)])
    # tmp2 = np.array([tmp1[i].compute() for i in range(30 * testing_parameter_space.shape[0])])
    tmp_list = []
    tmp_list2 = []
    for i in range(30):
        for j in range(testing_parameter_space.shape[0]):
            # save MLE estimate indices for parameter j and realization i
            tmp_list.append(i + j * 30)
        # get MLE estimates for the above parameters
        tmp3 = tmp1[tmp_list]
        # empty out the tmp_list
        tmp_list = []
        # save the index of the minimum parameter for i-th realization
        tmp_list2.append(np.argmin(tmp3))
    # store parameter values at the point of interest for each realization
    tmp4 = testing_parameter_space[tmp_list2, :]    # save the averages
    mle_estimates_30[l, 0] = np.average(tmp4[:, 0])
    mle_estimates_30[l, 1] = np.average(tmp4[:, 1])

# ------- SAVE MLE PRED FOR THIRTY REALIZATIONS ------- #

with open("./my_idea/ML30/preds_ML30.npy", mode="wb") as file:
    np.save(file, mle_estimates_30)

print("\nScript has successfully ended")


