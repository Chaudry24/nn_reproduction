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
cov_mats_train = np.empty([256, 256, training_parameter_space.shape[0]])
for i in range(training_parameter_space.shape[0]):
    print(f"\ngenerating training covariance matrix for {i}th value\n")
    cov_mats_train[:, :, i] = some_file_1.Spatial.compute_covariance(covariance_type="matern",
                                                                     distance_matrix=spatial_distance,
                                                                     variance=1.0, smoothness=1.0,
                                                                     spatial_range=training_parameter_space[i, 1],
                                                                     nugget=np.exp(training_parameter_space[i, 0]))
    print(f"\ntraining covariance matrix for {i}th value generated\n")

# GENERATE COVARIANCE MATRICES FOR TESTING SET
cov_mats_test = np.empty([256, 256, testing_parameter_space.shape[0]])
for i in range(testing_parameter_space.shape[0]):
    print(f"\ngenerating testing covariance matrix for {i}th value\n")
    cov_mats_test[:, :, i] = some_file_1.Spatial.compute_covariance(covariance_type="matern",
                                                                    distance_matrix=spatial_distance,
                                                                    variance=1.0, smoothness=1.0,
                                                                    spatial_range=testing_parameter_space[i, 1],
                                                                    nugget=np.exp(testing_parameter_space[i, 0]))
    print(f"\ntesting covariance matrix for {i}th value generated\n")


# GENERATE OBSERVATIONS FOR TRAINING FOR A SINGLE REALIZATION
observations_train = np.empty([training_parameter_space.shape[0], 16, 16, 1])
semi_variogram_train = np.empty([training_parameter_space.shape[0], 10])
for i in range(training_parameter_space.shape[0]):
    print(f"\ngenerating training data for the {i}th covariance matrix\n")
    tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i])
    tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1, bins=10)
    semi_variogram_train[i, :] = tmp_var.ravel()
    observations_train[i, :, :, :] = tmp_array.reshape(16, 16, 1)
    print(f"\ntraining data generated for the {i}th covariance matrix\n")

# GENERATE OBSERVATIONS FOR TESTING FOR A SINGLE REALIZATION
observations_test = np.empty([testing_parameter_space.shape[0], 16, 16, 1])
semi_variogram_test = np.empty([testing_parameter_space.shape[0], 10])
for i in range(testing_parameter_space.shape[0]):
    print(f"\ngenerating testing data for the {i}th covariance matrix\n")
    tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_test[:, :, i])
    tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1, bins=10)
    semi_variogram_test[i, :] = tmp_var.ravel()
    observations_test[i, :, :, :] = tmp_array.reshape(16, 16, 1)
    print(f"\ntesting data generated for the {i}th covariance matrix\n")

# GENERATE OBSERVATIONS FOR TRAINING FOR THIRTY REALIZATIONS
observations_train_30 = np.empty([training_parameter_space.shape[0], 16, 16, 30])
semi_variogram_train_30 = np.empty([training_parameter_space.shape[0], 10, 30])
for i in range(training_parameter_space.shape[0]):
    print(f"\ngenerating training data for the {i}th covariance matrix\n")
    for j in range(30):
        print(f"\ngetting {j}th training realization from {i}th covariance matrix\n")
        tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i])
        tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1)
        semi_variogram_train_30[i, :, j] = tmp_var.ravel()
        observations_train_30[i, :, :, j] = tmp_array.reshape(16, 16)
        print(f"\n{j}th training realization from {i}th covariance matrix generated\n")

# GENERATE OBSERVATIONS FOR TESTING FOR THIRTY REALIZATIONS
observations_test_30 = np.empty([testing_parameter_space.shape[0], 16, 16, 30])
semi_variogram_test_30 = np.empty([testing_parameter_space.shape[0], 10, 30])
for i in range(testing_parameter_space.shape[0]):
    print(f"\ngenerating testing data for the {i}th covariance matrix\n")
    for j in range(30):
        print(f"\ngetting {j}th testing realization from {i}th covariance matrix\n")
        tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_test[:, :, i])
        tmp_var = some_file_1.Spatial.compute_semivariogram(spatial_grid, tmp_array, realizations=1)
        semi_variogram_test_30[i, :, j] = tmp_var.ravel()
        observations_test_30[i, :, :, j] = tmp_array.reshape(16, 16)
        print(f"\n{j}th testing realization from {i}th covariance matrix generated\n")


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
    tf.keras.layers.Dense(units=2, activity_regularizer=tf.keras.regularizers.l1())
])

model_NF30 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=10, input_shape=(16, 16, 30), activation="relu",
                           data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation="relu"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=500, activation="relu"),
    tf.keras.layers.Dense(units=2, activity_regularizer=tf.keras.regularizers.l1())
])

model_NV = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3000, activation="relu"),
    tf.keras.layers.Dense(units=1000),
    tf.keras.layers.Dense(units=2, activity_regularizer=tf.keras.regularizers.l1())
])

model_NV30 = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=3000, activation="relu"),
    tf.keras.layers.Dense(units=1000),
    tf.keras.layers.Dense(units=2, activity_regularizer=tf.keras.regularizers.l1())
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

history_NF = model_NF.fit(x=observations_train, y=training_parameter_space, batch_size=16,
                          epochs=n_epochs)

history_NF30 = model_NF30.fit(x=observations_train_30, y=training_parameter_space, batch_size=16,
                              epochs=n_epochs)

history_NV = model_NV.fit(x=semi_variogram_train, y=training_parameter_space, batch_size=16,
                          epochs=n_epochs)

history_NV30 = model_NV30.fit(x=semi_variogram_train_30, y=training_parameter_space, batch_size=16,
                              epochs=n_epochs)

# ------- SAVE TRAINING LOSS FOR EACH NN ------- #

with open("./my_idea/NF/training_loss_NF.npy", mode="wb") as file:
    np.save(file, history_NF.history["loss"])

with open("./my_idea/NF30/training_loss_NF30.npy", mode="wb") as file:
    np.save(file, history_NF30.history["loss"])

with open("./my_idea/NV/training_loss_NV.npy", mode="wb") as file:
    np.save(file, history_NV.history["loss"])

with open("./my_idea/NV30/training_loss_NV30.npy", mode="wb") as file:
    np.save(file, history_NV30.history["loss"])

# ------- GET PREDICTIONS FOR EACH NN ------- #

preds_NF = model_NF.predict(x=observations_test)

preds_NF30 = model_NF30.predict(x=observations_test_30)

preds_NV = model_NV.predict(x=semi_variogram_test)

preds_NV30 = model_NV30.predict(x=semi_variogram_test_30)

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
tmp_array = np.empty([observations_test.shape[0], testing_parameter_space.shape[0]])
# for loop for each sample
for i in range(observations_test.shape[0]):
    # for loop for each parameter value
    for j in range(testing_parameter_space.shape[0]):
        print(f"\nStarting MLE for {i}th observation using {j}th parameters\n")
        tmp_array[i, j] = negative_log_likelihood(variance=1.0, spatial_range=testing_parameter_space[j, 1],
                                                  smoothness=1.0, nugget=np.exp(testing_parameter_space[j, 0]),
                                                  covariance_mat=cov_mats_test[:, :, j],
                                                  observations=observations_test[i, :].reshape(256))
        print(f"\nThe MLE for {i}th observation using {j}th parameters is {tmp_array[i, j]}\n")
        print(f"\nEnded MLE for {i}th observation using {j}th parameters\n")
    print(f"\nSaving the maximum estimates for the {i}th sample\n")
    mle_estimates[i, 0] = testing_parameter_space[np.argmin(tmp_array[i, :]), 0]
    mle_estimates[i, 1] = testing_parameter_space[np.argmin(tmp_array[i, :]), 1]
    print(f"\nSaved the maximum estimates for the {i}th sample\n")

# ------- SAVE MLE PRED FOR A SINGLE REALIZATION ------- #

with open("./my_idea/ML/preds_MLE.npy", mode="wb") as file:
    np.save(file, mle_estimates)

# ------- COMPUTE MLE FOR THIRTY REALIZATIONS ------- #

mle_estimates_30 = np.empty([testing_parameter_space.shape[0], 2])
tmp_array1 = np.empty([30, testing_parameter_space.shape[0]])
tmp_array2 = np.empty([30, 2])

# for loop for each sample
for l in range(observations_test_30.shape[0]):
    # for loop to save min parameters for each realization
    for k in range(30):
        # for loop for each parameter value
        for i in range(testing_parameter_space.shape[0]):
            # for loop for each realization
            for j in range(30):
                print(f"\nStarting MLE for {l}th sample {j}th realization using {i}th parameters\n")
                tmp_array1[j, i] = negative_log_likelihood(variance=1.0, spatial_range=testing_parameter_space[i, 1],
                                                           smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 0]),
                                                           covariance_mat=cov_mats_test[:, :, i],
                                                           observations=observations_test_30[l, :, :, j].reshape(256))
                print(f"\nThe MLE for {l}th sample {j}th realization using {i}th parameters is {tmp_array1[j, i]}\n")
                print(f"\nEnded MLE for {l}th sample {j}th realization using {i}th parameters\n")
        print(f"\nFinding average maximum estimates for the {k}th realization\n")
        point_of_interest = np.argmin(tmp_array1[k, :])
        tmp_array2[k, 0] = testing_parameter_space[point_of_interest, 0]
        tmp_array2[k, 1] = testing_parameter_space[point_of_interest, 1]
        print(f"\nFound average maximum estimates for the {k}th realization\n")
    print(f"\nSaving average maximum estimates for the {l}th sample\n")
    mle_estimates_30[l, 0] = np.average(tmp_array2[:, 0])
    mle_estimates_30[l, 1] = np.average(tmp_array2[:, 1])
    print(f"\nSaved average maximum estimates for the {l}th sample\n")

# ------- SAVE MLE PRED FOR THIRTY REALIZATIONS ------- #

with open("./my_idea/ML30/preds_ML30.npy", mode="wb") as file:
    np.save(file, mle_estimates_30)
