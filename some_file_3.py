import tensorflow as tf
import some_file_2
import some_file_1
import numpy as np

# number of training and test samples
n_train_samples = 64
n_test_samples = 32
n_train_samples = 5
n_test_samples = 3
# number of realizations
realization1 = 1
realization2 = 30
# sets of testing data
total_testing_data = 60
total_testing_data = 1
# number of params estimated
n_params = 2
# number of epochs
n_epochs = 10000
n_epochs = 4
# batch size
batch_size = 16
# spatial coordinates
spatial_grid = some_file_1.Spatial().domain

# TOTAL TRAIN SAMPLES = n_train_samples * n_epochs
# TOTAL TESTING DATA = n_test_samples * total_testing_data


# @retry(Exception, tries=-1, delay=0, backoff=0)
def generate_training_data(realizations=1):
    # printing to debug
    print("\nTraining data is being generated\n")

    # generate training data
    some_file_2.save_data(file_name_data="training_data",
                          file_name_params="training_params",
                          n_samples=n_train_samples, realizations=realizations,
                          sample_spatial_range=True, sample_nugget=True)

    # load training data
    training_data = some_file_2.load_data("training_data")
    training_data = training_data.reshape((n_train_samples, 16, 16, realizations))
    training_params = some_file_2.load_data("training_params")
    training_data = tf.convert_to_tensor(training_data)
    training_params = tf.convert_to_tensor(training_params)

    # printing to debug
    print("\nTraining data has been generated")

    return training_data, training_params


# @retry(Exception, tries=-1, delay=0, backoff=0)
def generate_testing_data(realizations=1):
    # printing to debug
    print("\nTesting data is being generated\n")

    # generate and save testing data
    for i in range(total_testing_data):
        some_file_2.save_data(file_name_data=f"testing_data{i}_{realizations}",
                              file_name_params=f"testing_params{i}_{realizations}",
                              n_samples=n_test_samples, realizations=realizations,
                              sample_spatial_range=True, sample_nugget=True)

    # save memory for testing data
    testing_data = np.empty((n_test_samples * total_testing_data, 16, 16, realizations))
    testing_params = np.empty((n_test_samples * total_testing_data, 2))

    # load and save testing data
    for i in range(total_testing_data):
        # load testing data
        testing_data_temp = some_file_2.load_data(f"testing_data{i}_{realizations}", is_testing=True)
        # reshape testing data
        testing_data_temp = testing_data_temp.reshape((n_test_samples, 16, 16, realizations))
        # load testing parameters
        testing_params_temp = some_file_2.load_data(f"testing_params{i}_{realizations}", is_testing=True)
        # save testing data
        testing_data[i * (n_test_samples):
                     (i + 1) * (n_test_samples), :, :, :] = testing_data_temp
        # save testing parameters
        testing_params[i * (n_test_samples):
                       (i + 1) * (n_test_samples), :] = testing_params_temp

    # save testing data
    with open(f"./data/testing_data_{realizations}.npy", mode="wb") as file:
        np.save(file, testing_data)
    # save testing parameters
    with open(f"./data/testing_params_{realizations}.npy", mode="wb") as file:
        np.save(file, testing_params)

    # convert the saved data to tf tensor
    testing_data = tf.convert_to_tensor(testing_data)
    testing_params = tf.convert_to_tensor(testing_params)

    # printing to debug
    print("\nTesting data has been generated\n")

    return testing_data, testing_params


def run_simulation_field(testing_data, realizations=1):

    # make a NN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=10, input_shape=(16, 16, realizations), activation="relu",
                               data_format="channels_last"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=200, activation="relu"),
        tf.keras.layers.Dense(units=n_params,
                              activity_regularizer=tf.keras.regularizers.l1())
    ])

    # compile the NN
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.losses.MeanAbsoluteError(),
                  metrics=[tf.metrics.RootMeanSquaredError()])

    # a list to save the loss of the nn
    loss = []

    # train the model and generate new data at the end of every other epoch
    for i in range(n_epochs):

        # print start of epoch
        print(f"\nThis is the start of epoch: {i}\n")

        # load training data
        training_data, training_params = generate_training_data(realizations=realizations)

        # train the model for 2 epochs with same data
        history = model.fit(x=training_data, y=training_params, batch_size=batch_size,
                            epochs=2)

        # save the loss
        loss.append(history.history["loss"])

        # print end of epoch
        print(f"\nThis is the end of epoch: {i}\n")

    # save the trained model
    model.save(filepath=f"./trained_model_F{realizations}")

    # convert loss to numpy array
    loss = np.array([nums for lists in loss for nums in lists])

    # save the loss
    with open(f"./results/training_loss_F{realizations}.npy", mode="wb") as loss_info:
        np.save(loss_info, loss)

    # get NN predictions for test set (outputs a numpy array)
    preds = model.predict(x=testing_data)

    # save NN predictions
    with open(f"./results/predictions_NN_F{realizations}.npy", mode="wb") as file:
        np.save(file, preds)


def run_simulation_variogram(testing_data, realizations=1):

    # convert testing data to a numpy array
    testing_data = testing_data.numpy()

    # reshape testing data
    testing_data = testing_data.reshape(n_test_samples * total_testing_data, 256, realizations)

    # initialize memory to store variogram for each test sample
    testing_variogram = np.empty((n_test_samples * total_testing_data, 10, realizations))

    # compute the variogram for each sample
    for j in range(n_test_samples * total_testing_data):
        testing_variogram[j] = some_file_1.Spatial.compute_variogram(coordinates=spatial_grid,
                                                                     observations=testing_data[j],
                                                                     realizations=realizations)

    # convert into tensor
    testing_variogram = tf.convert_to_tensor(testing_variogram)

    # make a NN
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=3000, activation="relu"),
        tf.keras.layers.Dense(units=1000, activation="relu"),
        tf.keras.layers.Dense(units=n_params,
                              activity_regularizer=tf.keras.regularizers.l1())
    ])

    # compile the NN
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.losses.MeanAbsoluteError(),
                  metrics=[tf.metrics.RootMeanSquaredError()])

    # a list to save the loss of the nn
    loss = []

    # train the model and generate new data at the end of every other epoch
    for i in range(n_epochs):

        # print start of epoch
        print(f"\nThis is the start of epoch: {i}\n")

        # load training data
        training_data, training_params = generate_training_data(realizations=realizations)

        # convert training data to a numpy array
        training_data = training_data.numpy()

        # reshape training data
        training_data = training_data.reshape(n_train_samples, 256, realizations)

        # initialize memory to store variogram for each train sample
        training_variogram = np.empty((n_train_samples, 10, realizations))

        # compute the variogram for each sample
        for j in range(n_train_samples):
            training_variogram[j] = some_file_1.Spatial.compute_variogram(coordinates=spatial_grid,
                                                                          observations=training_data[j],
                                                                          realizations=realizations)

        # convert into tensor
        training_variogram = tf.convert_to_tensor(training_variogram)

        # train the model for 2 epochs with same data
        history = model.fit(x=training_variogram, y=training_params, batch_size=batch_size,
                            epochs=2)

        # save the loss
        loss.append(history.history["loss"])

        # print end of epoch
        print(f"\nThis is the end of epoch: {i}\n")

    # save the trained model
    model.save(filepath=f"./trained_model_VG{realizations}")

    # convert loss to numpy array
    loss = np.array([nums for lists in loss for nums in lists])

    # save the loss
    with open(f"./results/training_loss_VG{realizations}.npy", mode="wb") as loss_info:
        np.save(loss_info, loss)

    # get NN predictions for test set (outputs a numpy array)
    preds = model.predict(x=testing_variogram)

    # save NN predictions
    with open(f"./results/predictions_NN_VG{realizations}.npy", mode="wb") as file:
        np.save(file, preds)


def run_simulation_mle(testing_data, realizations=1):

    # generate some data to get the distance matrix
    dist_mat = some_file_1.Spatial().distance_matrix

    # use preds list to save the predictions of MLE
    preds = []

    # solve for parameters using MLE
    for i in range(n_test_samples * total_testing_data):
        
        # print the start of the MLE for current sample
        print(f"\nThis is the start of MLE on sample no: {i}\n")

        # a temporary list to save predictions for each realization
        tmp_array = []

        # for-loop for realizations
        for j in range(realizations):
            # get observations and convert to numpy array
            observations = testing_data[i, :, :, j].numpy().reshape((256, -1))

            # generate a model given the observations and the distance matrix
            model = some_file_1.Optimization(observations=observations,
                                             distance_matrix=dist_mat,
                                             estimate_spatial_range=True,
                                             estimate_nugget=True)

            # use maximum likelihood estimate to estimate the parameters
            results = model.optimize()

            # save the estimated spatial range and nugget in the temporary array for each realization
            tmp_array.append([results["spatial_range"], results["nugget"]])

        # save the average predicted spatial range and nugget by summing the rows of tmp_array
        avg_preds = np.average(tmp_array, axis=0).reshape(2)

        # save the estimated spatial range and smoothness in a list
        preds.append([avg_preds[0], avg_preds[1]])

        # print the end of the MLE for current sample
        print(f"\nThis is the end of MLE on sample no: {i}\n")

    # convert the MLE predictions into a numpy array
    preds = np.array(preds)

    # save the mle predictions
    if realizations > 1:
        with open(f"./results/predictions_MLE{realizations}.npy", mode="wb") as file:
            np.save(file, preds)
    else:
        with open("./results/predictions_MLE.npy", mode="wb") as file:
            np.save(file, preds)


# generate testing data for a single realization
print("\nAbout to generate test data\n")
testing_data, testing_params = generate_testing_data(realizations=realization1)
print("\nTest data has been generated\n")

# run simulation for different models
print("\nstarting field model\n")
run_simulation_field(realizations=realization1, testing_data=testing_data)
print("\nfield model ended\n")
print("\nstaring variogram model\n")
run_simulation_variogram(realizations=realization1, testing_data=testing_data)
print("\nvariogram model ended\n")
print("\nstarting mle\n")
run_simulation_mle(realizations=realization1, testing_data=testing_data)
print("\nmle ended\n")

# generate testing data for more than one realization
print("\nAbout to generate test data for more than one realization\n")
testing_data, testing_params = generate_testing_data(realizations=realization2)
print("\nTest data has been generated for more than one realization\n")

# run simulation for different models
print("\nstarting field model for more than one realization\n")
run_simulation_field(realizations=realization2, testing_data=testing_data)
print("\nfield model ended for more than one realization\n")
print("\nstarting variogram model for more than one realization\n")
run_simulation_variogram(realizations=realization2, testing_data=testing_data)
print("\nvariogram model ended for more than one realization\n")
print("\nstarting mle for more than one realization\n")
run_simulation_mle(realizations=realization2, testing_data=testing_data)
print("\nmle ended for more than one realization\n")


