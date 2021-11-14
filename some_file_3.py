import tensorflow as tf
from retry import retry
import some_file_2
import some_file_1
import numpy as np

# number of training and test samples
n_train_samples = 3
n_test_samples = 2
# number of params estimated
n_params = 2


@retry(Exception, tries=10, delay=0, backoff=0)
def generate_training_data():

    # generate training data
    some_file_2.save_data(file_name_data="training_data",
                          file_name_params="training_params",
                          n_samples=n_train_samples,
                          sample_spatial_range=True, sample_smoothness=True)

    # load training data
    training_data = some_file_2.load_data("training_data").T
    training_data = training_data.reshape((n_train_samples ** n_params, 32, 32, 1))
    training_params = some_file_2.load_data("training_params")
    training_data = tf.convert_to_tensor(training_data)
    training_params = tf.convert_to_tensor(training_params)

    return training_data, training_params


@retry(Exception, tries=10, delay=0, backoff=0)
def generate_testing_data():

    # generate and save testing data
    some_file_2.save_data(file_name_data="testing_data",
                          file_name_params="testing_params",
                          n_samples=n_test_samples, save_distance_mat=True,
                          sample_spatial_range=True, sample_smoothness=True)

    # load testing data
    testing_data = some_file_2.load_data("testing_data").T
    testing_data = testing_data.reshape((n_test_samples ** n_params, 32, 32, 1))
    testing_params = some_file_2.load_data("testing_params")
    testing_data = tf.convert_to_tensor(testing_data)
    testing_params = tf.convert_to_tensor(testing_params)

    return testing_data, testing_params


# load testing data
testing_data, testing_params = generate_testing_data()

# make a NN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=10, input_shape=(32, 32, 1), activation="relu",
                           data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu"),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=200, activation="relu"),
    tf.keras.layers.Dense(units=n_params)
])
# compile the NN
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.MeanAbsoluteError(),
              metrics=[tf.metrics.RootMeanSquaredError()])
# train the model and generate new data at the end of each epoch
loss = []
for i in range(1):
    # load training data
    training_data, training_params = generate_training_data()
    # train the model for 100 epochs
    history = model.fit(x=training_data, y=training_params, batch_size=5,
                        epochs=2)
    # save the loss
    loss.append(history.history["loss"])

# convert loss to numpy array
loss = np.array([nums for lists in loss for nums in lists])
# save the loss
with open("training_loss.npy", mode="wb") as loss_info:
    np.save(loss_info, loss)

# predictions output a numpy array
preds = model.predict(x=testing_data)
# save NN predictions
with open("predictions_NN.npy", mode="wb") as file:
    np.save(file, preds)

preds = []
for i in range(n_test_samples ** n_params):
    # TODO: small caveat: the distance matrix does not correspond to the actual distance matrix of the data since
    #  that was not saved get distance matrix but that should not make much or any difference
    dist_mat = some_file_1.Spatial().distance_matrix
    # get observations and convert to numpy array
    observations = testing_data[i, :].numpy().reshape((1024, -1))
    # generate a model given the observations and the distance
    model = some_file_1.Optimization(observations=observations,
                                     distance_matrix=dist_mat,
                                     estimate_spatial_range=True,
                                     estimate_smoothness=True)
    # use maximum likelihood estimate to estimate the parameters
    results = model.optimize()
    # save the estimated spatial range and smoothness in a list
    preds.append([results["spatial_range"], results["smoothness"]])

# convert the predictions into a numpy array
preds = np.array(preds)
# save the mle predictions
with open("predictions_MLE.npy", mode="wb") as file:
    np.save(file, preds)



