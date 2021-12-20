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

# convert train parameters to tensor
training_parameter_space = tf.convert_to_tensor(training_parameter_space)

# open test data
with open("../npy/test_my_idea.npy", mode="rb") as file:
    observations_test = np.load(file)

# NN architecture
model_NF = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=10, input_shape=(16, 16, 1), activation="relu",
                           data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation="relu"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=500, activation="relu"),
    tf.keras.layers.Dense(units=2)
])

# compile NN
model_NF.compile(optimizer=tf.optimizers.Adam(),
                 loss=tf.losses.MeanAbsoluteError(),
                 metrics=[tf.metrics.RootMeanSquaredError()])

# empty list to store losses
loss_NF = []

# ------- TRAIN DIFFERENT NNs ------- #

for i in range(n_epochs):

    # print start of epoch
    print(f"starting iteration {i}")

    # generate data at every 5th iteration
    N = 20
    if i % N == 0:
        print(f"generating data for {i}th time (mod {N})")

        # GENERATE OBSERVATIONS FOR TRAINING FOR A SINGLE REALIZATION
        observations_train = (chol_mats_train @ np.random.randn(training_parameter_space.shape[0], 256, 1)).reshape(training_parameter_space.shape[0], 16, 16, 1)
        # convert observations to tensor
        observations_train = tf.convert_to_tensor(observations_train)

    history_NF = model_NF.fit(x=observations_train,
                              y=training_parameter_space, batch_size=16,
                              epochs=3)

    # store losses for each "epoch"
    loss_NF.append(history_NF.history["loss"])

    # print end of epoch
    print(f"iteration {i} ended")

# ------- SAVE TRAINING LOSS FOR NN ------- #

loss_NF = np.array([nums for lists in loss_NF for nums in lists])

with open("NF/training_loss_NF.npy", mode="wb") as file:
    np.save(file, loss_NF)

# ------- GET PREDICTIONS FOR NN ------- #

preds_NF = model_NF.predict(x=tf.convert_to_tensor(observations_test))

# ------- SAVE PREDICTIONS FOR NN ------- #

with open("NF/preds_NF.npy", mode="wb") as file:
    np.save(file, preds_NF)

# ------- SAVE TRAINED NN ------- #

model_NF.save(filepath="./NF")
