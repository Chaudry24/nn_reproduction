import numpy as np
import tensorflow as tf
import some_file_1
import sys
print("apple")
# SPATIAL GRID
x = np.linspace(1, 16, 16)
y = x
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
spatial_grid = np.array([x, y]).T
print("apple1")
# COMPUTE DISTANCE MATRIX
spatial_distance = some_file_1.Spatial.compute_distance(spatial_grid[:, 0], spatial_grid[:, 1])
print("apple3")
# set number of epochs
n_epochs = 1000

# # open train cholesky matrices
# with open("../npy/chol_mats_train_tf_stat.npy", mode="rb") as file:
#     chol_mats_train = np.load(file)
print("apple4")
# open parameter space for training
with open("../npy/training_201_200_y.npy", mode="rb") as file:
    training_parameter_space = np.load(file)
print("apple5")
# GENERATE COVARIANCE MATRICES FOR TRAINING SET
# compute the covariance matrices
cov_mats_train = np.array([some_file_1.Spatial.compute_covariance
                           (covariance_type="matern", distance_matrix=spatial_distance,
                            variance=1.0, smoothness=1.0,
                            spatial_range=training_parameter_space[i, 1],
                            nugget=np.exp(training_parameter_space[i, 0]))
                           for i in range(training_parameter_space.shape[0])])

print("apple6")
# compute cholesky matrices for training
chol_mats_train = np.linalg.cholesky(cov_mats_train)

# free-up space
del cov_mats_train

# convert to tf
print(f"converting train space to tf tensor")
training_parameter_space = tf.convert_to_tensor(training_parameter_space)

# open test data
#with open("../npy/test30_tf_stat.npy", mode="rb") as file:
#    observations_test_30 = np.load(file)
print("apple7")
# NN architecture
model_NF30 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=10, input_shape=(16, 16, 30), activation="relu",
                           data_format="channels_last"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation="relu"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=500, activation="relu"),
    tf.keras.layers.Dense(units=2)
])

# compile NN
model_NF30.compile(optimizer=tf.optimizers.Adam(),
                   loss=tf.losses.MeanAbsoluteError(),
                   metrics=[tf.metrics.RootMeanSquaredError()])
print("apple8")
# empty list to store losses
loss_NF30 = []

# print sizes of vars before for loop
print("\nsizes before for loop\n")
print(f"model_size: {sys.getsizeof(model_NF30)*1e-9}GB")
print(f"size of training parameter space {training_parameter_space.numpy().nbytes*1e-9}GB")
print(f"size of chol mats: {chol_mats_train.nbytes*1e-9}GB")
print(f"size of loss list: {sys.getsizeof(loss_NF30)*1e-9}GB")
print(f"size of x: {x.nbytes*1e-9}GB")
print(f"size of y: {y.nbytes*1e-9}GB")
print(f"size of spatial grid: {spatial_grid.nbytes*1e-9}GB")
print(f"size of spatial dist: {spatial_distance.nbytes*1e-9}GB")
print(f"size of int epoch: {sys.getsizeof(n_epochs)*1e-9}GB")

# ------- TRAIN DIFFERENT NNs ------- #

for i in range(n_epochs):

    # print start of epoch
    print(f"starting iteration {i}")

    # generate data at every 5th iteration
    if i % 41 == 0:
        print(f"generating data for {i}th time (mod 5)")

        # GENERATE OBSERVATIONS FOR TRAINING FOR THIRTY REALIZATIONS
        z = np.random.randn(training_parameter_space.shape[0], 256, 30)
        observations_train_30 = (chol_mats_train @ z).reshape(training_parameter_space.shape[0], 16, 16, 30)
        # print size for debugging
        print("\nsize of observations and stuff in the if loop:\n")
        print(f"size of z: {z.nbytes*1e-9}GB at {i}th iteration")
        print(f"size of obs: {observations_train_30.nbytes*1e-9}GB at {i}th iteration")
        print("the training observations are being converted to tf tensor")
        observations_train_30 = tf.convert_to_tensor(observations_train_30)
        del z

    history_NF30 = model_NF30.fit(x=observations_train_30,
                                  y=training_parameter_space, batch_size=16,
                                  epochs=20)

    # store losses for each "epoch"
    loss_NF30.append(history_NF30.history["loss"])

    # print end of epoch
    print(f"iteration {i} ended")

    # print the sizes of different variables  
    print("\nsize of observations and stuff in the for loop:\n") 
    print(f"size of obs: {observations_train_30.numpy().nbytes*1e-9}GB at {i}th iteration")
    print(f"model_size: {sys.getsizeof(model_NF30)*1e-9}GB")
    print(f"history_size: {sys.getsizeof(history_NF30)*1e-9}GB at {i}th iteration")
    print(f"size of training parameter space {training_parameter_space.numpy().nbytes*1e-9}GB at {i}th iteration")
    print(f"size of chol mats: {chol_mats_train.nbytes*1e-9}GB at {i}th iteration")
    print(f"size of loss list: {sys.getsizeof(loss_NF30)*1e-9}GB at {i}th iteration")
    print(f"size of x: {x.nbytes*1e-9}GB at {i}th iteration")
    print(f"size of y: {y.nbytes*1e-9}GB at {i}th iteration")
    print(f"size of spatial grid: {spatial_grid.nbytes*1e-9}GB at {i}th iteration")
    print(f"size of spatial dist: {spatial_distance.nbytes*1e-9}GB at {i}th iteration")
    print(f"size of int epoch: {sys.getsizeof(n_epochs)*1e-9}GB at {i}th iteration")


# ------- SAVE TRAINING LOSS FOR NN ------- #

loss_NF30 = np.array([nums for lists in loss_NF30 for nums in lists])

#with open("NF30/training_loss_NF30.npy", mode="wb") as file:
#    np.save(file, loss_NF30)

# ------- GET PREDICTIONS FOR NN ------- #

# free-up space
del chol_mats_train, observations_train_30

# open test data
with open("../npy/test30_tf_stat.npy", mode="rb") as file:
    observations_test_30 = tf.convert_to_tensor(np.load(file))

preds_NF30 = model_NF30.predict(x=observations_test_30)

print(f"size of test obs: {observations_test_30.numpy().nbytes * 1e-9}GB")
print(f"size of preds: {preds_NF30.nbytes*1e-9}GB")

# ------- SAVE PREDICTIONS FOR NN ------- #

#with open("NF30/preds_NF30.npy", mode="wb") as file:
#    np.save(file, preds_NF30)

# ------- SAVE TRAINED NN ------- #

#model_NF30.save(filepath="./NF30")
