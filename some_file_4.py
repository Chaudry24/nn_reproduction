import numpy as np
import some_file_1

bla bla
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
                                                                     nugget=training_parameter_space[i, 0])

# GENERATE COVARIANCE MATRICES FOR TESTING SET
cov_mats_test = np.empty([256, 256, testing_parameter_space.shape[0]])
for i in range(testing_parameter_space.shape[0]):
    print(f"generating testing covariance matrix for {i}th value")
    cov_mats_test[:, :, i] = some_file_1.Spatial.compute_covariance(covariance_type="matern",
                                                                    distance_matrix=spatial_distance,
                                                                    variance=1.0, smoothness=1.0,
                                                                    spatial_range=testing_parameter_space[i, 1],
                                                                    nugget=testing_parameter_space[i, 0])


# GENERATE OBSERVATIONS FOR TRAINING
observations_train = np.empty([training_parameter_space.shape[0], 32, 32, 1])
for i in range(training_parameter_space.shape[0]):
    print(f"generating training data for the {i}th covariance matrix")
    tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i]).reshape(32, 32)
    observations_train[i, :, :, :] = tmp_array

# GENERATE OBSERVATIONS FOR TESTING
observations_test = np.empty([testing_parameter_space.shape[0], 32, 32, 1])
for i in range(testing_parameter_space.shape[0]):
    print(f"generating testing data for the {i}th covariance matrix")
    tmp_array = some_file_1.Spatial.observations(realizations=1, covariance=cov_mats_train[:, :, i]).reshape(32, 32)
    observations_test[i, :, :, :] = tmp_array



testing_data = some_file_1.Spatial.observations(realizations=1)

