import numpy as np
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

# # LOAD PARAMETER SPACE FOR TRAINING
# with open("npy/training_201_200_y.npy", mode="rb") as file:
#     training_parameter_space = np.load(file)

# print as a progress update
# print("training parameters loaded")

# LOAD PARAMETER SPACE FOR TESTING
with open("npy/test_y.npy", mode="rb") as file:
    n_test_samples = 500
    testing_parameter_space = np.load(file)
    test_sample_indices = np.random.choice(testing_parameter_space.shape[0], n_test_samples, replace=False)
    testing_parameter_space = testing_parameter_space[test_sample_indices]

# SAVE TESTING PARAMETER SUBSET
with open("npy/test_subset.npy", mode="wb") as file:
    np.save(file, testing_parameter_space)

# print as a progress update
print("testing parameters loaded")

# GENERATE COVARIANCE MATRICES FOR TRAINING SET
# compute the covariance matrices
# cov_mats_train = np.array([some_file_1.Spatial.compute_covariance
#                            (covariance_type="matern", distance_matrix=spatial_distance,
#                             variance=1.0, smoothness=1.0,
#                             spatial_range=training_parameter_space[i, 1],
#                             nugget=np.exp(training_parameter_space[i, 0]))
#                            for i in range(training_parameter_space.shape[0])])
#
# # compute cholesky matrices for training
# chol_mats_train = np.linalg.cholesky(cov_mats_train)

# GENERATE COVARIANCE MATRICES FOR TESTING SET
# compute the covariance matrices
cov_mats_test = np.array([some_file_1.Spatial.compute_covariance
                          (covariance_type="matern", distance_matrix=spatial_distance,
                           variance=1.0, smoothness=1.0,
                           spatial_range=testing_parameter_space[i, 1],
                           nugget=np.exp(testing_parameter_space[i, 0]))
                          for i in range(testing_parameter_space.shape[0])])

# compute cholesky matrices for testing
chol_mats_test = np.linalg.cholesky(cov_mats_test)

# GENERATE OBSERVATIONS FOR TESTING FOR A SINGLE REALIZATION
observations_test = (chol_mats_test @ np.random.randn(cov_mats_test.shape[0], cov_mats_test.shape[1], 1)).reshape(
    cov_mats_test.shape[0], 16, 16, 1)
semi_variogram_test = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                          observations_test[i, :].reshape(256, 1),
                                                                          realizations=1, bins=10).ravel() for i in
                                range(testing_parameter_space.shape[0])])

# GENERATE OBSERVATIONS FOR TESTING FOR THIRTY REALIZATIONS
observations_test_30 = (chol_mats_test @ np.random.randn(testing_parameter_space.shape[0], 256, 30)).reshape(testing_parameter_space.shape[0], 16, 16, 30)
semi_variogram_test_30 = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid, observations_test_30[i, :, :, j].reshape(256, -1)).ravel()
                                   for i in range(testing_parameter_space.shape[0])
                                   for j in range(30)]).reshape(testing_parameter_space.shape[0], -1)

# save train covariance matrices and train cholesky matrices
# with open("npy/cov_mats_train_tf_stat.npy", mode="wb") as file:
#     np.save(file, cov_mats_train)
# with open("npy/chol_mats_train_tf_stat.npy", mode="wb") as file:
#     np.save(file, chol_mats_train)

# save test covariance matrices and test cholesky matrices
with open("npy/cov_mats_test_tf_stat.npy", mode="wb") as file:
    np.save(file, cov_mats_test)
with open("npy/chol_mats_test_tf_stat.npy", mode="wb") as file:
    np.save(file, chol_mats_test)

# save test observation and test semivariogram for 1 realization
with open("npy/test_tf_stat.npy", mode="wb") as file:
    np.save(file, observations_test)
with open("npy/test_semivariogram_tf_stat.npy", mode="wb") as file:
    np.save(file, semi_variogram_test)

# save test observation and test semivariogram for 30 realizaitons
with open("npy/test30_tf_stat.npy", mode="wb") as file:
    np.save(file, observations_test_30)
with open("npy/test30_semivariogram_tf_stat.npy", mode="wb") as file:
    np.save(file, semi_variogram_test_30)
