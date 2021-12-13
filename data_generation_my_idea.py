import some_file_1
import numpy as np

# SPATIAL GRID
x = np.linspace(1, 16, 16)
y = x
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
spatial_grid = np.array([x, y]).T

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
n_test = 500
testing_log_nugget_dom = (-8.90021914537373, 0.5544872141687432)
testing_spatial_range_dom = (2.0, 25.0)
testing_log_nugget_vals = np.random.uniform(testing_log_nugget_dom[0], testing_log_nugget_dom[1], n_test)
testing_spatial_range_vals = np.random.uniform(testing_spatial_range_dom[0], testing_spatial_range_dom[1], n_test)
testing_parameter_space = np.column_stack((testing_log_nugget_vals, testing_spatial_range_vals))
with open("npy/test_subset_my_idea.npy", mode="wb") as file:
    np.save(file, testing_parameter_space)

# GENERATE COVARIANCE MATRICES FOR TRAINING SET
# # compute the covariance matrices
# cov_mats_train = np.array([some_file_1.Spatial.compute_covariance
#                           (covariance_type="matern", distance_matrix=spatial_distance,
#                            variance=1.0, smoothness=1.0,
#                            spatial_range=training_parameter_space[i, 1],
#                            nugget=np.exp(training_parameter_space[i, 0]))
#                           for i in range(training_parameter_space.shape[0])])
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
observations_test = (chol_mats_test @ np.random.randn(cov_mats_test.shape[0], cov_mats_test.shape[1], 1)).reshape(cov_mats_test.shape[0], 16, 16, 1)
semi_variogram_test = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid,
                                                                          observations_test[i, :].reshape(256, 1),
                                                                          realizations=1, bins=10).ravel() for i in
                                range(testing_parameter_space.shape[0])])

# GENERATE OBSERVATIONS FOR TESTING FOR THIRTY REALIZATIONS
observations_test_30 = (chol_mats_test @ np.random.randn(testing_parameter_space.shape[0], 256, 30)).reshape(testing_parameter_space.shape[0], 16, 16, 30)
semi_variogram_test_30 = np.array([some_file_1.Spatial.compute_semivariogram(spatial_grid, observations_test_30[i, :, :, j].reshape(256, -1)).ravel()
                                   for i in range(testing_parameter_space.shape[0])
                                   for j in range(30)]).reshape(testing_parameter_space.shape[0], -1)

# # save train covariance matrices and train cholesky matrices
# with open("npy/cov_mats_train_my_idea.npy", mode="wb") as file:
#     np.save(file, cov_mats_train)
# with open("npy/chol_mats_train_my_idea.npy", mode="wb") as file:
#     np.save(file, chol_mats_train)

# save test covariance matrices and test cholesky matrices
with open("npy/cov_mats_test_my_idea.npy", mode="wb") as file:
    np.save(file, cov_mats_test)
with open("npy/chol_mats_test_my_idea.npy", mode="wb") as file:
    np.save(file, chol_mats_test)

# save test observation and test semivariogram for 1 realizations
with open("npy/test_my_idea.npy", mode="wb") as file:
    np.save(file, observations_test)
with open("npy/test_semivariogram_my_idea.npy", mode="wb") as file:
    np.save(file, semi_variogram_test)

# save test observation and test semivariogram for 30 realizations
with open("npy/test30_my_idea.npy", mode="wb") as file:
    np.save(file, observations_test_30)
with open("npy/test30_semivariogram_my_idea.npy", mode="wb") as file:
    np.save(file, semi_variogram_test_30)

