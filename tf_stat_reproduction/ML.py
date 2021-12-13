import numpy as np

# load testing parameter space
with open("../npy/test_subset.npy", mode="rb") as file:
    testing_parameter_space = np.load(file)

# load test observations
with open("../npy/test_tf_stat.npy", mode="rb") as file:
    observations_test = np.load(file)

# load testing covariance and cholesky matrices
with open("../npy/cov_mats_test_tf_stat.npy", mode="rb") as file:
    cov_mats_test = np.load(file)
with open("../npy/chol_mats_test_tf_stat.npy", mode="rb") as file:
    chol_mats_test = np.load(file)


def negative_log_likelihood(variance, spatial_range, smoothness, nugget,
                            covariance_mat, lower_cholesky,
                            observations, n_points=256):

    # ravel observations
    observations = observations.ravel()
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


# ------- COMPUTE MLE FOR A SINGLE REALIZATION ------- #

mle_estimates = np.empty([testing_parameter_space.shape[0], 2])
tmp_array = np.array([negative_log_likelihood
                      (variance=1.0, spatial_range=testing_parameter_space[j, 1],
                       smoothness=1.0, nugget=np.exp(testing_parameter_space[j, 0]),
                       covariance_mat=cov_mats_test[j, :, :], lower_cholesky=chol_mats_test[j, :, :],
                       observations=observations_test[i, :].reshape(256)) for i in range(observations_test.shape[0])
                      for j in range(testing_parameter_space.shape[0])])
for i in range(observations_test.shape[0]):
    tmp1 = tmp_array[i * testing_parameter_space.shape[0]: (i + 1) * testing_parameter_space.shape[0]]
    mle_estimates[i, 0] = testing_parameter_space[np.argmin(tmp1), 0]
    mle_estimates[i, 1] = testing_parameter_space[np.argmin(tmp1), 1]

# ------- SAVE MLE PRED FOR A SINGLE REALIZATION ------- #

with open("ML/preds_MLE.npy", mode="wb") as file:
    np.save(file, mle_estimates)
