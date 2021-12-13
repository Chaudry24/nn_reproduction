import numpy as np

# load testing parameter space
with open("../npy/test_subset.npy", mode="rb") as file:
    testing_parameter_space = np.load(file)

# load test observations
with open("../npy/test30_tf_stat.npy", mode="rb") as file:
    observations_test_30 = np.load(file)

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


mle_estimates_30 = np.empty([testing_parameter_space.shape[0], 2])
tmp_array1 = np.empty([30, testing_parameter_space.shape[0]])
tmp_array2 = np.empty([30, 2])

for l in range(observations_test_30.shape[0]):
    # compute MLE
    tmp1 = np.array([negative_log_likelihood
                     (variance=1.0, spatial_range=testing_parameter_space[i, 1],
                      smoothness=1.0, nugget=np.exp(testing_parameter_space[i, 0]),
                      covariance_mat=cov_mats_test[i, :, :], lower_cholesky=chol_mats_test[i, :, :],
                      observations=observations_test_30[l, :, :, j].reshape(256))
                     for i in range(testing_parameter_space.shape[0])
                     for j in range(30)])
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

with open("ML30/preds_ML30.npy", mode="wb") as file:
    np.save(file, mle_estimates_30)
