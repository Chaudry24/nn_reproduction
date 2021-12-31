import numpy as np
import some_file_1
import scipy.optimize
from retry import retry


def objective_func(observations, distance_mat, variance, spatial_range, smoothness, nugget):
    cov = some_file_1.Spatial.compute_covariance("matern", distance_mat, variance, spatial_range,
                                                 smoothness, nugget)
    chol = np.linalg.cholesky(cov)
    first_term = 256.0 / 2.0 * np.log(2.0 * np.pi)
    log_det_term = 2.0 * np.trace(np.log(chol))
    second_term = 0.5 * log_det_term
    third_term = float(0.5 * observations.T @ np.linalg.inv(cov) @ observations)
    return first_term + second_term + third_term


# spatial grid
x = np.linspace(1, 16, 16)
y = x
x, y = np.meshgrid(x, y)
grid = np.array([x.ravel(), y.ravel()]).T

# spatial dist
spatial_dist = some_file_1.Spatial.compute_distance(grid[:, 0], grid[:, 1])

# open test data
with open("../npy/test30_tf_stat.npy", mode="rb") as file:
    observations_test_30 = np.load(file)

cons = ({"type": "ineq",
         "fun": lambda x: x},
        {"type": "ineq",
         "fun": lambda x: x})


@retry(np.linalg.LinAlgError, tries=-1, backoff=0, delay=0)
def mle():
    optimal_vals = scipy.optimize.minimize(fun=func, x0=np.random.uniform(0.1, 10.0, 2),
                                           constraints=cons, method="Nelder-Mead",
                                           options={"maxiter": int(1e4)})
    return optimal_vals.x


mle_optim_30 = np.empty((observations_test_30.shape[0], 2))
tmp = np.empty((30, 2))
for i in range(observations_test_30.shape[0]):
    for j in range(30):
        print(f"\nStarting mle for {i}th test sample and {j}th observation")
        data = observations_test_30[i, :, :, j].ravel()
        func = lambda x: objective_func(data, spatial_dist, variance=1.0, spatial_range=x[1], smoothness=1.0, nugget=x[0])
        results = mle()
        tmp[j, 0] = np.log(results[0])
        tmp[j, 1] = results[1]
    average_vals = np.average(tmp, axis=0)
    mle_optim_30[i, 0] = average_vals[0]
    mle_optim_30[i, 1] = average_vals[1]

with open("./ML30optim/ML30optim.npy", mode="rb") as file:
    np.save(file, mle_optim_30)
