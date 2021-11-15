import matplotlib.colors
import numpy as np
from scipy.special import gamma, kv
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
# from retry import retry
# import tensorflow as tf
# import autograd
# import torch


class Spatial:
    """This class is used for generating and visualizing spatial data."""

    def __init__(self, n_points=256, distance_metric="euclidean", variance=1.0,
                 smoothness=1.5, spatial_range=0.5, nugget=0.0,
                 covariance_type="matern", realizations=1):
        self.n_points = n_points
        self.distance_metric = distance_metric
        self.variance = variance
        self.spatial_range = spatial_range
        self.smoothness = smoothness
        self.nugget = nugget
        self.covariance_type = covariance_type
        self.realizations = realizations
        self.domain = self.generate_grid()
        self.distance_matrix = self.compute_distance_normalized(self.domain[:, 0], self.domain[:, 1],
                                                                self.distance_metric)
        self.covariance = self.compute_covariance(self.covariance_type, self.distance_matrix,
                                                  variance=self.variance, smoothness=self.smoothness,
                                                  spatial_range=self.spatial_range, nugget=self.nugget)
        self.observed_data = self.observations()

    def generate_grid(self):
        """Generates a grid on a unit cube"""
        # x = np.random.uniform(0, 1, self.n_points)
        # y = np.random.uniform(0, 1, self.n_points)
        x = np.linspace(0, 1, self.n_points)
        y = np.linspace(0, 1, self.n_points)
        grid = np.array([x, y]).T
        return grid

    @staticmethod
    def compute_distance_normalized(x_coords, y_coords, distance_metric):
        """Returns normalized distance matrix for euclidean or
        great circle distances i.e. distance_matrix / max_distance
        WARNING: MAKE SURE THAT DATA IN RADIANS BEFORE USING GREAT CIRCLE
        METRIC."""
        domain = np.array([x_coords, y_coords]).T
        if distance_metric == 'euclidean':
            distance_matrix = sklearn.metrics.pairwise.euclidean_distances(domain)
            max_distance = np.max(distance_matrix)
            return distance_matrix / max_distance
        elif distance_metric == 'great_circle':
            distance_matrix = sklearn.metrics.pairwise.haversine_distances(domain)
            max_distance = np.max(distance_matrix)
            return distance_matrix / max_distance

    @staticmethod
    def compute_covariance(covariance_type, distance_matrix,
                           variance, smoothness, spatial_range, nugget,
                           n_points=256):
        """Computes the covariance matrix given the distance and covariance_type"""
        if covariance_type == 'matern':
            first_term = variance / (2 ** (smoothness - 1) * gamma(smoothness))
            second_term = (distance_matrix / spatial_range) ** smoothness
            third_term = kv(smoothness, distance_matrix / spatial_range)
            matern_covariance = first_term * second_term * third_term
            matern_covariance = np.nan_to_num(matern_covariance, copy=True,
                                              posinf=variance, nan=variance)
            if nugget > 0:
                matern_covariance += nugget * np.eye(n_points)
            return matern_covariance
        else:
            pass

    # # TODO: this is a second function for tensorlfow
    # @staticmethod
    # def compute_covariance2(covariance_type, distance_matrix,
    #                         variance, smoothness, spatial_range, nugget,
    #                         n_points=256):
    #     """Computes the covariance matrix given the distance and covariance_type"""
    #     # distance_matrix = tf.Variable
    #     smoothness = tf.Variable
    #     spatial_range = tf.Variable
    #     nugget = tf.Variable
    #     if covariance_type == 'matern':
    #         first_term = variance / (2 ** (smoothness - tf.Variable(1)) * gamma(smoothness))
    #         second_term = (distance_matrix / spatial_range) ** smoothness
    #         third_term = kv(smoothness, distance_matrix / spatial_range)
    #         matern_covariance = first_term * second_term * third_term
    #         matern_covariance = tf.where(tf.math.is_nan(matern_covariance),
    #                                      variance * tf.ones_like(matern_covariance),
    #                                      matern_covariance)
    #         matern_covariance = tf.where(tf.math.is_inf(matern_covariance),
    #                                      variance * tf.ones_like(matern_covariance),
    #                                      matern_covariance)
    #         if nugget > 0:
    #             matern_covariance += nugget * np.eye(n_points)
    #         return matern_covariance
    #     else:
    #         pass

    # TODO: this is a third function to pytorch
    # @staticmethod
    # def compute_covariance3(covariance_type, distance_matrix,
    #                         variance, smoothness, spatial_range, nugget,
    #                         n_points=256):
    #     """Computes the covariance matrix given the distance and covariance_type"""
    #     if covariance_type == 'matern':
    #         first_term = variance / (2 ** (smoothness - 1) * gamma(smoothness))
    #         second_term = (torch.tensor(distance_matrix) / spatial_range) ** smoothness
    #         third_term = kv(smoothness, torch.tensor(distance_matrix) / spatial_range)
    #         matern_covariance = first_term * second_term * third_term
    #         # replace infinity by variance
    #         matern_covariance[matern_covariance == float("inf")] = variance.double()
    #         # replace nan by variance
    #         matern_covariance[matern_covariance != matern_covariance] = variance.double()
    #         if nugget > 0:
    #             matern_covariance += nugget * np.eye(n_points)
    #         return matern_covariance
    #     else:
    #         pass

    def observations(self):
        """Returns observations from a GP with a given covariance"""
        if self.realizations == 1:
            iid_data = np.random.randn(self.n_points, 1)
            chol_decomp_lower = np.linalg.cholesky(self.covariance)
            observed_data = chol_decomp_lower @ iid_data
            return observed_data
        elif self.realizations > 1:
            observed_data = np.zeros((self.realizations, self.n_points, self.n_points))
            chol_decomp_lower = np.linalg.cholesky(self.covariance)
            # TODO: parallelize this loop
            for i in range(self.realizations):
                iid_data = np.random.randn(self.n_points, 1)
                observed_data[i, :, :] = chol_decomp_lower @ iid_data
        else:
            return SyntaxError("realizations need to be greater than or equal to 1")

    def plot_observed_data(self):
        """Makes a scatter plot of the observed data"""
        x_coord = self.domain[:, 0]
        y_coord = self.domain[:, 1]
        if self.realizations == 1:
            colors = self.observed_data.reshape(1, -1)
            plt.scatter(x=x_coord, y=y_coord, c=colors)
        elif self.realizations > 1:
            # TODO: parallelize this loop
            for i in range(self.realizations):
                colors = self.observed_data[i, :, :].reshape(-1, 1)
                plt.figure(num=i)
                normalize = matplotlib.colors.Normalize(0, 1)
                plt.scatter(x=x_coord, y=y_coord, cmap="PuRd", norm=normalize, c=colors)

    def plot_covariogram(self):
        """Makes a scatter plot of the covariogram"""
        covariance = self.covariance.reshape(-1, 1)
        distance_mat = self.distance_matrix.reshape(-1, 1)
        plt.scatter(x=distance_mat, y=covariance, c="red")


class Optimization(Spatial):
    """This class has the optimization schemes for minimizing the
    negative log-likelihood function"""

    def __init__(self, observations, distance_matrix, estimate_variance=False,
                 estimate_spatial_range=True, estimate_smoothness=False,
                 estimate_nugget=False, variance=1.0, spatial_range=0.2,
                 smoothness=1.5, nugget=0.0, covariance_type="matern",
                 optim_method="gradient-descent", n_points=256):
        super().__init__()
        self.estimate_variance = estimate_variance
        self.estimate_spatial_range = estimate_spatial_range
        self.estimate_smoothness = estimate_smoothness
        self.estimate_nugget = estimate_nugget
        self.n_points = n_points
        self.variance = variance
        self.spatial_range = spatial_range
        self.smoothness = smoothness
        self.nugget = nugget
        self.covariance_type = covariance_type
        self.observations = observations
        self.distance_matrix = distance_matrix
        self.optim_method = optim_method
        self.covariance = self.compute_covariance(self.covariance_type, self.distance_matrix,
                                                  variance=self.variance, smoothness=self.smoothness,
                                                  spatial_range=self.spatial_range, nugget=self.nugget,
                                                  n_points=self.n_points)
        self.objective_value = self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                       smoothness=self.smoothness, nugget=self.nugget)

    def objective_function(self, variance, spatial_range,
                           smoothness, nugget, n_points=256):
        """Computes the objective functional"""
        covariance_mat = self.compute_covariance(self.covariance_type, self.distance_matrix,
                                                 variance=variance, smoothness=smoothness,
                                                 spatial_range=spatial_range, nugget=nugget,
                                                 n_points=n_points)
        lower_cholesky = np.linalg.cholesky(covariance_mat)
        first_term = n_points / 2.0 * np.log(2.0 * np.pi)
        log_determinant_term = 2.0 * np.trace(np.log(lower_cholesky))
        second_term = 0.5 * log_determinant_term
        third_term = float(0.5 * self.observations.T @ np.linalg.inv(covariance_mat)
                           @ self.observations)
        return first_term + second_term + third_term

    # # TODO: this function is a test function which uses tesnorflow tensors
    # def objective_function2(self, variance, spatial_range,
    #                         smoothness, nugget, n_points=256):
    #     """Computes the objective functional"""
    #     covariance_mat = self.compute_covariance2(self.covariance_type, self.distance_matrix,
    #                                               variance=variance, smoothness=smoothness,
    #                                               spatial_range=spatial_range, nugget=nugget,
    #                                               n_points=n_points)
    #     lower_cholesky = np.linalg.cholesky(covariance_mat)
    #     first_term = n_points / 2.0 * np.log(2.0 * np.pi)
    #     log_determinant_term = 2.0 * np.trace(np.log(lower_cholesky))
    #     second_term = 0.5 * log_determinant_term
    #     third_term = float(0.5 * self.observations.T @ np.linalg.inv(covariance_mat)
    #                        @ self.observations)
    #     return first_term + second_term + third_term

    # # TODO: this function is a test function which uses pytorch tensors
    # def objective_function3(self, variance, spatial_range,
    #                         smoothness, nugget, n_points=256):
    #     """Computes the objective functional"""
    #     covariance_mat = self.compute_covariance3(self.covariance_type, self.distance_matrix,
    #                                               variance=variance, smoothness=smoothness,
    #                                               spatial_range=spatial_range, nugget=nugget,
    #                                               n_points=n_points)
    #     lower_cholesky = torch.cholesky(covariance_mat)
    #     first_term = n_points / 2.0 * np.log(2.0 * np.pi)
    #     log_determinant_term = 2.0 * torch.trace(torch.log(lower_cholesky))
    #     second_term = 0.5 * log_determinant_term
    #     third_term = torch.tensor(0.5 * torch.tensor(self.observations.T) @ torch.inverse(covariance_mat)
    #                               @ torch.tensor(self.observations))
    #     return first_term + second_term + third_term

    # def autograd_variance(self):
    #     pass

    def der_wrt_variance(self, h=1e-2):
        """This function approximates the derivative of the objective function
        with respect to variance using the finite difference approximation"""
        if self.estimate_variance:
            dobj_dvar = (self.objective_function(variance=self.variance + h, spatial_range=self.spatial_range,
                                                 smoothness=self.smoothness, nugget=self.nugget,
                                                 n_points=self.n_points) -
                         self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                 smoothness=self.smoothness, nugget=self.nugget,
                                                 n_points=self.n_points)) * 1 / h
            return float(dobj_dvar)
        else:
            return 0.0

    def der_wrt_spatial_range(self, h=1e-2):
        """This function approximates the derivative of the objective function
                with respect to spatial range using the finite difference approximation"""
        if self.estimate_spatial_range:
            dobj_dspatialrange = (self.objective_function(variance=self.variance, spatial_range=self.spatial_range + h,
                                                          smoothness=self.smoothness, nugget=self.nugget,
                                                          n_points=self.n_points) -
                                  self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                          smoothness=self.smoothness, nugget=self.nugget,
                                                          n_points=self.n_points)) * 1 / h
            return float(dobj_dspatialrange)
        else:
            return 0.0

    def der_wrt_smoothness(self, h=1e-2):
        """This function approximates the derivative of the objective function
                with respect to smoothness using the finite difference approximation"""
        if self.estimate_smoothness:
            dobj_dsmoothness = (self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                        smoothness=self.smoothness + h, nugget=self.nugget,
                                                        n_points=self.n_points) -
                                self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                        smoothness=self.smoothness, nugget=self.nugget,
                                                        n_points=self.n_points)) * 1 / h
            return float(dobj_dsmoothness)
        else:
            return 0.0

    def der_wrt_nugget(self, h=1e-2):
        """This function approximates the derivative of the objective function
                with respect to nugget using the finite difference approximation"""
        if self.estimate_nugget:
            dobj_dnugget = (self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                    smoothness=self.smoothness, nugget=self.nugget + h,
                                                    n_points=self.n_points) -
                            self.objective_function(variance=self.variance, spatial_range=self.spatial_range,
                                                    smoothness=self.smoothness, nugget=self.nugget,
                                                    n_points=self.n_points)) * 1 / h
            return float(dobj_dnugget)
        else:
            return 0.0

    def optimize(self, tolerance=1e-6):
        if self.optim_method == "gradient-descent":
            # used to start while loop
            stopping_condition = False
            # used to prevent infinite loop
            k = 0
            # step size scaling
            step_size_scale = 0.9
            # start the gradient descent algorithm
            while (not stopping_condition) and k < 1:
                # set learning rate
                step_size = 1e-3
                # increment to prevent infinite loop
                k += 1
                # compute gradients
                var_der = self.der_wrt_variance()
                spatial_range_der = self.der_wrt_spatial_range()
                smoothness_der = self.der_wrt_smoothness()
                nugget_der = self.der_wrt_nugget()
                # update step
                variance = self.variance - step_size * var_der
                spatial_range = self.spatial_range - step_size * spatial_range_der
                smoothness = self.smoothness - step_size * smoothness_der
                nugget = self.nugget - step_size * nugget_der
                # bad_conds = variance < 0 or spatial_range < 0 or smoothness < 0 or nugget < 0
                bad_conds = True
                # make sure that all parameters are positive
                while bad_conds:
                    step_size *= step_size_scale
                    variance = self.variance - step_size * var_der
                    spatial_range = self.spatial_range - step_size * spatial_range_der
                    smoothness = self.smoothness - step_size * smoothness_der
                    nugget = self.nugget - step_size * nugget_der
                    bad_conds = variance < 0 or spatial_range < 0 or smoothness < 0 or nugget < 0
                # compute new objective value
                new_objective_value = self.objective_function(variance=variance, spatial_range=spatial_range,
                                                              smoothness=smoothness, nugget=nugget,
                                                              n_points=self.n_points)
                # to prevent infinite loop
                j = 0
                # to guarantee convergence
                while new_objective_value > self.objective_value and j < 100:
                    # increment to prevent infinite loop
                    j += 1
                    # scale the step size
                    step_size *= step_size_scale
                    # update parameters
                    variance = self.variance - step_size * var_der
                    spatial_range = self.spatial_range - step_size * spatial_range_der
                    smoothness = self.smoothness - step_size * smoothness_der
                    nugget = self.nugget - step_size * nugget_der
                    # compute new objective value
                    new_objective_value = self.objective_function(variance=variance, spatial_range=spatial_range,
                                                                  smoothness=smoothness, nugget=nugget,
                                                                  n_points=self.n_points)
                # set new objective value and parameters
                self.objective_value = new_objective_value
                self.variance = variance
                self.smoothness = smoothness
                self.spatial_range = spatial_range
                self.nugget = nugget
                # calculate current gradient
                current_gradient = np.array([var_der, spatial_range,
                                             smoothness_der, nugget_der])
                # calculate the norm of the current gradient
                norm_current_gradient = np.linalg.norm(current_gradient)
                # stopping condition
                # stopping_condition = (norm_current_gradient / norm_initial_gradient < tolerance) or \
                #                      (norm_current_gradient < tolerance)
                # use the norm of the current gradient as the stopping condition
                stopping_condition = norm_current_gradient < tolerance
            return {"variance": self.variance, "spatial_range": self.spatial_range,
                    "smoothness": self.smoothness, "nugget": self.nugget,
                    "norm_current_gradient": norm_current_gradient, "objective_value": self.objective_value}


# field = Spatial()
# distance = field.distance_matrix
# data = field.observed_data
# model = Optimization(observations=data, distance_matrix=distance,
#                      estimate_variance=True, estimate_spatial_range=False)
#
# model.autograd_variance()
# optimized_model = model.optimizer()
#
#
# field.plot_observed_data()
# # field.plot_observed_data()
# asjkduh

# TODO: fix the color is scatter
# TODO: add documentation to functions
# TODO: implement automatic differentiation
# TODO: use property decorator/simplify methods
# a = 1
# b = 2
# torch.autograd.grad()
