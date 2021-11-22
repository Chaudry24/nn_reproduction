import some_file_1
import numpy as np
from retry import retry
import os


@retry(Exception, tries=-1, delay=0, backoff=0)
def save_data(file_name_data="tmp_data_file", file_name_params="tmp_params_file",
              file_path="./data", sample_variance=False, sample_spatial_range=False,
              sample_smoothness=False, sample_nugget=True, realizations=5,
              variance=1.0, spatial_range=0.2,
              smoothness=1.0, nugget=0.15,
              variance_range=(0.01, 5.0), spatial_range_range=(0.01, 0.60),
              smoothness_range=(0.05, 4.0), nugget_range=(0.0, 10.0),
              n_samples=10):

    """This function samples parameters from a given range.
     Then it samples gaussian processes with the given covariance parameters.
     Then it finally saves the resulting arrays in a .npy file"""

    if sample_variance and sample_spatial_range and sample_smoothness and sample_nugget:

        # sample parameters
        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)

        # make a parameter space
        # variance_range_vals, spatial_range_vals, smoothness_vals, nugget_vals = np.meshgrid(variance_range_vals,
        #                                                                                     spatial_range_vals,
        #                                                                                     smoothness_vals,
        #                                                                                     nugget_vals)

        parameter_space = np.stack([variance_vals,
                                    spatial_range_vals,
                                    smoothness_vals,
                                    nugget_vals], 1)

        # compute data with given parameters
        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=parameter_space[i, 1],
                                             smoothness=parameter_space[i, 2],
                                             nugget=parameter_space[i, 3],
                                             realizations=realizations).observed_data
                        for i in range(n_samples)], 0)
        # data is of shape: n_samples x 256 x realizations

        # save parameters
        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)
        # save data
        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_spatial_range and sample_smoothness:

        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)

        # variance_range_vals, spatial_range_vals, smoothness_vals = np.meshgrid(variance_range_vals,
        #                                                                        spatial_range_vals,
        #                                                                        smoothness_vals)

        parameter_space = np.stack([variance_vals,
                                    spatial_range_vals,
                                    smoothness_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=parameter_space[i, 1],
                                             smoothness=parameter_space[i, 2],
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_spatial_range and sample_nugget:
        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        # variance_range_vals, spatial_range_vals, nugget_vals = np.meshgrid(variance_range_vals,
        #                                                                    spatial_range_vals,
        #                                                                    nugget_vals)

        parameter_space = np.stack([variance_vals,
                                    spatial_range_vals,
                                    nugget_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=parameter_space[i, 1],
                                             smoothness=smoothness,
                                             nugget=parameter_space[i, 2],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_smoothness and sample_nugget:
        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        # variance_range_vals, smoothness_vals, nugget_vals = np.meshgrid(variance_range_vals,
        #                                                                 smoothness_vals,
        #                                                                 nugget_vals)
        parameter_space = np.stack([variance_vals,
                                    smoothness_vals,
                                    nugget_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=spatial_range,
                                             smoothness=parameter_space[i, 1],
                                             nugget=parameter_space[i, 2],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range and sample_smoothness and sample_nugget:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        # spatial_range_vals, smoothness_vals, nugget_vals = np.meshgrid(spatial_range_vals,
        #                                                                smoothness_vals,
        #                                                                nugget_vals)
        parameter_space = np.stack([spatial_range_vals,
                                    smoothness_vals,
                                    nugget_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=variance,
                                             spatial_range=parameter_space[i, 0],
                                             smoothness=parameter_space[i, 1],
                                             nugget=parameter_space[i, 2],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_spatial_range:
        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        # variance_range_vals, spatial_range_vals = np.meshgrid(variance_vals,
        #                                                       spatial_range_vals,)
        parameter_space = np.stack([variance_vals,
                                    spatial_range_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=parameter_space[i, 1],
                                             smoothness=smoothness,
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_smoothness:
        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        # variance_vals, smoothness_vals = np.meshgrid(variance_vals,
        #                                              smoothness_vals)
        parameter_space = np.stack([variance_vals,
                                    smoothness_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=spatial_range,
                                             smoothness=parameter_space[i, 1],
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_nugget:
        variance_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        # variance_vals, nugget_vals = np.meshgrid(variance_vals,
        #                                          nugget_vals)
        parameter_space = np.stack([variance_vals,
                                    nugget_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                             spatial_range=spatial_range,
                                             smoothness=smoothness,
                                             nugget=parameter_space[i, 1],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range and sample_smoothness:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        # spatial_range_vals, smoothness_vals = np.meshgrid(spatial_range_vals,
        #                                                  smoothness_vals)
        parameter_space = np.stack([spatial_range_vals,
                                    smoothness_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=variance,
                                             spatial_range=parameter_space[i, 0],
                                             smoothness=parameter_space[i, 1],
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range and sample_nugget:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        # spatial_range_vals, nugget_vals = np.meshgrid(spatial_range_vals,
        #                                               nugget_vals)
        parameter_space = np.stack([spatial_range_vals,
                                    nugget_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=variance,
                                             spatial_range=parameter_space[i, 0],
                                             smoothness=smoothness,
                                             nugget=parameter_space[i, 1],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_smoothness and sample_nugget:
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        # smoothness_vals, nugget_vals = np.meshgrid(smoothness_vals,
        #                                            nugget_vals)
        parameter_space = np.stack([smoothness_vals,
                                    nugget_vals], 1)

        data = np.stack([some_file_1.Spatial(variance=variance,
                                             spatial_range=spatial_range,
                                             smoothness=parameter_space[i, 0],
                                             nugget=parameter_space[i, 1],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance:
        parameter_space = np.random.uniform(variance_range[0], variance_range[1], n_samples)

        data = np.stack([some_file_1.Spatial(variance=parameter_space[i],
                                             spatial_range=spatial_range,
                                             smoothness=smoothness,
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range:
        parameter_space = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)

        data = np.stack([some_file_1.Spatial(variance=parameter_space,
                                             spatial_range=spatial_range[i],
                                             smoothness=smoothness,
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_smoothness:
        parameter_space = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)

        data = np.stack([some_file_1.Spatial(variance=variance,
                                             spatial_range=spatial_range,
                                             smoothness=parameter_space[i],
                                             nugget=nugget,
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    else:  # sample_nugget
        parameter_space = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)

        data = np.stack([some_file_1.Spatial(variance=variance,
                                             spatial_range=spatial_range,
                                             smoothness=smoothness,
                                             nugget=parameter_space[i],
                                             realizations=realizations).observed_data
                         for i in range(n_samples)], 0)

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)


def load_data(file_name="tmp_file", file_path="./data", is_testing=False):
    """This function loads the data"""
    # load data
    with open(f"{file_path}/{file_name}.npy", mode="rb") as file_temp:
        # save the data
        file = np.load(file_temp)
        # delete data if it is testing data
        if is_testing:
            os.remove(f"{file_path}/{file_name}.npy")
    return file

