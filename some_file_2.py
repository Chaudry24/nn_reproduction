import some_file_1
import numpy as np

# TODO: save distance matrix for testing data


def save_data(file_name_data="tmp_data_file", file_name_params="tmp_params_file",
              file_path=".", sample_variance=False, sample_spatial_range=True,
              sample_smoothness=False, sample_nugget=False,
              variance=1.0, spatial_range=0.2,
              smoothness=1.2, nugget=0.0,
              variance_range=(0.01, 5.0), spatial_range_range=(0.05, 0.98),
              smoothness_range=(0.05, 4.0), nugget_range=(0.0, 10.0),
              n_samples=10):

    """This function samples parameters from a given range.
     Then it samples gaussian processes with the given covariance parameters.
     Then it finally saves the resulting arrays in a .npy file"""

    if sample_variance and sample_spatial_range and sample_smoothness and sample_nugget:

        # sample parameters
        variance_range_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)

        # make a parameter space
        variance_range_vals, spatial_range_vals, smoothness_vals, nugget_vals = np.meshgrid(variance_range_vals,
                                                                                            spatial_range_vals,
                                                                                            smoothness_vals,
                                                                                            nugget_vals)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    spatial_range_vals.ravel(),
                                    smoothness_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        # compute data with given parameters
        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=parameter_space[i, 1],
                                                    smoothness=parameter_space[i, 2],
                                                    nugget=parameter_space[i, 3]).observed_data
                                for i in range(n_samples ** 4)])

        # save parameters
        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)
        # save data
        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_spatial_range and sample_smoothness:

        variance_range_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)

        variance_range_vals, spatial_range_vals, smoothness_vals = np.meshgrid(variance_range_vals,
                                                                               spatial_range_vals,
                                                                               smoothness_vals)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    spatial_range_vals.ravel(),
                                    smoothness_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=parameter_space[i, 1],
                                                    smoothness=parameter_space[i, 2],
                                                    nugget=nugget).observed_data
                                for i in range(n_samples ** 3)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_spatial_range and sample_nugget:
        variance_range_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        variance_range_vals, spatial_range_vals, nugget_vals = np.meshgrid(variance_range_vals,
                                                                           spatial_range_vals,
                                                                           nugget_vals)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    spatial_range_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=parameter_space[i, 1],
                                                    smoothness=smoothness,
                                                    nugget=parameter_space[i, 2]).observed_data
                                for i in range(n_samples ** 3)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_smoothness and sample_nugget:
        variance_range_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        variance_range_vals, smoothness_vals, nugget_vals = np.meshgrid(variance_range_vals,
                                                                        smoothness_vals,
                                                                        nugget_vals)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    smoothness_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=spatial_range,
                                                    smoothness=parameter_space[i, 1],
                                                    nugget=parameter_space[i, 2]).observed_data
                                for i in range(n_samples ** 3)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range and sample_smoothness and sample_nugget:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        spatial_range_vals, smoothness_vals, nugget_vals = np.meshgrid(spatial_range_vals,
                                                                       smoothness_vals,
                                                                       nugget_vals)
        parameter_space = np.stack([spatial_range_vals.ravel(),
                                    smoothness_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=parameter_space[i, 0],
                                                    smoothness=parameter_space[i, 1],
                                                    nugget=parameter_space[i, 2]).observed_data
                                for i in range(n_samples ** 3)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_spatial_range:
        variance_range_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        variance_range_vals, spatial_range_vals = np.meshgrid(variance_range_vals,
                                                              spatial_range_vals,)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    spatial_range_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=parameter_space[i, 1],
                                                    smoothness=smoothness,
                                                    nugget=nugget).observed_data
                                for i in range(n_samples ** 2)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_smoothness:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        variance_range_vals, smoothness_vals = np.meshgrid(spatial_range_vals,
                                                           nugget_vals)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    smoothness_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=smoothness_range,
                                                    smoothness=parameter_space[i, 1],
                                                    nugget=nugget).observed_data
                                for i in range(n_samples ** 2)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance and sample_nugget:
        variance_range_vals = np.random.uniform(variance_range[0], variance_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        variance_range_vals, nugget_vals = np.meshgrid(variance_range_vals,
                                                       nugget_vals)
        parameter_space = np.stack([variance_range_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i, 0],
                                                    spatial_range=spatial_range,
                                                    smoothness=smoothness,
                                                    nugget=parameter_space[i, 1]).observed_data
                                for i in range(n_samples ** 2)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range and sample_smoothness:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        spatial_range_vals, smoothness_vals= np.meshgrid(spatial_range_vals,
                                                         smoothness_vals)
        parameter_space = np.stack([spatial_range_vals.ravel(),
                                    smoothness_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=parameter_space[i, 0],
                                                    smoothness=parameter_space[i, 1],
                                                    nugget=nugget).observed_data
                                for i in range(n_samples ** 2)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range and sample_nugget:
        spatial_range_vals = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        spatial_range_vals, nugget_vals = np.meshgrid(spatial_range_vals,
                                                      nugget_vals)
        parameter_space = np.stack([spatial_range_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=parameter_space[i, 0],
                                                    smoothness=smoothness,
                                                    nugget=parameter_space[i, 1]).observed_data
                                for i in range(n_samples ** 2)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_smoothness and sample_nugget:
        smoothness_vals = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)
        nugget_vals = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)
        smoothness_vals, nugget_vals = np.meshgrid(smoothness_vals,
                                                   nugget_vals)
        parameter_space = np.stack([smoothness_vals.ravel(),
                                    nugget_vals.ravel()], 1)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=spatial_range,
                                                    smoothness=parameter_space[i, 0],
                                                    nugget=parameter_space[i, 1]).observed_data
                                for i in range(n_samples ** 2)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_variance:
        parameter_space = np.random.uniform(variance_range[0], variance_range[1], n_samples)

        data = np.column_stack([some_file_1.Spatial(variance=parameter_space[i],
                                                    spatial_range=spatial_range,
                                                    smoothness=smoothness,
                                                    nugget=nugget).observed_data
                                for i in range(n_samples)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_spatial_range:
        parameter_space = np.random.uniform(spatial_range_range[0], spatial_range_range[1], n_samples)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=parameter_space[i],
                                                    smoothness=smoothness,
                                                    nugget=nugget).observed_data
                                for i in range(n_samples)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    elif sample_smoothness:
        parameter_space = np.random.uniform(smoothness_range[0], smoothness_range[1], n_samples)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=spatial_range,
                                                    smoothness=parameter_space[i],
                                                    nugget=nugget).observed_data
                                for i in range(n_samples)])

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)

    else:  # sample_nugget
        parameter_space = np.random.uniform(nugget_range[0], nugget_range[1], n_samples)

        data = np.column_stack([some_file_1.Spatial(variance=variance,
                                                    spatial_range=spatial_range,
                                                    smoothness=smoothness,
                                                    nugget=parameter_space[i]).observed_data
                                for i in range(n_samples)])

        with open(f"{file_path}/{file_name_params}.npy", mode='wb') as params:
            np.save(params, parameter_space)

        with open(f"{file_path}/{file_name_data}.npy", mode='wb') as file:
            np.save(file, data)


def load_data(file_name="tmp_file", file_path="."):
    """This function loads the data"""
    with open(f"{file_path}/{file_name}.npy", mode="rb") as file:
        return np.load(file)

