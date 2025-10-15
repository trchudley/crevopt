"""
crevopt

Glacier crevasse detection utilising a gabor filter following Van Wyk de Vries et al.
(2023). Adapted from matlab script supplement to Van Wyk de Vries et al. (2023) at:
https://github.com/MaxVWDV/Gabor-crevasse-detector

Van Wyk de Vries, M., Lea, J. M., & Ashmore, D. W. (2023). Crevasse density,
orientation and temporal variability at Narsap Sermia, Greenland. Journal of Glaciology,
69(277), 1125–1137. doi:https://doi.org/10.1017/jog.2023.3

Tom Chudley | thomas.r.chudley@durham.ac.uk | Durham University
October 2025
"""

import cv2
import numpy as np
import xarray as xr
import rioxarray as rxr

from numba import jit

from rasterio.transform import Affine

# from xarray import DataArray
from typing import Tuple, Optional


def gabor_filter(
    image: np.ndarray | xr.DataArray,
    separation: float = 10,
    gab_size: float = 2,
    gab_band: float = 2,
    gab_ar: float = 0.1,
    minangle: float = 0,
    maxangle: float = 179.99,
) -> Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Glacier crevasse detection utilising the openCV gabor filter. Calculates the
    crevasse 'intensity' and 'orientation' in a satellite image.

    Default parameters follow that of VWdV and optimised for band 8 Sentinel-2 imagery.
    These may work well for some contexts but not all. Users may have to play with the
    parameters (particularly the `threshold`) to qualitatively adjust for the best
    output for their given study site.

    :param img: The input image as a 2D numpy array or rioxarray/xarray xr.DataArray.
    :type img: np.ndarray | xr.DataArray
    :param angle: The seperation angle of the Gabor filter, in degrees. Defaults
        to 10.
    :type angle: float
    :param gab_size: Scale of the gabor filter (wavelength in pixels). Defaults to 4.
    :type gab_size: float
    :param gab_band: Gabor spatial frequency bandwidth. Defaults to 2.
    :type gab_band: float
    :param gab_ar: Gabor filter spatial aspect ratio. Defaults to 0.1.
    :type gab_ar: float
    :param minangle: Minimum gabor filter angle. Defaults to 0.
    :type minangle: float
    :param maxangle: Maximum gabor filter angle. Defaults to 179.99.
    :type maxangle: float

    :return likelihood: Local gabor phase angle maxima. A measure of how strongly linear
        a certain area of the image is.
    :type likelihood: np.ndarray | xr.DataArray
    :return orientation: Local dominant line direction, in degrees clockwise from North.
    :type orientation: np.ndarray | xr.DataArray
    """

    # Sanitise input array
    if isinstance(image, xr.DataArray):
        img_arr = image.values
        revert_rioxarray = True
    elif isinstance(image, np.ndarray):
        img_arr = image
        revert_rioxarray = False
    else:
        raise ValueError(
            "Input `image` must be a (rio)xarray datarray or a 2D numpy array."
        )

    # Ensure image is 2D (grayscale)
    img_arr = img_arr.squeeze()
    if len(img_arr.shape) > 2:
        image = image.squeeze()
    if len(img_arr.shape) != 2:
        raise ValueError("Input array `img` must be 2D, or able to be squeezed to 2D.")

    # Convert image to float32 if needed
    if img_arr.dtype != np.float32:
        img_arr = img_arr.astype(np.float32)

    # Generate angles for gabor filter bank
    angles = np.arange(minangle, maxangle + separation, separation)

    # Calculate sigma values from bandwidth and aspect ratio
    # Solving for sigma in terms of wavelength and bandwidth
    wavelength = gab_size
    bandwidth = gab_band

    # Calculate sigma_x (along the filter direction)
    sigma_x = (
        wavelength
        / np.pi
        * np.sqrt(np.log(2) / 2)
        * ((2**bandwidth + 1) / (2**bandwidth - 1))
    )

    # Calculate sigma_y using aspect ratio
    sigma_y = sigma_x / gab_ar

    # Determine kernel size (make it odd and large enough)
    kernel_size = int(2 * np.ceil(3 * max(sigma_x, sigma_y)) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Store output size
    out_size = image.shape

    # Pad image to handle borders
    pad_size = kernel_size // 2
    image_padded = cv2.copyMakeBorder(
        img_arr, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE
    )

    # Initialize output array for all filter responses
    # Store complex responses (magnitude and phase)
    out_real = np.zeros((out_size[0], out_size[1], len(angles)), dtype=np.float32)
    out_imag = np.zeros((out_size[0], out_size[1], len(angles)), dtype=np.float32)

    # Apply Gabor filters at each orientation
    for idx, angle in enumerate(angles):
        # Convert angle to radians (OpenCV uses radians)
        theta = np.deg2rad(angle)

        # Create Gabor kernel
        # OpenCV's getGaborKernel uses different convention, so adjust angle
        gabor_kernel = cv2.getGaborKernel(
            ksize=(kernel_size, kernel_size),
            sigma=sigma_x,
            theta=theta,
            lambd=wavelength,
            gamma=gab_ar,
            psi=0,  # Phase offset = 0 for real part
            ktype=cv2.CV_32F,
        )

        # Get imaginary part (90 degree phase shift)
        gabor_kernel_imag = cv2.getGaborKernel(
            ksize=(kernel_size, kernel_size),
            sigma=sigma_x,
            theta=theta,
            lambd=wavelength,
            gamma=gab_ar,
            psi=np.pi / 2,  # 90 degree phase shift for imaginary part
            ktype=cv2.CV_32F,
        )

        # Apply filters
        filtered_real = cv2.filter2D(image_padded, cv2.CV_32F, gabor_kernel)
        filtered_imag = cv2.filter2D(image_padded, cv2.CV_32F, gabor_kernel_imag)

        # Crop to original size
        out_real[:, :, idx] = filtered_real[
            pad_size : pad_size + out_size[0], pad_size : pad_size + out_size[1]
        ]
        out_imag[:, :, idx] = filtered_imag[
            pad_size : pad_size + out_size[0], pad_size : pad_size + out_size[1]
        ]

    # Calculate phase angle from real and imaginary parts
    phase_angles = np.arctan2(out_real, out_imag)

    # Rescale phase angles to [0, 1] range
    phase_angles_rescaled = (phase_angles + np.pi) / (2 * np.pi)
    # phase_angles_rescaled = phase_angles

    # Find maximum phase response across all orientations
    likelihood = np.max(phase_angles_rescaled, axis=2)

    # Calculate the dominant direction
    max_idx = np.argmax(phase_angles_rescaled, axis=2)
    orientation = max_idx * separation

    # Correct phase for geographic orientation
    orientation = 180 - orientation

    # Revert to rioxarray
    if revert_rioxarray:
        likelihood = (image * 0 + likelihood).squeeze()
        orientation = (image * 0 + orientation).squeeze()

    return likelihood, orientation


def crevasse_mask(
    likelihood: np.ndarray | xr.DataArray,
    orientation: np.ndarray | xr.DataArray,
    threshold: float = 1.25,
    mask: np.ndarray | xr.DataArray = None,
) -> Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Produces masked version of the likelihood and orientation arrays as output from the
    `gabor_filter` function. Can provide a surface mask (where valid surface ==1)
    as well as a threshold value to clip the phase and phasedir arrays, which is
    defined as a multiple of the median phase value.

    The default threshold value is 1.25 following Van Wyk de Vries et al. (2023),
    but the final result is highly sensitive to this value in the context of other
    aspects of the output (e.g. the max/min absolute phase value), so this may need
    qualitatively adjusting to an optimum value -- hence why this is a seperate function.

    :param likelihood: The likelihood array output from the `gabor_filter` function.
    :type likelihood: np.ndarray | xr.DataArray
    :param orientation: The orientation array output from the `gabor_filter` function.
    :type orientation: np.ndarray | xr.DataArray
    :param threshold: The threshold for the mask clipping, in multiples of the median
        intensity value. Defaults to 1.25. If set to 'None', no masking will occur.
    :type threshold: float | None
    :param mask: A surface mask (where valid surface ==1). Defaults to None.
    :type mask: np.ndarray | xr.DataArray | None

    :return: A masked version of the likelihood and orientation arrays.
    :rtype: Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]

    """

    if isinstance(likelihood, xr.DataArray) and isinstance(orientation, xr.DataArray):
        rxr_copy = likelihood
        likelihood = likelihood.values
        orientation = orientation.values
        revert_rioxarray = True
    elif isinstance(likelihood, np.ndarray) and isinstance(orientation, np.ndarray):
        rxr_copy = None
        revert_rioxarray = False
    else:
        raise ValueError(
            "Input `phase` and `phasedir` must be (rio)xarray datarrays or 2D numpy arrays of the same type."
        )

    if mask is not None:
        likelihood = np.where(mask, likelihood, np.nan)
        orientation = np.where(mask, orientation, np.nan)

    # Threshold phase and phasedir to extract crevasses
    likelihood_mask = likelihood > (threshold * np.nanmedian(likelihood))
    likelihood = np.where(likelihood_mask, likelihood, np.nan)
    orientation = np.where(likelihood_mask, orientation, np.nan)

    if revert_rioxarray:
        likelihood = (rxr_copy * likelihood).squeeze()
        orientation = (rxr_copy * orientation).squeeze()

    return likelihood, orientation


def binned_statistics(
    likelihood: xr.DataArray,
    orientation: xr.DataArray,
    window_size: int = 20,
    crs: str | int = None,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    From the masked likelihood and orientation map, calculate three crevasse statistics
    at a given window size (default 20 pixels):

    1. The crevasse spatial density (proportion of pixels in the subregion identified as crevasses).
    2. The median crevasse orientation.
    3. The mean absolute deviation of the measured crevasse orientation.

    Unlike previous functions in the processing chain, this function requires an input
    of a rioxarray DataArray with valid geospatial metadata, not just a numpy array.

    :param likelihood: The masked likelihood array output from the `crevasse_mask` function.
    :type likelihood: xr.DataArray
    :param orientation: The masked phasedir array output from the `crevasse_mask` function.
    :type orientation: xr.DataArray
    :param window_size: The size of the window in pixels. Defaults to 20.
    :type window_size: int
    :param crs: The crs of the input arrays. Defaults to None, and will attempt to
        use the crs of the input rioxarray DataArrays.
    :type crs: str | int

    :return: A tuple of the crevasse density, median crevasse orientation, and mean
        absolute deviation of the crevasse orientation.
    :rtype: Tuple[xr.DataArray, xr.DataArray, xr.DataArray]
    """

    # Sanity check data types
    if not isinstance(likelihood, xr.DataArray) and isinstance(
        orientation, xr.DataArray
    ):
        raise ValueError(
            "Input `likelihood` and `orientation` must both be be (rio)xarray datarrays."
        )

    if crs is None:
        crs = likelihood.rio.crs
        if crs is None:
            raise ValueError(
                "Input `phase` does not have a crs value. Write one to it using `phase.rio.write_crs()` or provide a crs to this function using the `crs` parameter."
            )

    # Extract numpy arrays and truncate to complete windows
    likelihood_arr = likelihood.values.astype(np.float32)
    orientation_arr = orientation.values.astype(np.float32)

    # Truncate arrays to only include complete windows
    truncated_height = (likelihood_arr.shape[0] // window_size) * window_size
    truncated_width = (likelihood_arr.shape[1] // window_size) * window_size

    likelihood_arr = likelihood_arr[:truncated_height, :truncated_width]
    orientation_arr = orientation_arr[:truncated_height, :truncated_width]

    # Call numba-compiled function
    data_proportion, median_orientation, mad_orientation = (
        _calculate_window_stats_numba(likelihood_arr, orientation_arr, window_size)
    )

    # Set data_proportion == 0 to nan
    data_proportion[data_proportion == 0] = np.nan

    # Get original geospatial transform
    original_transform = likelihood.rio.transform()

    # # Calculate the actual extent covered by complete windows
    # # (this may be slightly smaller than the original extent)
    # pixels_used_height = (likelihood_arr.shape[0] // window_size) * window_size
    # pixels_used_width = (likelihood_arr.shape[1] // window_size) * window_size

    # Calculate new transform for reduced resolution
    # The extent is now truncated to only include complete windows
    new_transform = Affine(
        original_transform.a * window_size,  # x pixel size
        original_transform.b,
        original_transform.c,  # x offset (unchanged)
        original_transform.d,
        original_transform.e * window_size,  # y pixel size
        original_transform.f,  # y offset (unchanged)
    )

    # Generate coordinate arrays for the reduced resolution output
    # Use the transform to calculate the center coordinates of each output pixel
    out_height, out_width = data_proportion.shape

    # Calculate x and y coordinates (pixel centers)
    x_coords = (
        np.arange(out_width) * window_size * original_transform.a
        + original_transform.c
        + (window_size * original_transform.a / 2)
    )
    y_coords = (
        np.arange(out_height) * window_size * original_transform.e
        + original_transform.f
        + (window_size * original_transform.e / 2)
    )

    # Create output xr.DataArrays with updated geospatial metadata
    results_arrs = [data_proportion, median_orientation, mad_orientation]
    results = []
    for data in results_arrs:

        # Create new xr.DataArray with correct dimensions
        da = xr.DataArray(
            data,
            dims=likelihood.dims,
            coords={likelihood.dims[0]: y_coords, likelihood.dims[1]: x_coords},
            attrs=likelihood.attrs,
        )

        # Set CRS and transform
        da.rio.write_crs(crs, inplace=True)
        da.rio.write_transform(new_transform, inplace=True)

        results.append(da)

    return results


@jit(nopython=True)
def _calculate_window_stats_numba(likelihood_arr, orientation_arr, window_size):
    """
    Numba-compiled function to calculate windowed statistics at reduced resolution.
    Only processes complete windows (truncates partial windows at edges).
    """
    shape = likelihood_arr.shape

    # Calculate output dimensions (only complete windows)
    out_height = shape[0] // window_size
    out_width = shape[1] // window_size

    data_proportion = np.full((out_height, out_width), np.nan)
    median_orientation = np.full((out_height, out_width), np.nan)
    mad_orientation = np.full((out_height, out_width), np.nan)

    for i in range(out_height):
        for j in range(out_width):
            # Calculate window boundaries (complete windows only)
            i_start = i * window_size
            i_end = i_start + window_size
            j_start = j * window_size
            j_end = j_start + window_size

            # Extract windows
            phase_window = likelihood_arr[i_start:i_end, j_start:j_end]
            phasedir_window = orientation_arr[i_start:i_end, j_start:j_end]

            # 1. Calculate proportion of non-NaN data in phase
            total_pixels = phase_window.size
            valid_count = 0
            for pi in range(phase_window.shape[0]):
                for pj in range(phase_window.shape[1]):
                    if not np.isnan(phase_window[pi, pj]):
                        valid_count += 1

            data_proportion[i, j] = valid_count / total_pixels

            # Collect valid orientations
            valid_orientations = []
            for pi in range(phase_window.shape[0]):
                for pj in range(phase_window.shape[1]):
                    if not np.isnan(phase_window[pi, pj]):
                        valid_orientations.append(phasedir_window[pi, pj])

            if len(valid_orientations) > 0:
                # Convert list to array for calculations
                valid_orient_arr = np.array(valid_orientations)

                # 2. Calculate circular median for orientations (0-180°)
                # Convert to radians and double the angle
                angles_rad = np.deg2rad(valid_orient_arr * 2.0)

                # Convert to unit vectors
                sum_x = 0.0
                sum_y = 0.0
                for angle in angles_rad:
                    sum_x += np.cos(angle)
                    sum_y += np.sin(angle)

                mean_x = sum_x / len(valid_orientations)
                mean_y = sum_y / len(valid_orientations)
                mean_angle = np.arctan2(mean_y, mean_x)

                # Convert back to 0-180° range
                circ_median = (np.rad2deg(mean_angle) / 2.0) % 180.0
                median_orientation[i, j] = circ_median

                # 3. Calculate circular MAD
                diffs = np.empty(len(valid_orientations))
                for k in range(len(valid_orientations)):
                    diff = np.abs(valid_orient_arr[k] - circ_median)
                    # Handle wrap-around
                    diffs[k] = min(diff, 180.0 - diff)

                # Calculate median of diffs
                mad_orientation[i, j] = np.median(diffs)

    return data_proportion, median_orientation, mad_orientation
