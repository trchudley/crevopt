"""
crevopt

Glacier crevasse detection utilising a gabor filter following Van Wyk de Vries et al.
(2023). Designed for band 8 Sentinel-2 optical imagery.

Adapted from matlab script supplement to Van Wyk de Vries et al. (2023) available at:
https://github.com/MaxVWDV/Gabor-crevasse-detector

Van Wyk de Vries, M., Lea, J. M., & Ashmore, D. W. (2023). Crevasse density, 
orientation and temporal variability at Narsap Sermia, Greenland. Journal of Glaciology, 
69(277), 1125–1137. doi:https://doi.org/10.1017/jog.2023.3

Tom Chudley | thomas.r.chudley@durham.ac.uk | Durham University
February 2024
"""

__version__ = "1.0.0"


import cv2 as cv
import numpy as np

from typing import Tuple

# =================================================================================== #
# MAIN FUNCTION
# =================================================================================== #


def detect(
    img: np.ndarray,
    angle: float = 10,
    sigma: float = 4,
    lambd: float = 10.0,
    gamma: float = 0.2,
    ksize: bool | int = None,
    mask: bool = True,
    mask_clip_thresh: float = 1.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Glacier crevasse detection utilising the openCV gabor filter. Defaults optimised
    for band 8 Sentinel-2 imagery.

    Adapted from matlab script supplement to Van Wyk de Vries et al. (2023).

    Paper: https://doi.org/10.1017/jog.2023.3
    GitHub repository: https://github.com/MaxVWDV/Gabor-crevasse-detector

    Args:
        img (np.ndarray): The input image as a 2D numpy array.
        angle (float, optional): The angle in degrees of the Gabor filter. Defaults to 10.
        sigma (float, optional): The standard deviation of the Gaussian envelope. Defaults to 4.
        lambd (float, optional): The wavelength of the sinusoidal factor. Defaults to 10.0.
        gamma (float, optional): The aspect ratio. Defaults to 0.2.
        ksize (bool | int, optional): The size of the Gabor filter kernel. Defaults to None.
        mask (bool, optional): Whether to apply a mask to the output. Defaults to True.
        mask_clip_thresh (float, optional): The threshold for the mask clipping. Defaults to 1.25.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the likelihood and direction of the detected Gabor features as numpy arrays.
    """

    # Sanitise input array
    if type(img) != np.ndarray:
        raise ValueError("Input `img` must be a 2D numpy array.")
    if len(img.shape) > 2:
        img = img.squeeze()
    if len(img.shape) != 2:
        raise ValueError("Input array `img` must be 2D, or able to be squeezed to 2D.")

    # # Convert from VwDV-style to OpenCV-style
    # lambd = gab_size
    # sigma = 0.5 * lambd * gab_band
    # gamma = gab_ar

    # print(f"lambd/wavelength/gab_size: {lambd}")
    # print(f"sigma: {sigma}")
    # print(f"aspect ratio: {gamma}")
    # print(f"ksize: {ksize}")

    if ksize == None:
        ksize = int(1 + 2 * np.ceil(4 * sigma))

    filters_real, thetas = _construct_gabor_filter(
        angle=angle,
        ksize=ksize,
        sigma=sigma,
        lambd=lambd,
        gamma=gamma,
        psi=0,
    )
    filters_imag, _ = _construct_gabor_filter(
        angle=angle,
        ksize=ksize,
        sigma=sigma,
        lambd=lambd,
        gamma=gamma,
        psi=np.pi / 2,
    )

    img_flts = []

    # loop through filters and take highest from each to get final result
    for kern_real, kern_imag in zip(filters_real, filters_imag):

        img_flt_real = cv.filter2D(img, -1, kern_real)
        img_flt_imag = cv.filter2D(img, -1, kern_imag)

        img_flt = np.arctan2(img_flt_real, img_flt_imag)

        img_flts.append(img_flt)

    img_flt_likelihood = np.amax(img_flts, axis=0)
    img_flt_direction = np.argmax(img_flts, axis=0)

    img_flt_direction = np.float32(img_flt_direction * thetas[1])

    if mask:
        threshold = mask_clip_thresh * np.median(img_flt_likelihood)
        crev_mask = img_flt_likelihood < threshold
        img_flt_likelihood[crev_mask] = np.nan
        img_flt_direction[crev_mask] = np.nan

    return img_flt_likelihood, img_flt_direction


# =================================================================================== #
# PRIVATE FUNCTIONS
# =================================================================================== #


def _construct_gabor_filter(angle, ksize, sigma, lambd, gamma, psi):
    """
    Construct Gabor filters for edge detection.
    Parameters:
        angle (float): Rotation angle in degrees.
        ksize (int): Size of the filter.
        sigma (float): Standard deviation of the Gaussian envelope.
        lambd (float): Wavelength of the sinusoidal factor.
        gamma (float): Spatial aspect ratio.
        psi (float): Phase offset.
    Returns:
        Tuple[List[np.ndarray], np.ndarray]: A list of Gabor filter kernels and an array of orientation angles.
    """

    filters = []
    thetas = np.arange(0, np.pi, np.deg2rad(angle))
    for theta in thetas:  # Theta is the orientation for edge detection
        kern = cv.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_32F
        )
        #         kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters, thetas
