# crevopt

__Glacier crevasse detection from optical satellite imagery.__

Python implementation of Gabor filter crevasse detection approach following Van Wyk de Vries et al. (2023). Designed for band 8 Sentinel-2 optical imagery.

> Van Wyk de Vries, M., Lea, J. M., & Ashmore, D. W. (2023). Crevasse density, orientation and temporal variability at Narsap Sermia, Greenland. Journal of Glaciology, 69(277), 1125–1137. doi:https://doi.org/10.1017/jog.2023.3

Matlab scripts associated with the original paper are available at:
https://github.com/MaxVWDV/Gabor-crevasse-detector

## Install

Download or clone this repository. It is recommended that you install the dependencies (`opencv`, `numpy`, `rioxarray`, and `numba`) in your `conda` environment before you perform a local install, as `conda` can better manage non-Python dependencies (e.g. GDAL) than `pip`.

The package can then be installed from the main `crevopt` directory via a pip local dev install:

```bash
pip install -e .
```

At this point, you should be able to `import crevopt` when in the associated Python environment.

## Principles

Van Wyk de Vries _et al._ (2023) apply a gabor filter bank to band 8 (near infra-red) Sentinel-2 satellite imagery to automatically detect and calculate the orientation of crevasses.

### Gabor filter


The final crevasse likelihood map $C_L$ and crevasse orientation map $C_\theta$ are calculated by taking the maxima of the Gabor phase at each pixel:

$$
C_L, C_\theta = \max{}^\theta(\arctan2[\textnormal{Re}(r_G^\theta), \textnormal{Im}(r_G ^\theta)])
$$

With $\textnormal{Re}(r_G^\theta)$ and $\textnormal{Re}(r_G^\theta)$ being the real and imaginary parts of the individual Gabor response maps respectively, and $\max{}^\theta$ representing the maximum across all individual Gabor response maps.

- The crevasse likelihood is calculated as the maximum value of all individual Gabor response maps.
- The crevasse orientation is calculated as the Gabor filter angle $\theta$ which produces this maximum value. 

### Binary crevasse mask

A binary crevasse mask $C_B$ is produced by thresholding the crevasse likelihood map, with pixels exceeding the empirically calibrated threshold of $[1.25 \textnormal{ median}(C_L)]$ defined as crevasses (in practice, I have found that the exact optimal threshold may not be exactly 1.25 - this can be adjusted in the code).

### Statistics

As well as calculating the binary crevasse mask and crevasse orientation map, crevasse statistics are calculated from 20$\times$20 pixel subregions:

- The crevasse spatial density (proportion of pixels in the subregion identified as crevasses).
- The median crevasse orientation.
- The mean absolute deviation of the measured crevasse orientation.

## User Guide

`crevopt` has three functions, designed to be run in sequence:

- `crevopt.gabor_filter()`
- `crevopt.crevasse_mask()`
- `crevopt.binned_statistics()`

The documentation for the three individual functions can be found below, and [`./example_notebook.ipynb` contains a complete example of implementation](./example_notebook.ipynb).

### `crevopt.gabor_filter()`

```python
def gabor_filter(image: np.ndarray | xr.DataArray, separation: float = 10, gab_size: float = 2, gab_band: float = 2, gab_ar: float = 0.1, threshold: float | None = 1.25, minangle: float = 0, maxangle: float = 179.99) -> Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
```

Glacier crevasse detection utilising the openCV gabor filter. Calculates the crevasse 'intensity' and 'orientation' in a satellite image.

Default parameters follow that of VWdV and optimised for band 8 Sentinel-2 imagery.
These may work well for some contexts but not all. Users may have to play with the
parameters (particularly the `threshold`) to qualitatively adjust for the best
output for their given study site.

**Parameters:**

- **img** (`np.ndarray | xr.DataArray`): The input image as a 2D numpy array or rioxarray/xarray xr.DataArray.
- **angle** (`float`): The seperation angle of the Gabor filter, in degrees. Defaults
to 10.
- **gab_size** (`float`): Scale of the gabor filter (wavelength in pixels). Defaults to 2.
- **gab_band** (`float`): Gabor spatial frequency bandwidth. Defaults to 2.
- **gab_ar** (`float`): Gabor filter spatial aspect ratio. Defaults to 0.1.
- **minangle** (`float`): Minimum gabor filter angle. Defaults to 0.
- **maxangle** (`float`): Maximum gabor filter angle. Defaults to 179.99.

**Returns:**

- `Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]`: The likelihood and orientation arrays. 'Likelihood' represents the local gabor phase angle maxima, a measure of how strongly linear a certain area of the image is. `orientation` represents the dominant orientation of the linear features.

### `crevopt.crevasse_mask()`

```python
def crevasse_mask(likelihood: np.ndarray | xr.DataArray, orientation: np.ndarray | xr.DataArray, threshold: float = 1.25, mask: np.ndarray | xr.DataArray = None) -> Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
```

Produces masked version of the likelihood and orientation arrays as output from the `gabor_filter` function. Can provide a surface mask (where valid surface ==1)
as well as a threshold value to clip the phase and phasedir arrays, which is
defined as a multiple of the median phase value.

The default threshold value is 1.25 following Van Wyk de Vries et al. (2023),
but the final result is highly sensitive to this value in the context of other
aspects of the output (e.g. the max/min absolute phase value), so this may need
qualitatively adjusting to an optimum value -- hence why this is a seperate function.

**Parameters:**

- **likelihood** (`np.ndarray | xr.DataArray`): The likelihood array output from the `gabor_filter` function.
- **orientation** (`np.ndarray | xr.DataArray`): The orientation array output from the `gabor_filter` function.
- **threshold** (`float | None`): The threshold for the mask clipping, in multiples of the median
intensity value. Defaults to 1.25. If set to 'None', no masking will occur.
- **mask** (`np.ndarray | xr.DataArray | None`): A surface mask (where valid surface ==1). Defaults to None.

**Returns:**

- `Tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]`: A masked version of the likelihood and orientation arrays.

### `crevopt.binned_statistics()`

```python
def binned_statistics(likelihood: xr.DataArray, orientation: xr.DataArray, window_size: int = 20, crs: str | int = None) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
```

From the masked likelihood and orientation map, calculate three crevasse statistics at a given window size (default 20 pixels):

1. The crevasse spatial density (proportion of pixels in the subregion identified as crevasses).
2. The median crevasse orientation.
3. The mean absolute deviation of the measured crevasse orientation.

Unlike previous functions in the processing chain, this function requires an input
of a rioxarray DataArray with valid geospatial metadata, not just a numpy array.

**Parameters:**

- **likelihood** (`xr.DataArray`): The masked likelihood array output from the `crevasse_mask` function.
- **orientation** (`xr.DataArray`): The masked phasedir array output from the `crevasse_mask` function.
- **window_size** (`int`): The size of the window in pixels. Defaults to 20.
- **crs** (`str | int`): The crs of the input arrays. Defaults to None, and will attempt to
use the crs of the input rioxarray DataArrays.

**Returns:**

- `Tuple[xr.DataArray, xr.DataArray, xr.DataArray]`: A tuple of the crevasse density, median crevasse orientation, and mean absolute deviation of the crevasse orientation.
