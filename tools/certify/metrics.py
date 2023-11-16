"""Level A and Level B metrics for the certification harness.

Certification spec §4 judges a converted model on max-abs error, MAE and PSNR (Level A) and on SSIM
and LPIPS (Level B). Those numbers decide whether a model ships, so they are implemented here rather
than imported: the harness must be able to state exactly what it computed, and the test suite pins
every function to constants computed *outside* this file (scikit-image at fixture time), so a silent
change in behaviour cannot be ratified by the code that caused it.

Import policy: numpy only at module scope. CI installs pytest + numpy, and a
module-scope `import torch` here would turn every suite in the repo red at collection. LPIPS pulls
its stack in inside the function that needs it.

SSIM follows scikit-image's `structural_similarity` defaults exactly -- 7x7 uniform window,
sample-covariance normalisation, K1=0.01, K2=0.03, border cropped -- so the number this project
records is comparable with the number everyone else publishes.
"""

from __future__ import annotations

import math

import numpy as np

BYTE = 255.0

#: scikit-image `structural_similarity` defaults, restated so the harness records what it used.
SSIM_WINDOW = 7
SSIM_K1 = 0.01
SSIM_K2 = 0.03

#: LPIPS trunk. `alex` is the variant the LPIPS authors recommend as a perceptual metric.
LPIPS_NET = "alex"


class MetricError(ValueError):
    """The two tensors cannot be compared. Never silently coerced -- comparing a mismatched pair
    would produce a number that looks like evidence and is not."""


def _pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a comparison pair and lift it to float64.

    float64 is deliberate: the inputs are float32 tensors and several of these metrics subtract
    nearly equal quantities. The comparison should measure the model, not the accumulator.
    """
    if a.shape != b.shape:
        raise MetricError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise MetricError("empty tensors have no metrics")
    return a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)


def _nchw(array: np.ndarray) -> np.ndarray:
    """Accept the project's tensor convention (1, C, H, W) and return it unchanged."""
    if array.ndim != 4 or array.shape[0] != 1:
        raise MetricError(f"expected an NCHW tensor with N=1, got shape {array.shape}")
    return array


# --- Level A ------------------------------------------------------------------------------
def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """The single worst component difference. Reported in [0,1] units; the report shows /255."""
    x, y = _pair(a, b)
    return float(np.abs(x - y).max())


def mae(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    return float(np.abs(x - y).mean())


def mse(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _pair(a, b)
    return float(np.square(x - y).mean())


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, matching skimage.metrics.peak_signal_noise_ratio.

    Identical tensors give infinity rather than a large finite number: a made-up ceiling would
    be indistinguishable from a real measurement in the aggregate.
    """
    error = mse(a, b)
    if error == 0.0:
        return math.inf
    return float(10.0 * math.log10((data_range**2) / error))


def within_tolerance_fraction(a: np.ndarray, b: np.ndarray, tolerance: float) -> float:
    """Fraction of *pixels* whose every channel is within `tolerance`.

    Certification spec §4 reads "max-abs <= 4/255 on >= 99% of pixels". A pixel is counted as
    failing when any one of its channels exceeds the tolerance -- the conservative reading, and the
    one that matches what a viewer would see.
    """
    x, y = _pair(_nchw(a), _nchw(b))
    per_pixel = np.abs(x - y).max(axis=1)  # worst channel at each spatial position
    return float((per_pixel <= tolerance).mean())


# --- Level B ------------------------------------------------------------------------------
def _box_mean(image: np.ndarray, size: int) -> np.ndarray:
    """Mean over every fully contained size x size window.

    Returns only the 'valid' positions, which is exactly the region scikit-image keeps after it
    crops the border -- so no border-handling convention has to be guessed at.
    """
    integral = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def _ssim_plane(x: np.ndarray, y: np.ndarray, data_range: float) -> float:
    window = SSIM_WINDOW
    if min(x.shape) < window:
        raise MetricError(f"SSIM needs at least {window}x{window} pixels, got {x.shape}")

    count = window * window
    covariance_norm = count / (count - 1)  # skimage uses the sample covariance, not the population

    ux, uy = _box_mean(x, window), _box_mean(y, window)
    uxx, uyy = _box_mean(x * x, window), _box_mean(y * y, window)
    uxy = _box_mean(x * y, window)

    vx = covariance_norm * (uxx - ux * ux)
    vy = covariance_norm * (uyy - uy * uy)
    vxy = covariance_norm * (uxy - ux * uy)

    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    similarity = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return float(similarity.mean())


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Mean structural similarity over the channels of an NCHW pair."""
    x, y = _pair(_nchw(a), _nchw(b))
    planes = [_ssim_plane(x[0, c], y[0, c], data_range) for c in range(x.shape[1])]
    return float(np.mean(planes))


def lpips_distance(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS (AlexNet trunk) on CPU, for NCHW tensors in [0,1].

    Recorded, never gated: certification spec §4 warns that perceptual metrics can reward
    sharp-but-hallucinated detail, so LPIPS is a signal and Level D is the verdict. torch and lpips
    are imported here, not at module scope, so this file stays importable with numpy alone.
    """
    import torch  # noqa: PLC0415  (import policy: heavy imports stay inside their function)

    x, y = _pair(_nchw(a), _nchw(b))
    network = _lpips_network()
    with torch.no_grad():
        scaled_x = torch.from_numpy(x.astype(np.float32) * 2.0 - 1.0)
        scaled_y = torch.from_numpy(y.astype(np.float32) * 2.0 - 1.0)
        distance = network(scaled_x, scaled_y)
    return float(distance.reshape(-1)[0])


_LPIPS_CACHE: dict = {}


def _lpips_network():
    """One network per process. Reloading per tile would be slow and would say nothing new."""
    import lpips  # noqa: PLC0415

    if LPIPS_NET not in _LPIPS_CACHE:
        network = lpips.LPIPS(net=LPIPS_NET)
        network.eval()
        _LPIPS_CACHE[LPIPS_NET] = network
    return _LPIPS_CACHE[LPIPS_NET]


def lpips_version() -> str:
    """Attribution for the manifest: which lpips produced the numbers."""
    import importlib.metadata  # noqa: PLC0415

    return importlib.metadata.version("lpips")
