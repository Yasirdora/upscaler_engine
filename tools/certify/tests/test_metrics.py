"""Tests for the Level A/B metric implementations.

Every expected value below is an **externally computed constant**. None of them was produced by
`metrics.py`; each was computed once with scikit-image 0.25.2 / numpy 2.0.2 and pasted in. A test
that recomputed its expectation with the code under test would ratify whatever that code did,
including a regression -- and these numbers decide whether a model ships (certification spec §4).

Reproduce the constants with (scikit-image installed; this file itself never imports it):

    import numpy as np
    from skimage.metrics import (peak_signal_noise_ratio as sk_psnr,
                                 structural_similarity as sk_ssim,
                                 mean_squared_error as sk_mse)
    hwc = lambda t: np.ascontiguousarray(t[0].transpose(1, 2, 0))
    sk_psnr(hwc(a), hwc(b), data_range=1.0)
    sk_ssim(hwc(a), hwc(b), data_range=1.0, channel_axis=2)
    sk_mse(hwc(a), hwc(b))
    np.abs(hwc(a) - hwc(b)).mean()          # MAE -- skimage has no MAE, numpy is the oracle
    np.abs(hwc(a) - hwc(b)).max()           # max abs
    (np.abs(a - b).max(axis=1) <= tol).mean()   # within-tolerance fraction, per pixel

Runs with pytest + numpy only, which is CI's environment (see the import-policy tests).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics  # noqa: E402

#: skimage agreed with this implementation to 0 ULP on PSNR/MSE and to 5.4e-14 on SSIM when the
#: comparison was run; 1e-12 leaves headroom for accumulation order without hiding a real change.
TOLERANCE = 1e-12


def fixture_a() -> tuple[np.ndarray, np.ndarray]:
    """Uniform noise vs itself plus N(0, 0.01), clipped -- the shape of fp16 conversion error."""
    rng = np.random.default_rng(20260821)
    a = rng.random((1, 3, 32, 32))
    b = np.clip(a + rng.normal(0.0, 0.01, size=a.shape), 0.0, 1.0)
    return a, b


def fixture_b() -> tuple[np.ndarray, np.ndarray]:
    """A contrast/brightness change: structurally identical, numerically far apart."""
    rng = np.random.default_rng(7)
    c = rng.random((1, 3, 24, 16))
    return c, np.clip(c * 0.9 + 0.05, 0.0, 1.0)


def fixture_c() -> tuple[np.ndarray, np.ndarray]:
    """A smooth gradient vs the same gradient shifted one pixel -- a structural error, unlike A."""
    yy, xx = np.mgrid[0:64, 0:64] / 63.0
    gradient = np.stack([yy, xx, (yy + xx) / 2.0])[None, ...]
    shifted = gradient.copy()
    shifted[0, :, 1:, :] = gradient[0, :, :-1, :]
    return gradient, shifted


EXPECTED = {
    "A": {
        "psnr": 39.9828037035205,
        "ssim": 0.9993961886292828,
        "mse": 0.00010039674431404686,
        "mae": 0.00798684756680209,
        "max_abs": 0.037142448468811984,
        "fraction_4_255": 0.6953125,
        "fraction_0_01": 0.314453125,
    },
    "B": {
        "psnr": 30.78554616225386,
        "ssim": 0.9944707228613678,
        "mse": 0.0008345365916025611,
        "mae": 0.025043573218807576,
        "max_abs": 0.04990587554798154,
        "fraction_4_255": 0.018229166666666668,
        "fraction_0_01": 0.010416666666666666,
    },
    "C": {
        "psnr": 39.85731765149075,
        "ssim": 0.9986307171309239,
        "mse": 0.00010333994708994708,
        "mae": 0.0078125,
        "max_abs": 0.015873015873015928,
        "fraction_4_255": 0.015625,
        "fraction_0_01": 0.015625,
    },
}

FIXTURES = {"A": fixture_a, "B": fixture_b, "C": fixture_c}
CASES = sorted(FIXTURES)


# --- Level A against external constants ---------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_psnr_matches_scikit_image(case):
    a, b = FIXTURES[case]()
    assert metrics.psnr(a, b) == pytest.approx(EXPECTED[case]["psnr"], abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES)
def test_mse_matches_scikit_image(case):
    a, b = FIXTURES[case]()
    assert metrics.mse(a, b) == pytest.approx(EXPECTED[case]["mse"], abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES)
def test_mae_matches_numpy(case):
    a, b = FIXTURES[case]()
    assert metrics.mae(a, b) == pytest.approx(EXPECTED[case]["mae"], abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES)
def test_max_abs_matches_numpy(case):
    a, b = FIXTURES[case]()
    assert metrics.max_abs_error(a, b) == pytest.approx(EXPECTED[case]["max_abs"], abs=TOLERANCE)


@pytest.mark.parametrize("case", CASES)
def test_within_tolerance_fraction_matches_numpy(case):
    a, b = FIXTURES[case]()
    assert metrics.within_tolerance_fraction(a, b, 4.0 / 255.0) == pytest.approx(
        EXPECTED[case]["fraction_4_255"], abs=TOLERANCE
    )
    assert metrics.within_tolerance_fraction(a, b, 0.01) == pytest.approx(
        EXPECTED[case]["fraction_0_01"], abs=TOLERANCE
    )


# --- Level B against external constants ---------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_ssim_matches_scikit_image(case):
    a, b = FIXTURES[case]()
    assert metrics.ssim(a, b) == pytest.approx(EXPECTED[case]["ssim"], abs=TOLERANCE)


def test_ssim_defaults_are_the_scikit_image_defaults():
    """The constants above are only meaningful next to the parameters that produced them."""
    assert (metrics.SSIM_WINDOW, metrics.SSIM_K1, metrics.SSIM_K2) == (7, 0.01, 0.03)


# --- semantics that no external library pins ----------------------------------------------
def test_identical_tensors_give_infinite_psnr_not_a_ceiling():
    a, _ = fixture_a()
    assert metrics.psnr(a, a) == math.inf
    assert metrics.mae(a, a) == 0.0
    assert metrics.max_abs_error(a, a) == 0.0


def test_identical_tensors_give_ssim_of_one():
    a, _ = fixture_a()
    assert metrics.ssim(a, a) == pytest.approx(1.0, abs=TOLERANCE)


def test_a_pixel_fails_if_any_one_channel_exceeds_the_tolerance():
    """Hand-computed: 4 pixels, one of which has a single out-of-tolerance channel -> 0.75."""
    a = np.zeros((1, 3, 2, 2), dtype=np.float64)
    b = np.zeros((1, 3, 2, 2), dtype=np.float64)
    b[0, 1, 1, 1] = 5.0 / 255.0  # green only, one pixel, just over 4/255

    assert metrics.within_tolerance_fraction(a, b, 4.0 / 255.0) == 0.75
    assert metrics.within_tolerance_fraction(a, b, 6.0 / 255.0) == 1.0


def test_float32_inputs_give_the_same_answer_as_float64():
    """The harness compares float32 tensors; the metric must not depend on the accumulator."""
    a, b = fixture_a()
    wide = metrics.psnr(a, b)
    narrow = metrics.psnr(a.astype(np.float32), b.astype(np.float32))
    assert narrow == pytest.approx(wide, abs=1e-4)


# --- refusals -----------------------------------------------------------------------------
def test_mismatched_shapes_are_refused_not_broadcast():
    a = np.zeros((1, 3, 8, 8))
    b = np.zeros((1, 3, 8, 9))
    with pytest.raises(metrics.MetricError) as error:
        metrics.psnr(a, b)
    assert "shape mismatch" in str(error.value)


def test_empty_tensors_are_refused():
    empty = np.zeros((1, 3, 0, 8))
    with pytest.raises(metrics.MetricError):
        metrics.mae(empty, empty)


@pytest.mark.parametrize("shape", [(3, 8, 8), (2, 3, 8, 8), (1, 3, 8, 8, 1)])
def test_non_nchw_tensors_are_refused(shape):
    array = np.zeros(shape)
    with pytest.raises(metrics.MetricError) as error:
        metrics.ssim(array, array)
    assert "NCHW" in str(error.value)


def test_ssim_refuses_a_window_it_cannot_fit():
    small = np.zeros((1, 3, 4, 4))
    with pytest.raises(metrics.MetricError) as error:
        metrics.ssim(small, small)
    assert "7x7" in str(error.value)


# --- import policy -----------------------------------------------------------------------
def test_heavy_imports_stay_inside_their_functions():
    """CI installs pytest + numpy. A module-scope torch/lpips/skimage import here would make every
    suite in the repository uncollectable -- the exact defect the import policy exists to prevent.
    """
    import ast

    forbidden = {"torch", "lpips", "skimage", "PIL", "yaml", "scipy"}
    offenders = []
    for node in ast.parse(Path(metrics.__file__).read_text()).body:
        if isinstance(node, ast.Import):
            offenders += [n.name.split(".")[0] for n in node.names if n.name.split(".")[0] in forbidden]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in forbidden:
            offenders.append(node.module)

    assert not offenders, f"metrics.py imports {offenders} at module scope (import policy)"
