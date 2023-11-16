"""The prior-manifest comparison that makes `--check` mean something.

`convert.py --check` reads `manifest.json` from disk and compares the artifact to it. Run after a
conversion — which is the only order in which the artifact exists — both sides are that run's own
output, so the check is self-consistent by construction and can never fail for the reason it looks
like it is testing. A reproducibility check found this: recovering the committed
hash needed `git show`, a step no tool performs.

The fix compares the freshly derived manifest to the one already on disk *before* overwriting it,
which is the last instant the committed evidence still exists. These tests cover that comparison.
Expected values are written out as literals, never recomputed from the code under test.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest import (  # noqa: E402
    FIDELITY_FIELDS, VOLATILE_FIELDS, ManifestError, compare_to_prior, drift_against_prior,
)
from test_manifest import VALID  # noqa: E402


def test_an_identical_re_conversion_shows_no_drift():
    assert compare_to_prior(VALID, copy.deepcopy(VALID)) == []


def test_the_volatile_fields_do_not_count_as_drift():
    """A re-conversion that differs ONLY in these is the documented normal case.

    tools/artifacts/README.md: a .mlpackage is "re-derived by running the pinned conversion, not by
    matching a hash". The reproducibility check confirmed it — bytes differed, all three Level A
    metrics were bit-identical to the last float64 digit.
    """
    rerun = copy.deepcopy(VALID)
    rerun["outputs"]["mlpackage_sha256"] = "f" * 64
    rerun["created_utc"] = "2027-01-01T00:00:00+00:00"
    rerun["conversion"]["coremltools"] = "9.0.0"
    rerun["conversion"]["torch"] = "2.9.0"
    rerun["conversion"]["python"] = "3.13.0"
    rerun["conversion"]["compute_units"] = "CPU_ONLY"
    rerun["conversion"]["script_version"] = "2.0.0"

    assert compare_to_prior(VALID, rerun) == []


@pytest.mark.parametrize(
    "path, replacement",
    [
        (("weights", "sha256"), "a" * 64),
        (("weights", "parameters"), 1213296),
        (("conversion", "input_shape"), [1, 3, 512, 512]),
        (("conversion", "output_shape"), [1, 3, 2048, 2048]),
        (("conversion", "precision"), "float32"),
        (("outputs", "reference_metrics", "psnr_db"), 41.2),
    ],
)
def test_each_fidelity_field_is_caught_and_named(path, replacement):
    rerun = copy.deepcopy(VALID)
    node = rerun
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement

    drift = compare_to_prior(VALID, rerun)

    assert len(drift) == 1
    assert drift[0].startswith(".".join(path) + ": ")


def test_the_smallest_possible_metric_change_is_still_drift():
    """The metrics are the certified behaviour. A comparison that tolerates "close enough" would
    let the oracle move by one ulp per conversion and never report a thing."""
    rerun = copy.deepcopy(VALID)
    one_ulp_up = math.nextafter(0.0004, 1.0)
    rerun["outputs"]["reference_metrics"]["mae"] = one_ulp_up

    assert one_ulp_up != 0.0004, "guard the test itself: a literal like 0.00040000000000000002 " \
        "parses to the SAME float64 as 0.0004, and would make this test pass vacuously"
    assert len(compare_to_prior(VALID, rerun)) == 1


def test_several_differences_are_all_reported_not_just_the_first():
    rerun = copy.deepcopy(VALID)
    rerun["weights"]["sha256"] = "b" * 64
    rerun["scale_factor"] = 2
    rerun["outputs"]["reference_metrics"]["max_abs"] = 0.5

    assert len(compare_to_prior(VALID, rerun)) == 3


def test_a_field_missing_from_the_prior_manifest_is_drift_not_agreement():
    """A manifest that has lost a fidelity field cannot vouch for the artifact."""
    prior = copy.deepcopy(VALID)
    del prior["weights"]["sha256"]

    drift = compare_to_prior(prior, VALID)

    assert len(drift) == 1
    assert "<absent>" in drift[0]


def test_a_field_missing_from_this_run_is_also_drift():
    rerun = copy.deepcopy(VALID)
    del rerun["outputs"]["reference_metrics"]["psnr_db"]

    assert "<absent>" in compare_to_prior(VALID, rerun)[0]


def test_a_field_absent_from_BOTH_manifests_is_drift_not_agreement():
    """Found by mutation. Simplifying the guard to a bare `was == now` leaves this passing,
    because both sides are then the same sentinel and compare equal — two manifests that agree
    only in what they have both lost would be reported as agreeing."""
    prior = copy.deepcopy(VALID)
    rerun = copy.deepcopy(VALID)
    del prior["weights"]["sha256"]
    del rerun["weights"]["sha256"]

    drift = compare_to_prior(prior, rerun)

    assert drift == ["weights.sha256: manifest <absent> != this run <absent>"]


def test_an_empty_prior_manifest_disagrees_with_everything():
    """Not a pass. An absent manifest is handled by the caller, which says so explicitly;
    a *present but empty* one is a corrupted identity document."""
    assert len(compare_to_prior({}, VALID)) == len(FIDELITY_FIELDS) == 13


def test_no_field_is_both_compared_and_excused():
    """The two lists must not overlap: a field in both would be silently unenforced."""
    compared = {".".join(path) for path in FIDELITY_FIELDS}

    assert compared.isdisjoint(set(VOLATILE_FIELDS))


def test_mlpackage_sha256_is_deliberately_not_a_fidelity_field():
    """Pinned so that "tighten the check" never quietly makes every re-conversion fail."""
    assert "outputs.mlpackage_sha256" in VOLATILE_FIELDS
    assert ("outputs", "mlpackage_sha256") not in FIDELITY_FIELDS


def test_drift_against_a_manifest_on_disk(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(VALID))
    rerun = copy.deepcopy(VALID)
    rerun["weights"]["bytes"] = 1

    assert drift_against_prior(path, VALID) == []
    assert len(drift_against_prior(path, rerun)) == 1


def test_an_unreadable_prior_manifest_raises_rather_than_being_overwritten(tmp_path):
    """The dangerous failure mode: treating unparseable JSON as "no prior manifest" would make
    a corrupted file the easiest way to erase the evidence it was supposed to hold."""
    path = tmp_path / "manifest.json"
    path.write_text("{ truncated")

    with pytest.raises(ManifestError, match="refusing to overwrite"):
        drift_against_prior(path, VALID)


def test_a_json_scalar_where_the_manifest_should_be_is_refused(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('"not a manifest"')

    with pytest.raises(ManifestError, match="expected a JSON object"):
        drift_against_prior(path, VALID)
