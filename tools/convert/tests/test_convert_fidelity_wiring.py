"""`convert.py`'s side of the prior-manifest comparison.

Separate module because importing `convert` pulls in torch and coremltools (~1.9 s). The pure
comparison lives in `manifest.py` and is tested in `test_fidelity.py` without that cost; what is
left here is the wiring, and the branch that matters most is the one where there is no prior
manifest at all — a first conversion must not be blocked by a check designed to protect evidence
that does not exist yet.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("coremltools")
pytest.importorskip("yaml")
pytest.importorskip("PIL")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert  # noqa: E402
from test_manifest import VALID  # noqa: E402


def test_a_first_conversion_has_nothing_to_compare_and_is_not_blocked(tmp_path, capsys):
    absent = tmp_path / "manifest.json"

    assert convert.check_against_prior(absent, VALID) == []
    assert "no prior manifest" in capsys.readouterr().out


def test_an_agreeing_re_conversion_reports_what_it_compared(tmp_path, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(VALID))

    assert convert.check_against_prior(path, copy.deepcopy(VALID)) == []
    out = capsys.readouterr().out
    assert "all 13 fidelity fields" in out
    assert "not compared (expected to vary)" in out, "silence about the exclusions invites the " \
        "reader to assume the mlpackage hash was checked, which is the belief that produced the defect"


def test_a_drifting_re_conversion_returns_the_differences(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(VALID))
    rerun = copy.deepcopy(VALID)
    rerun["weights"]["sha256"] = "c" * 64

    drift = convert.check_against_prior(path, rerun)

    assert len(drift) == 1
    assert drift[0].startswith("weights.sha256: ")


def test_an_unreadable_prior_manifest_stops_the_run(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ truncated")

    with pytest.raises(convert.ManifestError, match="refusing to overwrite"):
        convert.check_against_prior(path, VALID)
