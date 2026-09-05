import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from ml.monitoring.drift import _psi_categorical, _psi_numeric, _status_for_psi


class _FakeSettings:
    DRIFT_PSI_WARNING = 0.10
    DRIFT_PSI_CRITICAL = 0.25


def test_psi_near_zero_for_identical_distributions():
    np.random.seed(0)
    ref = pd.Series(np.random.normal(100, 10, 500))
    cur = pd.Series(np.random.normal(100, 10, 500))
    psi = _psi_numeric(ref, cur)
    assert abs(psi) < 0.05  # same distribution -> should not trip even the WARNING threshold


def test_psi_large_for_shifted_distribution():
    np.random.seed(0)
    ref = pd.Series(np.random.normal(100, 10, 500))
    shifted = pd.Series(np.random.normal(180, 10, 500))  # 8 std devs away -- unambiguous shift
    psi = _psi_numeric(ref, shifted)
    assert psi > 0.25  # should clearly trip CRITICAL


def test_psi_numeric_handles_too_little_data_gracefully():
    ref = pd.Series([1.0, 2.0])
    cur = pd.Series([1.0, 2.0])
    psi = _psi_numeric(ref, cur)
    assert psi == 0.0  # not an error, not a false-positive drift signal


def test_psi_categorical_zero_for_identical_distribution():
    ref = pd.Series(["A"] * 50 + ["B"] * 50)
    cur = pd.Series(["A"] * 50 + ["B"] * 50)
    psi = _psi_categorical(ref, cur)
    assert abs(psi) < 0.01


def test_psi_categorical_large_for_new_dominant_category():
    ref = pd.Series(["A"] * 90 + ["B"] * 10)
    cur = pd.Series(["A"] * 10 + ["B"] * 90)  # flipped dominance
    psi = _psi_categorical(ref, cur)
    assert psi > 0.25


def test_status_thresholds():
    assert _status_for_psi(0.02, _FakeSettings()) == ("NORMAL", False)
    assert _status_for_psi(0.15, _FakeSettings()) == ("WARNING", True)
    assert _status_for_psi(0.30, _FakeSettings()) == ("CRITICAL", True)


def test_status_thresholds_are_configurable():
    class StricterSettings:
        DRIFT_PSI_WARNING = 0.01
        DRIFT_PSI_CRITICAL = 0.02

    # The same PSI value that was NORMAL under default thresholds trips
    # WARNING under stricter configured thresholds -- proves thresholds
    # aren't hardcoded into the comparison logic.
    assert _status_for_psi(0.02, StricterSettings())[0] != _status_for_psi(0.02, _FakeSettings())[0]
