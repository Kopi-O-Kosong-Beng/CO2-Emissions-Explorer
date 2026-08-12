"""
Tests for the static web export.

The browser app under `web/` reimplements the prediction in JavaScript, so the
important guarantee here is that the exported coefficients reproduce
`data_service.predict_co2` exactly. If those two ever drift, the hosted
explorer would quietly disagree with the Streamlit app.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_service import _FEATURES, _STATES, predict_co2  # noqa: E402
from scripts.export_web_data import (  # noqa: E402
    OUT_PATH,
    TRAIN_END,
    build_payload,
    t_two_sided_p,
)


@pytest.fixture(scope="module")
def payload():
    return build_payload()


def test_payload_has_expected_sections(payload):
    assert set(payload) == {"meta", "national", "spread", "cases"}
    assert payload["meta"]["observations"] == 1300
    assert payload["meta"]["states"] == 50
    assert payload["meta"]["year_range"] == [1998, 2023]


def test_spread_covers_every_state_in_ascending_order(payload):
    values = payload["spread"]["values"]
    assert len(values) == 50
    assert len({row["state"] for row in values}) == 50

    ordered = [row["value"] for row in values]
    assert ordered == sorted(ordered)

    for row in values:
        assert row["name"], f"{row['state']} has no display name"


def test_national_series_aligns_with_the_year_axis(payload):
    national = payload["national"]
    assert len(national["years"]) == len(national["median"]) == 26
    # the median must sit inside its own quartiles
    for low, mid, high in zip(national["p25"], national["median"], national["p75"]):
        assert low <= mid <= high


@pytest.mark.parametrize("state", _STATES)
def test_case_structure(payload, state):
    case = payload["cases"][state]

    assert len(case["years"]) == len(case["co2"]) == 26
    assert case["years"] == sorted(case["years"])
    assert set(case["features"]) == set(_FEATURES)
    assert set(case["model"]["coefficients"]) == set(_FEATURES)
    assert set(case["model"]["significant"]) <= set(_FEATURES)

    assert 0.0 < case["fit"]["adj_r_squared"] <= 1.0
    assert case["fit"]["n"] == 26


@pytest.mark.parametrize("state", _STATES)
def test_exported_coefficients_reproduce_the_python_prediction(payload, state):
    """The JavaScript app computes intercept + sum(coefficient * input)."""
    model = payload["cases"][state]["model"]

    # A non-trivial point: each feature held at its own observed maximum.
    inputs = {item["key"]: item["max"] for item in payload["cases"][state]["inputs"]}

    from_export = model["intercept"] + sum(
        model["coefficients"][key] * inputs[key] for key in _FEATURES
    )
    from_service = predict_co2(
        state,
        inputs["renewable energy"],
        inputs["Coal Electricity Consumption"],
        inputs["Natural Gas Electricity Consumption"],
        inputs["PCE per capita"],
        inputs["Estimated Urban Population"],
    )

    assert from_export == pytest.approx(from_service, rel=1e-12)


@pytest.mark.parametrize("state", _STATES)
def test_slider_bounds_contain_their_default(payload, state):
    for item in payload["cases"][state]["inputs"]:
        assert item["min"] <= item["default"] <= item["max"], item["key"]
        assert item["step"] > 0


@pytest.mark.parametrize("state", _STATES)
def test_holdout_never_touches_the_test_years(payload, state):
    holdout = payload["cases"][state]["holdout"]

    assert holdout["train_years"][1] == TRAIN_END
    assert holdout["test_years"][0] == TRAIN_END + 1
    assert holdout["train_n"] + holdout["test_n"] == 26
    assert holdout["rmse"] > 0
    assert len(holdout["rows"]) == holdout["test_n"]


def test_standard_errors_survive_the_ill_conditioned_design(payload):
    """
    The five predictors differ by ~9 orders of magnitude. Inverting XtX at that
    scale returns noise, so the export uses a QR decomposition instead. This
    guards the symptom that gave it away: an absurd intercept t statistic.
    """
    for state in _STATES:
        terms = payload["cases"][state]["fit"]["terms"]
        assert abs(terms["intercept"]["t"]) < 1e4, state
        for name, term in terms.items():
            assert term["std_error"] > 0, f"{state}/{name}"
            assert 0.0 <= term["p"] <= 1.0, f"{state}/{name}"


def test_t_distribution_matches_known_critical_values():
    # two-sided 5% critical value for 20 degrees of freedom is 2.086
    assert t_two_sided_p(2.086, 20) == pytest.approx(0.05, abs=1e-3)
    # large degrees of freedom converge on the normal 1.96
    assert t_two_sided_p(1.95996, 1_000_000) == pytest.approx(0.05, abs=1e-3)
    assert t_two_sided_p(0.0, 10) == pytest.approx(1.0)


def test_committed_json_is_in_step_with_the_generator(payload):
    """The committed export should not lag behind the code that writes it."""
    assert OUT_PATH.exists(), "run: python scripts/export_web_data.py"
    committed = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    assert committed["meta"]["observations"] == payload["meta"]["observations"]
    assert set(committed["cases"]) == set(payload["cases"])

    for state in _STATES:
        assert committed["cases"][state]["model"]["intercept"] == pytest.approx(
            payload["cases"][state]["model"]["intercept"], rel=1e-9
        )
        assert (
            committed["cases"][state]["model"]["significant"]
            == payload["cases"][state]["model"]["significant"]
        )
