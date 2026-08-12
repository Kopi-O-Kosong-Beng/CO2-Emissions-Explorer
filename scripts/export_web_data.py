"""
Export the panel dataset and the fitted state models to JSON for the static web build.

The browser app under `web/` has no Python runtime, so everything it needs is
precomputed here: the per-state time series, the OLS coefficients, the fit
diagnostics, and the holdout test. Coefficients come from `data_service`
itself rather than being refitted, so the hosted explorer and the Streamlit app
can never drift apart.

Run from anywhere:

    python scripts/export_web_data.py

Writes `web/data/co2.json` and prints a summary of what it computed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_service import _FEATURES, _STATES, fit_state_models, load_merged_data  # noqa: E402

OUT_PATH = ROOT / "web" / "data" / "co2.json"

TARGET = "co2 per capita"

# Last year used for training in the holdout test; everything after is unseen.
TRAIN_END = 2019

STATE_NAMES = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
    "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri",
    "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina",
    "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada", "NY": "New York",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VA": "Virginia",
    "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
    "WV": "West Virginia", "WY": "Wyoming",
}

# Presentation metadata for the five predictors, in model order.
FEATURE_META = {
    "renewable energy": {
        "label": "Renewable energy consumption",
        "short": "Renewables",
        "unit": "billion Btu per year",
        "unit_short": "bn Btu",
    },
    "Coal Electricity Consumption": {
        "label": "Coal consumed for electricity",
        "short": "Coal",
        "unit": "short tons per year",
        "unit_short": "short tons",
    },
    "Natural Gas Electricity Consumption": {
        "label": "Natural gas consumed for electricity",
        "short": "Natural gas",
        "unit": "thousand cubic feet per year",
        "unit_short": "k cu ft",
    },
    "PCE per capita": {
        "label": "Personal consumption expenditure",
        "short": "Consumer spending",
        "unit": "USD per person per year",
        "unit_short": "USD",
    },
    "Estimated Urban Population": {
        "label": "Urban population",
        "short": "Urban population",
        "unit": "people",
        "unit_short": "people",
    },
}


# ────────────────────────────────────────────────────────
# Statistics helpers (pure NumPy, no SciPy dependency)
# ────────────────────────────────────────────────────────
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function, via Lentz's method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
        return front * _betacf(a, b, x) / a
    front = math.exp(log_beta + b * math.log1p(-x) + a * math.log(x))
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t_stat: float, dof: int) -> float:
    """Two-sided p-value for a t statistic with `dof` degrees of freedom."""
    if dof <= 0 or not math.isfinite(t_stat):
        return float("nan")
    return betainc(0.5 * dof, 0.5, dof / (dof + t_stat * t_stat))


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Stack a leading column of 1s onto the five predictors."""
    raw = frame[_FEATURES].to_numpy(dtype=float)
    return np.hstack([np.ones((raw.shape[0], 1)), raw])


def standard_errors(X: np.ndarray, sigma_sq: float) -> np.ndarray:
    """
    Coefficient standard errors via QR decomposition.

    The five predictors span nine orders of magnitude (urban population ~1e5
    against natural gas ~1e7), which pushes the condition number of X to ~1e9
    and the condition number of XtX to ~1e18. Inverting XtX at that point
    returns numerical noise: it reports the intercept standard error as ~5e-10
    and turns a t statistic of 8.6 into 2.5e11, which then flips at least one
    predictor from "not significant" to "significant". QR keeps the arithmetic
    on X itself, so the errors stay trustworthy.
    """
    _, R = np.linalg.qr(X)
    R_inv = np.linalg.inv(R)
    return np.sqrt(np.diag(R_inv @ R_inv.T) * sigma_sq)


def fit_diagnostics(frame: pd.DataFrame, coefficients: np.ndarray) -> dict:
    """R squared, adjusted R squared, and per-term significance for one state."""
    X = design_matrix(frame)
    y = frame[TARGET].to_numpy(dtype=float)
    residuals = y - X @ coefficients

    n, k = X.shape
    n_predictors = k - 1
    dof = n - k

    ss_residual = float(residuals @ residuals)
    ss_total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_residual / ss_total
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / (n - n_predictors - 1)

    sigma_sq = ss_residual / dof
    errors = standard_errors(X, sigma_sq)
    t_stats = coefficients / errors

    terms = {}
    for name, beta, err, t_stat in zip(["intercept"] + _FEATURES, coefficients, errors, t_stats):
        p_value = t_two_sided_p(float(t_stat), dof)
        terms[name] = {
            "coefficient": float(beta),
            "std_error": float(err),
            "t": float(t_stat),
            "p": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    return {
        "n": int(n),
        "dof": int(dof),
        "r_squared": float(r_squared),
        "adj_r_squared": float(adjusted),
        "residual_std_error": float(math.sqrt(sigma_sq)),
        "condition_number": float(np.linalg.cond(X)),
        "terms": terms,
    }


def holdout_test(frame: pd.DataFrame) -> dict:
    """Refit on 1998..TRAIN_END, then score the untouched later years."""
    train = frame[frame["Year"] <= TRAIN_END]
    test = frame[frame["Year"] > TRAIN_END]

    beta, *_ = np.linalg.lstsq(
        design_matrix(train), train[TARGET].to_numpy(dtype=float), rcond=None
    )

    actual = test[TARGET].to_numpy(dtype=float)
    predicted = design_matrix(test) @ beta
    errors = predicted - actual

    rmse = float(np.sqrt((errors**2).mean()))
    observed_range = float(frame[TARGET].max() - frame[TARGET].min())

    return {
        "train_years": [int(train["Year"].min()), int(train["Year"].max())],
        "test_years": [int(test["Year"].min()), int(test["Year"].max())],
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "rmse": rmse,
        # RMSE as a share of how much this state's emissions actually moved,
        # which is the only way the three states compare on equal footing.
        "rmse_share_of_range": float(rmse / observed_range),
        "rmse_share_of_mean": float(rmse / actual.mean()),
        "mean_signed_error": float(errors.mean()),
        "rows": [
            {"year": int(year), "actual": float(a), "predicted": float(p), "error": float(p - a)}
            for year, a, p in zip(test["Year"], actual, predicted)
        ],
    }


def input_spec(frame: pd.DataFrame) -> list[dict]:
    """
    Slider bounds for each predictor, taken from what the state actually recorded.

    A linear model says nothing useful outside the range it was fitted on, so the
    controls stop where the observations stop instead of at a round number.
    """
    spec = []
    for feature in _FEATURES:
        column = frame[feature].dropna().astype(float)
        low, high = float(column.min()), float(column.max())
        latest = float(frame.loc[frame["Year"].idxmax(), feature])
        span = high - low
        # ~200 steps across the observed span, snapped to a readable increment.
        raw_step = span / 200 if span > 0 else 1.0
        magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1.0
        step = max(magnitude, 1.0) if high > 1000 else magnitude

        meta = FEATURE_META[feature]
        spec.append(
            {
                "key": feature,
                "label": meta["label"],
                "short": meta["short"],
                "unit": meta["unit"],
                "unit_short": meta["unit_short"],
                "min": low,
                "max": high,
                "default": latest,
                "step": float(step),
            }
        )
    return spec


def build_payload() -> dict:
    frame = load_merged_data()
    models = fit_state_models()

    years = sorted(int(y) for y in frame["Year"].unique())

    # National context: the spread every case study sits inside.
    by_year = frame.groupby("Year")[TARGET]
    national = {
        "years": years,
        "median": [float(v) for v in by_year.median().reindex(years)],
        "p25": [float(v) for v in by_year.quantile(0.25).reindex(years)],
        "p75": [float(v) for v in by_year.quantile(0.75).reindex(years)],
    }

    latest_year = max(years)
    latest = frame[frame["Year"] == latest_year].sort_values(TARGET)
    spread = [
        {
            "state": str(code),
            "name": STATE_NAMES.get(str(code), str(code)),
            "value": float(value),
        }
        for code, value in zip(latest["State"], latest[TARGET])
    ]

    cases = {}
    for rank, state in enumerate(_STATES, start=1):
        subset = (
            frame[frame["State"] == state]
            .dropna(subset=_FEATURES + [TARGET])
            .sort_values("Year")
        )

        coefficient_map = models[state]
        beta = np.array(
            [coefficient_map["intercept"]] + [coefficient_map[f] for f in _FEATURES], dtype=float
        )

        diagnostics = fit_diagnostics(subset, beta)
        significant = [f for f in _FEATURES if diagnostics["terms"][f]["significant"]]

        cases[state] = {
            "code": state,
            "name": STATE_NAMES[state],
            "rank": rank,
            "years": [int(y) for y in subset["Year"]],
            "co2": [float(v) for v in subset[TARGET]],
            "features": {f: [float(v) for v in subset[f]] for f in _FEATURES},
            "model": {
                "intercept": float(coefficient_map["intercept"]),
                "coefficients": {f: float(coefficient_map[f]) for f in _FEATURES},
                "significant": significant,
            },
            "fit": diagnostics,
            "holdout": holdout_test(subset),
            "inputs": input_spec(subset),
        }

    return {
        "meta": {
            "source": "assets/All main data (1998 to 2023).xlsx (sheet: merged)",
            "generated_by": "scripts/export_web_data.py",
            "observations": int(len(frame)),
            "states": int(frame["State"].nunique()),
            "year_range": [years[0], years[-1]],
            "target": TARGET,
            "target_unit": "tonnes CO2 per person per year",
            "features": _FEATURES,
            "feature_meta": FEATURE_META,
        },
        "national": national,
        "spread": {"year": latest_year, "values": spread},
        "cases": cases,
    }


def main() -> None:
    payload = build_payload()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    meta = payload["meta"]
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    print(
        f"  {meta['observations']} observations, {meta['states']} states, "
        f"{meta['year_range'][0]}-{meta['year_range'][1]}"
    )
    for state, case in payload["cases"].items():
        fit, holdout = case["fit"], case["holdout"]
        drivers = ", ".join(FEATURE_META[f]["short"] for f in case["model"]["significant"])
        print(
            f"  {state}  adj R2 {fit['adj_r_squared']:.3f}  "
            f"holdout RMSE {holdout['rmse']:.2f} "
            f"({holdout['rmse_share_of_range']:.0%} of range)  "
            f"significant: {drivers or 'none'}"
        )


if __name__ == "__main__":
    main()
