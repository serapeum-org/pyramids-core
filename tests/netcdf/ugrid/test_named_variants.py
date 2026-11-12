"""Tests for the named curve-fit models and focal reducers registries (#1542).

These registries back the ``mesh curvefit --model`` and ``mesh focal-reduce --reducer`` CLI
selectors; here they are exercised directly so the model maths and the reducer mapping are
covered independently of the CLI wiring.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyramids.netcdf.ugrid.curve_models import CURVE_MODELS
from pyramids.netcdf.ugrid.focal import FOCAL_REDUCERS

pytestmark = pytest.mark.core


class TestCurveModels:
    """The ``CURVE_MODELS`` registry of ``f(x, *params)`` fitting models."""

    def test_registry_names(self):
        """The registry exposes exactly the documented model names."""
        assert sorted(CURVE_MODELS) == [
            "exponential",
            "gaussian",
            "linear",
            "logistic",
            "sine",
        ]

    @pytest.mark.parametrize(
        "name, arity",
        [
            ("linear", 2),
            ("exponential", 2),
            ("logistic", 3),
            ("gaussian", 3),
            ("sine", 4),
        ],
    )
    def test_parameter_arity(self, name, arity):
        """Each model declares the expected number of fitted parameters after ``x``.

        Args:
            name: The model name.
            arity: The number of positional parameters after ``x`` curve_fit will solve for.
        """
        params = list(inspect.signature(CURVE_MODELS[name]).parameters)
        assert params[0] == "x"
        assert len(params) - 1 == arity, f"{name}: {params}"

    def test_linear_value(self):
        """``linear`` evaluates ``a * x + b``."""
        assert float(CURVE_MODELS["linear"](2.0, 3.0, 1.0)) == pytest.approx(7.0)

    def test_exponential_value(self):
        """``exponential`` evaluates ``a * exp(b * x)`` elementwise."""
        got = CURVE_MODELS["exponential"](np.array([0.0, 1.0]), 2.0, np.log(2.0))
        assert np.allclose(got, [2.0, 4.0])

    def test_gaussian_peaks_at_the_mean(self):
        """``gaussian`` peaks at ``x == mean`` with value ``amplitude``."""
        assert float(CURVE_MODELS["gaussian"](5.0, 3.0, 5.0, 1.0)) == pytest.approx(3.0)

    def test_logistic_midpoint_is_half_capacity(self):
        """``logistic`` passes through ``capacity / 2`` at its midpoint."""
        assert float(CURVE_MODELS["logistic"](0.0, 4.0, 1.0, 0.0)) == pytest.approx(2.0)

    def test_sine_value(self):
        """``sine`` evaluates ``a * sin(b * x + c) + d``."""
        got = CURVE_MODELS["sine"](0.0, 2.0, 1.0, np.pi / 2, 1.0)
        assert float(got) == pytest.approx(3.0)


class TestFocalReducers:
    """The ``FOCAL_REDUCERS`` registry of 1-D neighbourhood reducers."""

    def test_registry_names(self):
        """The registry exposes the named reducers, excluding mean/std (their own verbs)."""
        assert sorted(FOCAL_REDUCERS) == ["max", "median", "min", "range", "sum", "var"]
        assert "mean" not in FOCAL_REDUCERS
        assert "std" not in FOCAL_REDUCERS

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("median", 2.0),
            ("min", 1.0),
            ("max", 9.0),
            ("sum", 12.0),
            ("range", 8.0),
            ("var", 38.0 / 3.0),
        ],
    )
    def test_reducer_values(self, name, expected):
        """Each reducer collapses ``[1, 2, 9]`` to its known scalar.

        Args:
            name: The reducer name.
            expected: The scalar it should produce for ``[1, 2, 9]``.
        """
        got = float(FOCAL_REDUCERS[name](np.array([1.0, 2.0, 9.0])))
        assert got == pytest.approx(expected)
