"""Tests for the no-dependency CLI progress reporter (E-3 / #1491)."""

from __future__ import annotations

import argparse
import io

import pytest

from pyramids._cli_shared import ProgressReporter
from pyramids.cli import main

pytestmark = pytest.mark.core


class _FakeTTY(io.StringIO):
    """A StringIO that claims to be a TTY so the reporter draws."""

    def isatty(self) -> bool:
        """Report as an interactive terminal."""
        return True


def _args(progress=True, quiet=False):
    """Build a namespace with the progress/quiet flags the reporter reads."""
    return argparse.Namespace(progress=progress, quiet=quiet)


class TestProgressReporterEnablement:
    """When the reporter is active vs a silent no-op."""

    def test_enabled_on_tty_with_flag(self):
        """`--progress` on an interactive terminal enables drawing."""
        reporter = ProgressReporter(_args(progress=True), 3, stream=_FakeTTY())
        assert reporter.enabled is True, "progress on a TTY should be enabled"

    def test_disabled_without_flag(self):
        """Without ``--progress`` the reporter is a no-op even on a TTY."""
        reporter = ProgressReporter(_args(progress=False), 3, stream=_FakeTTY())
        assert reporter.enabled is False, "no --progress means no drawing"

    def test_disabled_when_quiet(self):
        """``--quiet`` suppresses the reporter even with ``--progress``."""
        reporter = ProgressReporter(
            _args(progress=True, quiet=True), 3, stream=_FakeTTY()
        )
        assert reporter.enabled is False, "--quiet should win over --progress"

    def test_disabled_on_non_tty(self):
        """A non-TTY stream (pipe/log/CI) disables the reporter."""
        reporter = ProgressReporter(_args(progress=True), 3, stream=io.StringIO())
        assert reporter.enabled is False, "a non-TTY stream should stay clean"


class TestProgressReporterOutput:
    """What the reporter writes as it advances."""

    def test_advance_draws_fraction_and_percent(self):
        """Each advance redraws ``label: done/total (pct%)`` with a carriage return."""
        stream = _FakeTTY()
        reporter = ProgressReporter(_args(), 4, label="windows", stream=stream)
        reporter.advance()
        reporter.advance()
        assert "\rwindows: 2/4 (50%)" in stream.getvalue(), "should show 2/4 at 50%"

    def test_unknown_total_shows_count_only(self):
        """With ``total=None`` the reporter shows a running count, no percentage."""
        stream = _FakeTTY()
        reporter = ProgressReporter(_args(), None, label="items", stream=stream)
        reporter.advance()
        assert "\ritems: 1" in stream.getvalue(), (
            "count-only line when total is unknown"
        )
        assert "%" not in stream.getvalue(), "no percentage without a known total"

    def test_close_writes_newline(self):
        """Closing ends the bar line with a newline so later output is separated."""
        stream = _FakeTTY()
        reporter = ProgressReporter(_args(), 1, stream=stream)
        reporter.advance()
        reporter.close()
        assert stream.getvalue().endswith("\n"), "close should terminate the line"

    def test_context_manager_closes(self):
        """Using it as a context manager draws inside and closes on exit."""
        stream = _FakeTTY()
        with ProgressReporter(_args(), 2, label="w", stream=stream) as reporter:
            reporter.advance()
        assert stream.getvalue().endswith("\n"), "exit should close the line"

    def test_noop_reporter_writes_nothing(self):
        """A disabled reporter writes nothing at all."""
        stream = io.StringIO()
        reporter = ProgressReporter(_args(progress=False), 2, stream=stream)
        reporter.advance()
        reporter.close()
        assert stream.getvalue() == "", "a disabled reporter must stay silent"

    def test_zero_total_shows_count_only(self):
        """A ``total`` of 0 is falsy, so the reporter shows a count without a percentage."""
        stream = _FakeTTY()
        reporter = ProgressReporter(_args(), 0, label="items", stream=stream)
        reporter.advance()
        assert "\ritems: 1" in stream.getvalue(), (
            "zero total falls back to a plain count"
        )
        assert "%" not in stream.getvalue(), "no percentage when total is zero"


class TestProgressFlagPlumbing:
    """The global ``--progress`` flag parses and does not disturb a verb."""

    def test_flag_accepted_before_command(self, cli_points, capsys):
        """`--progress` placed before the command is accepted and the verb still runs."""
        assert main(["--progress", "vector", "info", cli_points]) == 0
        assert "features: 5" in capsys.readouterr().out, "the verb output is unchanged"
