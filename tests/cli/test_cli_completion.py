"""Tests for the ``pyramids completion`` command group (E-5 / #1493)."""

from __future__ import annotations

import pytest

from pyramids.cli import main

pytestmark = pytest.mark.core


class TestCompletion:
    """`pyramids completion <shell>`."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_emits_nonempty_hook(self, shell, capsys):
        """Each supported shell prints a non-empty argcomplete hook and exits 0."""
        assert main(["completion", shell]) == 0, f"completion {shell} should succeed"
        out = capsys.readouterr().out
        assert "register-python-argcomplete" in out, "the hook should call argcomplete"
        assert "pyramids" in out, "the hook should target the pyramids program"

    def test_fish_uses_fish_syntax(self, capsys):
        """The fish hook uses fish's ``| source`` form, not bash ``eval``."""
        assert main(["completion", "fish"]) == 0
        assert "| source" in capsys.readouterr().out, "fish uses `... | source`"

    def test_unknown_shell_is_rejected(self, capsys):
        """An unsupported shell is an argparse error (exit 2), not a traceback."""
        with pytest.raises(SystemExit) as exc:
            main(["completion", "powershell"])
        assert exc.value.code == 2, "argparse rejects an invalid choice with code 2"
