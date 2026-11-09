"""The .env loader must never override the shell and never leak a value."""

from __future__ import annotations

import os

from memcheck.utilities import dotenv


class TestParse:
    def test_parses_the_documented_forms(self) -> None:
        values = dotenv.parse(
            "\n".join(
                [
                    "# a comment",
                    "",
                    "PLAIN=value",
                    "export EXPORTED=other",
                    'QUOTED="has spaces"',
                    "SINGLE='single'",
                    "TRAILING=value # inline comment",
                    "EMPTY=",
                    "no_equals_sign",
                ]
            )
        )
        assert values == {
            "PLAIN": "value",
            "EXPORTED": "other",
            "QUOTED": "has spaces",
            "SINGLE": "single",
            "TRAILING": "value",
            "EMPTY": "",
        }

    def test_a_hash_inside_a_quoted_value_is_kept(self) -> None:
        assert dotenv.parse('KEY="a#b"') == {"KEY": "a#b"}


class TestLoad:
    def test_missing_file_is_not_an_error(self, tmp_path) -> None:
        assert dotenv.load(tmp_path / "absent") == []

    def test_the_shell_wins_over_the_file(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / ".env"
        path.write_text("ALLOW_PAID_RUNS=true\n")
        monkeypatch.setenv("ALLOW_PAID_RUNS", "false")
        applied = dotenv.load(path)
        assert applied == []
        assert os.environ["ALLOW_PAID_RUNS"] == "false"

    def test_override_is_explicit(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / ".env"
        path.write_text("SOME_KEY=from_file\n")
        monkeypatch.setenv("SOME_KEY", "from_shell")
        assert dotenv.load(path, override=True) == ["SOME_KEY"]
        assert os.environ["SOME_KEY"] == "from_file"

    def test_unset_variables_are_populated(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / ".env"
        path.write_text("MEMCHECK_TEST_ONLY=1\n")
        monkeypatch.delenv("MEMCHECK_TEST_ONLY", raising=False)
        assert dotenv.load(path) == ["MEMCHECK_TEST_ONLY"]
        assert os.environ["MEMCHECK_TEST_ONLY"] == "1"


class TestDescribe:
    def test_secret_values_are_never_named(self) -> None:
        summary = dotenv.describe(
            ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MAX_EXPERIMENT_COST_USD"]
        )
        assert "value hidden" in summary
        assert "ANTHROPIC_API_KEY" in summary
        assert "MAX_EXPERIMENT_COST_USD" in summary

    def test_nothing_loaded_says_nothing(self) -> None:
        assert dotenv.describe([]) == ""
