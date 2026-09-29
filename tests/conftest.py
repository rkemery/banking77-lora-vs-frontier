"""Shared fixtures. Tests that download from Hugging Face are skipped unless B77_NETWORK_TESTS=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("B77_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="downloads from Hugging Face; set B77_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every path in the package is relative to the repo root, as when run from the Makefile."""
    monkeypatch.chdir(REPO)
