"""Tests for the shared retry policy and atomic file writes."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from shortforge.fsutil import atomic_path
from shortforge.retry import call_with_retries, is_transient, retry_after_seconds


def status_error(status: int, **headers: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, request=request, headers=headers)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


@pytest.mark.parametrize(
    ("status", "expected"), [(429, True), (503, True), (400, False), (401, False)]
)
def test_transient_status_codes(status: int, expected: bool) -> None:
    assert is_transient(status_error(status)) is expected


def test_network_errors_are_transient() -> None:
    assert is_transient(httpx.ConnectTimeout("timed out"))


def test_retry_after_is_parsed_and_capped() -> None:
    assert retry_after_seconds(status_error(429, **{"retry-after": "7"})) == 7.0
    assert retry_after_seconds(status_error(429, **{"retry-after": "9999"})) == 60.0
    assert retry_after_seconds(status_error(429)) is None


def test_transient_failures_are_retried() -> None:
    outcomes: list[Exception | str] = [status_error(503), status_error(429), "ok"]

    def flaky() -> str:
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    assert call_with_retries(flaky, what="test", attempts=3, wait=wait_none()) == "ok"


def test_permanent_failures_are_not_retried() -> None:
    calls = 0

    def forbidden() -> None:
        nonlocal calls
        calls += 1
        raise status_error(403)

    with pytest.raises(httpx.HTTPStatusError):
        call_with_retries(forbidden, what="test", attempts=5, wait=wait_none())
    assert calls == 1


def test_atomic_path_moves_file_into_place(tmp_path: Path) -> None:
    target = tmp_path / "scene_01.wav"
    with atomic_path(target) as tmp:
        assert tmp.name == "scene_01.tmp.wav"
        tmp.write_bytes(b"data")
    assert target.read_bytes() == b"data"
    assert not tmp.exists()


def test_atomic_path_leaves_nothing_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "scene_01.png"
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_bytes(b"partial")
        raise RuntimeError("render crashed")
    assert list(tmp_path.iterdir()) == []
