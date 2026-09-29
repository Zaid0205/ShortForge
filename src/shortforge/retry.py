"""One retry policy for every external API call.

Transient failures (network errors, timeouts, HTTP 408/409/429/5xx) are retried with
exponential backoff and jitter. When a server sends ``Retry-After`` (free-tier rate
limits usually do), that delay is honored instead. Anything else, such as a bad key
or a malformed request, fails immediately so the error surfaces right away.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import groq
import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from shortforge.log import console

T = TypeVar("T")

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0


def _status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from an httpx or Groq SDK exception."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code
    return None


def is_transient(exc: BaseException) -> bool:
    """Return True if `exc` is worth retrying."""
    if isinstance(exc, httpx.TransportError | groq.APIConnectionError):
        return True
    return _status_code(exc) in RETRYABLE_STATUS


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """Return the server's ``Retry-After`` delay in seconds, capped, if it sent one."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        delay = float(response.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(delay, MAX_RETRY_AFTER_SECONDS))


class wait_retry_after(wait_base):  # noqa: N801  (tenacity naming convention)
    """Wait for the server's ``Retry-After`` if present, else defer to `fallback`."""

    def __init__(self, fallback: wait_base) -> None:
        self.fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        delay = retry_after_seconds(exc)
        return delay if delay is not None else self.fallback(retry_state)


DEFAULT_WAIT = wait_retry_after(wait_exponential_jitter(initial=1, max=30, jitter=1))


def call_with_retries(
    fn: Callable[[], T], *, what: str, attempts: int, wait: wait_base = DEFAULT_WAIT
) -> T:
    """Call `fn`, retrying transient failures.

    Args:
        fn: Zero-argument callable that performs one API request.
        what: Short label for log messages, for example ``"Groq request"``.
        attempts: Maximum number of attempts, including the first.
        wait: Tenacity wait strategy; tests pass a zero wait.

    Returns:
        Whatever `fn` returns.

    Raises:
        The last exception if every attempt fails, or the first non-transient one.
    """

    def log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        delay = state.next_action.sleep if state.next_action else 0
        console.print(
            f"  [yellow]![/] {what} failed ({type(exc).__name__}: {exc}); "
            f"retry {state.attempt_number}/{attempts - 1} in {delay:.1f}s"
        )

    retrying = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait,
        retry=retry_if_exception(is_transient),
        before_sleep=log_retry,
        reraise=True,
    )
    return retrying(fn)
