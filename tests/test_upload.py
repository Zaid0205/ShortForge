"""YouTube upload tests with a fake API client. No network access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httplib2
import pytest
from googleapiclient.errors import HttpError

from shortforge.config import ConfigError, Settings
from shortforge.models import Script
from shortforge.upload import (
    MAX_DESCRIPTION_BYTES,
    SCOPES,
    UploadError,
    YouTubeUploader,
    build_description,
    build_metadata,
    load_credentials,
)


def make_script(description: str = "How retrieval keeps AI answers grounded.") -> Script:
    narration = "One clear idea explained in plain words so any curious viewer can follow it."
    return Script.model_validate(
        {
            "title": "RAG in 45 seconds",
            "description": description,
            "tags": ["AI", "RAG", "LLM"],
            "scenes": [
                {
                    "narration": narration,
                    "search_query": "library shelf",
                    "image_prompt": "a quiet library aisle in soft daylight",
                }
            ]
            * 6,
        }
    )


class FakeStatus:
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def progress(self) -> float:
        return self.fraction


class FakeRequest:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.retries: list[int] = []

    def next_chunk(self, num_retries: int = 0) -> Any:
        self.retries.append(num_retries)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeService:
    def __init__(self, *replies: Any) -> None:
        self.request = FakeRequest(list(replies))
        self.calls: list[dict[str, Any]] = []

    def videos(self) -> FakeService:
        return self

    def insert(self, **kwargs: Any) -> FakeRequest:
        self.calls.append(kwargs)
        return self.request


def http_error(status: int, reason: str = "") -> HttpError:
    content = {"error": {"code": status, "message": "nope", "errors": [{"reason": reason}]}}
    return HttpError(httplib2.Response({"status": status}), json.dumps(content).encode())


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "final.mp4"
    path.write_bytes(b"\x00" * 1024)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        youtube_token=tmp_path / "token.json",
        youtube_client_secret=tmp_path / "client_secret.json",
        max_retries=3,
    )


def test_description_has_credits_and_shorts_tag() -> None:
    text = build_description(make_script(), ["Photo by Ana on Pexels", "Photo by Bo on Pexels"])
    assert text == (
        "How retrieval keeps AI answers grounded.\n\n"
        "Photos: Photo by Ana on Pexels; Photo by Bo on Pexels\n\n#Shorts"
    )


def test_description_is_trimmed_to_youtube_limit_keeping_the_tag() -> None:
    credits = ["Photo by Ünïcode Name on Pexels"] * 400
    text = build_description(make_script(), credits)
    assert len(text.encode("utf-8")) <= MAX_DESCRIPTION_BYTES
    assert text.endswith("\n\n#Shorts")


def test_metadata_is_private_labelled_and_not_for_kids() -> None:
    body = build_metadata(make_script(), "private", [])
    assert body["snippet"]["title"] == "RAG in 45 seconds"
    assert body["snippet"]["tags"] == ["AI", "RAG", "LLM"]
    assert body["snippet"]["categoryId"] == "28"
    assert body["status"] == {
        "privacyStatus": "private",
        "selfDeclaredMadeForKids": False,
        "containsSyntheticMedia": True,
    }


def test_upload_sends_chunks_and_returns_the_video(settings: Settings, video: Path) -> None:
    service = FakeService((FakeStatus(0.5), None), (None, {"id": "abc123"}))
    result = YouTubeUploader(settings, service).upload(video, make_script(), [])
    assert result.video_id == "abc123"
    assert result.url == "https://youtube.com/shorts/abc123"
    assert result.privacy == "private"
    call = service.calls[0]
    assert call["part"] == "snippet,status"
    assert call["body"]["status"]["privacyStatus"] == "private"
    assert call["media_body"].resumable()
    assert service.request.retries == [3, 3]


def test_missing_video_is_reported(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        YouTubeUploader(settings, FakeService()).upload(tmp_path / "nope.mp4", make_script(), [])


def test_quota_error_is_explained(settings: Settings, video: Path) -> None:
    service = FakeService(http_error(403, "quotaExceeded"))
    with pytest.raises(UploadError, match="daily quota"):
        YouTubeUploader(settings, service).upload(video, make_script(), [])


def test_rejected_login_becomes_config_error(settings: Settings, video: Path) -> None:
    service = FakeService(http_error(401, "authError"))
    with pytest.raises(ConfigError, match="token.json"):
        YouTubeUploader(settings, service).upload(video, make_script(), [])


def test_other_forbidden_errors_include_setup_help(settings: Settings, video: Path) -> None:
    service = FakeService(http_error(403, "forbidden"))
    with pytest.raises(UploadError, match="Desktop app"):
        YouTubeUploader(settings, service).upload(video, make_script(), [])


def test_missing_response_id_is_an_error(settings: Settings, video: Path) -> None:
    with pytest.raises(UploadError, match="no video ID"):
        YouTubeUploader(settings, FakeService((None, {}))).upload(video, make_script(), [])


def test_cached_token_is_used_without_signing_in(settings: Settings) -> None:
    token = {
        "token": "access",
        "refresh_token": "refresh",
        "client_id": "id",
        "client_secret": "secret",
        "scopes": SCOPES,
        "expiry": "2099-01-01T00:00:00Z",
    }
    settings.youtube_token.write_text(json.dumps(token), encoding="utf-8")
    creds = load_credentials(settings)
    assert creds.token == "access"


def test_missing_client_secret_explains_setup(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="client_secret.json not found"):
        load_credentials(settings)


def test_expired_login_falls_back_to_sign_in(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials

    token = {
        "token": "old",
        "refresh_token": "refresh",
        "client_id": "id",
        "client_secret": "secret",
        "scopes": SCOPES,
        "expiry": "2020-01-01T00:00:00Z",
    }
    settings.youtube_token.write_text(json.dumps(token), encoding="utf-8")

    def refuse(self: Credentials, request: Any) -> None:
        raise RefreshError("invalid_grant")

    monkeypatch.setattr(Credentials, "refresh", refuse)
    with pytest.raises(ConfigError, match="client_secret.json not found"):
        load_credentials(settings)
