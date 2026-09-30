"""YouTube upload through the Data API v3 with the OAuth desktop flow.

The first upload opens a browser to sign in; the resulting token is cached in
``YOUTUBE_TOKEN`` (gitignored) and refreshed automatically. Only the upload scope is
requested. Videos go up as a resumable upload in chunks, and transient server errors are
retried by the client library with exponential backoff.

Uploads default to private: YouTube locks videos uploaded by unverified API projects to
private anyway, and publishing is a deliberate step in YouTube Studio.

Run `python -m shortforge.upload` to sign in and cache the token without uploading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from shortforge.config import ConfigError, Settings, get_settings
from shortforge.fsutil import write_text_atomic
from shortforge.log import console
from shortforge.models import Script

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CATEGORY_SCIENCE_AND_TECHNOLOGY = "28"
CHUNK_BYTES = 8 * 1024 * 1024
MAX_DESCRIPTION_BYTES = 5000
SHORTS_TAG = "#Shorts"

SETUP_HELP = (
    "In Google Cloud Console: enable YouTube Data API v3, create an OAuth client of type "
    "'Desktop app', download it as client_secret.json into the project folder, and add your "
    "Google account as a test user on the OAuth consent screen."
)


class UploadError(RuntimeError):
    """Raised when YouTube rejects or fails an upload, with the reason it gave."""


@dataclass(frozen=True)
class UploadResult:
    """A video that is now on YouTube."""

    video_id: str
    privacy: str

    @property
    def url(self) -> str:
        """Link that opens the video in the Shorts player."""
        return f"https://youtube.com/shorts/{self.video_id}"


class YouTubeService(Protocol):
    """The slice of the googleapiclient YouTube resource this module uses."""

    def videos(self) -> Any: ...


def build_description(script: Script, credits: list[str]) -> str:
    """Return the video description: summary, photo credits, then the Shorts hashtag.

    Trimmed to YouTube's 5,000-byte limit, keeping the hashtag.
    """
    parts = [script.description]
    if credits:
        parts.append("Photos: " + "; ".join(credits))
    footer = f"\n\n{SHORTS_TAG}"
    body = "\n\n".join(parts)
    budget = MAX_DESCRIPTION_BYTES - len(footer.encode("utf-8"))
    encoded = body.encode("utf-8")
    if len(encoded) > budget:
        body = encoded[:budget].decode("utf-8", errors="ignore").rstrip()
    return body + footer


def build_metadata(script: Script, privacy: str, credits: list[str]) -> dict[str, Any]:
    """Return the `videos.insert` request body for `script`.

    The synthetic-media flag is set because scene images are realistic AI-generated
    photos, which YouTube asks creators to disclose.
    """
    return {
        "snippet": {
            "title": script.title,
            "description": build_description(script, credits),
            "tags": script.tags,
            "categoryId": CATEGORY_SCIENCE_AND_TECHNOLOGY,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,
        },
    }


def load_credentials(settings: Settings) -> Any:
    """Return valid OAuth credentials, refreshing or signing in through the browser as needed.

    Raises:
        ConfigError: If client_secret.json is missing or invalid.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = settings.youtube_token
    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except (ValueError, json.JSONDecodeError):
            console.print(f"  [yellow]![/] {token_path} is unreadable, signing in again")
    if creds is not None and creds.valid:
        return creds
    if creds is not None and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            write_text_atomic(token_path, creds.to_json())
            return creds
        except RefreshError:
            console.print(
                "  [yellow]![/] saved YouTube login expired or was revoked (logins for apps in "
                "Testing mode last 7 days), signing in again"
            )

    secret_path = settings.youtube_client_secret
    if not secret_path.exists():
        raise ConfigError(f"{secret_path} not found. {SETUP_HELP}")
    console.print("  opening a browser to sign in to YouTube...")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
    except ValueError as exc:
        raise ConfigError(
            f"{secret_path} is not a valid OAuth client file ({exc}). "
            "It must be the JSON of a 'Desktop app' OAuth client."
        ) from exc
    creds = flow.run_local_server(port=0, prompt="consent")
    write_text_atomic(token_path, creds.to_json())
    console.print(f"  signed in, login saved to {token_path}")
    return creds


def _reason(exc: Exception) -> str:
    """Pull YouTube's machine-readable reason (such as ``quotaExceeded``) out of an HttpError."""
    details = getattr(exc, "error_details", None) or []
    for detail in details:
        if isinstance(detail, dict) and detail.get("reason"):
            return str(detail["reason"])
    return ""


class YouTubeUploader:
    """Upload finished Shorts to the signed-in channel."""

    def __init__(self, settings: Settings, service: YouTubeService | None = None) -> None:
        self._settings = settings
        self._service = service

    @property
    def service(self) -> YouTubeService:
        """The YouTube API client, signing in on first use."""
        if self._service is None:
            from googleapiclient.discovery import build

            creds = load_credentials(self._settings)
            self._service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def upload(self, video: Path, script: Script, credits: list[str]) -> UploadResult:
        """Upload `video` with metadata from `script` and return its YouTube ID.

        Raises:
            FileNotFoundError: If `video` does not exist.
            ConfigError: If the login is rejected.
            UploadError: If YouTube refuses the upload, for example over quota.
        """
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        if not video.is_file():
            raise FileNotFoundError(f"No video to upload at {video}")
        privacy = self._settings.youtube_privacy
        body = build_metadata(script, privacy, credits)
        media = MediaFileUpload(
            str(video), mimetype="video/mp4", chunksize=CHUNK_BYTES, resumable=True
        )
        request = self.service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        try:
            while response is None:
                status, response = request.next_chunk(num_retries=self._settings.max_retries)
                if status is not None:
                    console.print(f"  uploaded {status.progress():.0%}")
        except HttpError as exc:
            raise self._explain(exc) from exc
        video_id = response.get("id") if isinstance(response, dict) else None
        if not video_id:
            raise UploadError(f"YouTube finished the upload but returned no video ID: {response}")
        return UploadResult(video_id, privacy)

    def _explain(self, exc: Exception) -> Exception:
        """Map an HttpError to an error that says what to do about it."""
        status = getattr(getattr(exc, "resp", None), "status", None)
        reason = _reason(exc)
        if status == 401:
            return ConfigError(
                f"YouTube rejected the login (401). Delete {self._settings.youtube_token} "
                "and run again to sign in."
            )
        if reason in {"quotaExceeded", "dailyLimitExceeded"}:
            return UploadError(
                "YouTube API daily quota used up (each upload costs about 1,600 of 10,000 "
                "units). It resets at midnight Pacific time; resume the run with --upload then."
            )
        if reason == "uploadLimitExceeded":
            return UploadError("This channel hit YouTube's upload limit for now. Try later.")
        if status == 403:
            return UploadError(f"YouTube refused the upload (403 {reason}). {SETUP_HELP}")
        return UploadError(f"YouTube upload failed (HTTP {status} {reason}): {exc}")


def _self_check() -> None:
    """Sign in (or refresh the cached login) without uploading anything."""
    settings = get_settings()
    try:
        load_credentials(settings)
    except ConfigError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc
    console.print(
        f"[bold green]OK[/] YouTube login ready ({settings.youtube_token}). "
        f"Uploads will be {settings.youtube_privacy}."
    )


if __name__ == "__main__":
    _self_check()
