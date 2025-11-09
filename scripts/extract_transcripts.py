"""Utility for downloading YouTube video transcripts.

This script reads newline separated YouTube video URLs or IDs from an input
file and stores each transcript inside the provided output directory. Each
transcript is written as a plain text file named after the video id.

The implementation combines several public techniques collected while
researching how to obtain transcripts from blocked or otherwise restricted
videos:

* The historic ``timedtext`` endpoint for first-party caption tracks
* The ``youtube-transcript-api`` project for auto-generated subtitles
* ``yt-dlp`` as a final fallback able to download a wide range of caption
  formats

Only freely available libraries are required and both geo-blocks and missing
auto captions are handled more gracefully than before.

Example usage::

    python scripts/extract_transcripts.py video_urls.txt transcripts/
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)

# Default language candidates to try. The official timedtext endpoint accepts
# BCP-47 language codes, so we try a few English variants that commonly exist.
LANGUAGE_CANDIDATES = ("en", "en-US", "en-GB")


class TranscriptDownloadError(RuntimeError):
    """Raised when a transcript cannot be downloaded."""


class TranscriptNotAvailableError(TranscriptDownloadError):
    """Raised when the transcript is not available for any language."""


def extract_video_id(raw: str) -> str:
    """Return the YouTube video id from ``raw``.

    ``raw`` can already be an id or a YouTube URL containing the id.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("empty video id/URL provided")

    # If the caller already provided the id just return it.
    if all(ch.isalnum() or ch in ("-", "_") for ch in raw) and len(raw) in range(10, 20):
        return raw

    parsed = urlparse(raw)
    if parsed.hostname in {"youtu.be"}:
        video_id = parsed.path.lstrip("/")
        if not video_id:
            raise ValueError(f"unable to determine video id from short URL: {raw}")
        return video_id

    if parsed.hostname and "youtube" in parsed.hostname:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

    raise ValueError(f"Unrecognised YouTube video URL or id: {raw}")


def read_video_ids(source: Path) -> Iterable[str]:
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        yield extract_video_id(line)


def _download_json_transcript(video_id: str, language: str) -> Optional[str]:
    """Download and decode a transcript in json3 format.

    Returns the transcript text if the download succeeded and contained
    meaningful content. Otherwise ``None`` is returned.
    """
    url = (
        "https://www.youtube.com/api/timedtext"
        f"?lang={language}&v={video_id}&fmt=json3"
    )
    try:
        with urlopen(url, timeout=10) as response:  # nosec B310 - trusted URL
            payload = response.read().decode("utf-8")
    except HTTPError as err:
        if err.code in {404, 403}:
            return None
        raise TranscriptDownloadError(
            f"HTTP error {err.code} while downloading transcript for {video_id}"
        ) from err
    except OSError as err:  # handles URLError and socket errors
        raise TranscriptDownloadError(
            f"Network error while downloading transcript for {video_id}: {err}"
        ) from err

    if not payload.strip():
        return None

    data = json.loads(payload)
    events = data.get("events", [])
    lines = []
    for event in events:
        for segment in event.get("segs", []) or []:
            text = (segment.get("utf8") or "").replace("\n", " ").strip()
            if text:
                lines.append(text)
    transcript = "\n".join(lines).strip()
    return transcript or None


def _fetch_with_timedtext(video_id: str) -> str:
    """Try fetching the transcript via the timedtext endpoint."""

    for language in LANGUAGE_CANDIDATES:
        transcript = _download_json_transcript(video_id, language)
        if transcript:
            LOGGER.debug(
                "Retrieved transcript for %s using timedtext language %s",
                video_id,
                language,
            )
            return transcript
    raise TranscriptNotAvailableError(
        f"Transcript not available in languages {LANGUAGE_CANDIDATES}"
    )


def _fetch_with_youtube_transcript_api(video_id: str) -> str:
    """Fetch transcript using the youtube-transcript-api package."""

    try:
        from youtube_transcript_api import (  # type: ignore
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )
    except ImportError as err:  # pragma: no cover - optional dependency
        raise TranscriptDownloadError(
            "youtube-transcript-api is not installed"
        ) from err

    try:
        segments = YouTubeTranscriptApi.get_transcript(
            video_id, languages=list(LANGUAGE_CANDIDATES)
        )
    except (NoTranscriptFound, TranscriptsDisabled) as err:
        raise TranscriptNotAvailableError(str(err)) from err
    except VideoUnavailable as err:
        raise TranscriptDownloadError(f"Video unavailable: {err}") from err
    except Exception as err:  # pragma: no cover - defensive
        raise TranscriptDownloadError(
            f"youtube-transcript-api failed for {video_id}: {err}"
        ) from err

    lines = []
    for segment in segments:
        text = (segment.get("text") or "").replace("\n", " ").strip()
        if text:
            lines.append(text)

    transcript = "\n".join(lines).strip()
    if not transcript:
        raise TranscriptNotAvailableError(
            "youtube-transcript-api returned an empty transcript"
        )

    LOGGER.debug("Retrieved transcript for %s using youtube-transcript-api", video_id)
    return transcript


def _parse_caption_payload(payload: str, extension: str) -> Optional[str]:
    """Convert a caption payload into plain text."""

    if not payload.strip():
        return None

    if extension == "json3":
        data = json.loads(payload)
        events = data.get("events", [])
        lines = []
        for event in events:
            for segment in event.get("segs", []) or []:
                text = (segment.get("utf8") or "").replace("\n", " ").strip()
                if text:
                    lines.append(text)
        return "\n".join(lines).strip() or None

    lines = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        if "-->" in line:
            continue
        if extension == "srt" and line.isdigit():
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        lines.append(line)
    return "\n".join(lines).strip() or None


@dataclass
class CaptionCandidate:
    url: str
    extension: str


def _iter_caption_candidates(info: dict) -> Iterable[CaptionCandidate]:
    """Yield caption download candidates from a yt-dlp info dict."""

    for language in LANGUAGE_CANDIDATES:
        for store in ("subtitles", "automatic_captions"):
            tracks = info.get(store, {}).get(language, [])
            for track in tracks:
                url = track.get("url")
                ext = track.get("ext")
                if not url or not ext:
                    continue
                yield CaptionCandidate(url=url, extension=ext)


def _fetch_with_yt_dlp(video_id: str) -> str:
    """Fetch transcript information via yt-dlp."""

    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except ImportError as err:  # pragma: no cover - optional dependency
        raise TranscriptDownloadError("yt-dlp is not installed") from err

    ydl_options = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "logger": LOGGER,
    }

    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(video_id, download=False)

    for candidate in _iter_caption_candidates(info):
        try:
            with urlopen(candidate.url, timeout=10) as response:  # nosec B310
                payload = response.read().decode("utf-8", errors="ignore")
        except HTTPError as err:
            if err.code in {403, 404}:
                continue
            raise TranscriptDownloadError(
                f"HTTP error {err.code} downloading yt-dlp captions for {video_id}"
            ) from err
        except URLError as err:
            raise TranscriptDownloadError(
                f"Network error downloading yt-dlp captions for {video_id}: {err}"
            ) from err

        transcript = _parse_caption_payload(payload, candidate.extension)
        if transcript:
            LOGGER.debug(
                "Retrieved transcript for %s using yt-dlp track (%s)",
                video_id,
                candidate.extension,
            )
            return transcript

    raise TranscriptNotAvailableError(
        "yt-dlp did not return usable caption tracks"
    )


def fetch_transcript(video_id: str) -> str:
    """Attempt to fetch a transcript using multiple public techniques."""

    strategies = (
        _fetch_with_timedtext,
        _fetch_with_youtube_transcript_api,
        _fetch_with_yt_dlp,
    )
    last_error: Optional[Exception] = None

    for strategy in strategies:
        try:
            return strategy(video_id)
        except TranscriptNotAvailableError as err:
            LOGGER.debug("Strategy %s did not have a transcript: %s", strategy.__name__, err)
            last_error = err
        except TranscriptDownloadError as err:
            LOGGER.debug("Strategy %s failed to download transcript: %s", strategy.__name__, err)
            last_error = err

    if isinstance(last_error, TranscriptNotAvailableError):
        raise last_error
    if last_error:
        raise TranscriptDownloadError(str(last_error))

    raise TranscriptDownloadError("Unable to fetch transcript for unknown reasons")


def save_transcript(destination: Path, video_id: str, transcript: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{video_id}.txt"
    output_path.write_text(transcript, encoding="utf-8")
    LOGGER.info("Saved transcript for %s to %s", video_id, output_path)


def save_error(destination: Path, video_id: str, error: Exception) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    (destination / f"{video_id}.error.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.error("Failed to fetch transcript for %s: %s", video_id, error)


def run(input_file: Path, output_dir: Path) -> None:
    for video_id in read_video_ids(input_file):
        try:
            transcript = fetch_transcript(video_id)
        except TranscriptDownloadError as err:
            save_error(output_dir, video_id, err)
        else:
            save_transcript(output_dir, video_id, transcript)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="File containing YouTube video URLs/ids")
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where the transcripts will be stored",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Configure the verbosity of log output",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    run(args.input, args.output)
