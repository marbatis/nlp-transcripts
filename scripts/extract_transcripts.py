"""Utility for downloading YouTube video transcripts.

This script reads newline separated YouTube video URLs or IDs from an input
file and stores each transcript inside the provided output directory.  Each
transcript is written as a plain text file named after the video id.

Example usage:
    python scripts/extract_transcripts.py video_urls.txt transcripts/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError
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


def fetch_transcript(video_id: str) -> str:
    for language in LANGUAGE_CANDIDATES:
        transcript = _download_json_transcript(video_id, language)
        if transcript:
            return transcript
    raise TranscriptNotAvailableError(
        f"Transcript not available in languages {LANGUAGE_CANDIDATES}"
    )


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
