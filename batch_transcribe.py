"""
Batch transcription — process thousands of short audio files with resume.

Usage:
    python batch_transcribe.py /input/audio/ /output/transcripts/ --lang ben_Beng --batch-size 8

Resume: a manifest.json in the output dir tracks completed files.
Re-running skips anything marked "done" and picks up where it left off.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import config
from model import ASRModel

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"WARNING: corrupt manifest at {path}, starting fresh")
        return {}


def save_manifest(path: Path, manifest: dict) -> None:
    """Atomic write — temp file then rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


def find_audio_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files with resume support"
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory for output .txt files")
    parser.add_argument(
        "--lang", default=None,
        help="Language code (e.g. ben_Beng, eng_Latn). Default: from config"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Files per GPU batch (default: from config, currently {config.model.batch_size})"
    )
    parser.add_argument(
        "--ext", nargs="*", default=None,
        help="File extensions to include (default: wav mp3 flac m4a ogg opus)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    lang = args.lang or config.model.default_lang
    batch_size = args.batch_size or config.model.batch_size
    extensions = (
        {f".{e.lstrip('.').lower()}" for e in args.ext}
        if args.ext
        else AUDIO_EXTENSIONS
    )

    all_files = find_audio_files(input_dir, extensions)
    if not all_files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    manifest_path = output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    pending = [
        f for f in all_files
        if manifest.get(f.relative_to(input_dir).as_posix()) != "done"
    ]
    total = len(all_files)
    done_count = total - len(pending)

    print(f"Found {total} files ({done_count} done, {len(pending)} pending)")
    print(f"Batch size: {batch_size} | Lang: {lang or 'auto'}")
    print()

    model = ASRModel.get_instance()
    model.load()
    print()

    processed = 0
    audio_secs = 0.0
    t0 = time.perf_counter()

    for i in range(0, len(pending), batch_size):
        batch = pending[i: i + batch_size]

        try:
            results = model.transcribe_batch(batch, lang=lang)

            for path, result in zip(batch, results):
                rel = path.relative_to(input_dir)
                out = output_dir / rel.with_suffix(".txt")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(result.text, encoding="utf-8")
                manifest[rel.as_posix()] = "done"
                audio_secs += result.duration

            save_manifest(manifest_path, manifest)
            processed += len(batch)

        except Exception as e:
            print(f"  BATCH ERROR: {e}")
            print("  Retrying individually...")
            for path in batch:
                rel = path.relative_to(input_dir)
                try:
                    results = model.transcribe_batch([path], lang=lang)
                    text = results[0].text if results else ""
                    out = output_dir / rel.with_suffix(".txt")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(text, encoding="utf-8")
                    manifest[rel.as_posix()] = "done"
                    audio_secs += results[0].duration
                except Exception as e2:
                    print(f"    FAILED: {path.name} — {e2}")
                    manifest[rel.as_posix()] = f"error: {e2}"
                save_manifest(manifest_path, manifest)
            processed += len(batch)

        elapsed = time.perf_counter() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = len(pending) - processed
        eta = remaining / rate if rate > 0 else 0
        print(
            f"  [{done_count + processed}/{total}] "
            f"{rate:.1f} files/s | "
            f"ETA {eta / 60:.0f}min | "
            f"last: {', '.join(f.name for f in batch)}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\nDone! {processed} files in {elapsed:.1f}s")
    if audio_secs > 0 and elapsed > 0:
        print(
            f"Audio: {audio_secs / 3600:.1f}hr | "
            f"Wall: {elapsed / 60:.1f}min | "
            f"Throughput: {audio_secs / elapsed:.1f}x realtime"
        )


if __name__ == "__main__":
    main()
