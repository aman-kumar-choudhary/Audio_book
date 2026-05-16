

import os
import subprocess
import tempfile


def stitch_audio(wav_files: list[str], output_mp3: str) -> str:
    """
    Merge *wav_files* into *output_mp3* (MP3, 192 kb/s).

    Parameters
    ----------
    wav_files  : Ordered list of paths to WAV / MP3 chunk files.
    output_mp3 : Destination MP3 path.

    Returns
    -------
    Absolute path of the created MP3.

    Raises
    ------
    RuntimeError  — if no valid input files exist or FFmpeg fails.
    """
    if not wav_files:
        raise RuntimeError("stitch_audio: no input files provided")

    # ── Validate inputs ───────────────────────────────────────────────────────
    valid: list[str] = []
    for path in wav_files:
        if not os.path.exists(path):
            print(f"   Missing chunk file: {path}")
            continue
        if os.path.getsize(path) == 0:
            print(f"   Empty chunk file (skipped): {path}")
            continue
        valid.append(path)

    if not valid:
        raise RuntimeError("stitch_audio: no valid input files found")

    print(f"  Stitching {len(valid)} chunk(s) → {os.path.basename(output_mp3)}")

    # ── Write FFmpeg concat manifest ──────────────────────────────────────────
    fd, concat_path = tempfile.mkstemp(suffix='.txt', text=True)
    try:
        with os.fdopen(fd, 'w') as fh:
            for path in valid:
                # Use absolute paths and escape single quotes.
                safe = os.path.abspath(path).replace("'", "'\\''")
                fh.write(f"file '{safe}'\n")

        # ── FFmpeg command ────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_mp3)), exist_ok=True)

        cmd = [
            'ffmpeg', '-y',
            '-f',    'concat',
            '-safe', '0',
            '-i',    concat_path,
            '-c:a',  'libmp3lame',
            '-b:a',  '192k',
            '-ar',   '44100',      # standard sample rate
            output_mp3,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (code {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        if not os.path.exists(output_mp3) or os.path.getsize(output_mp3) == 0:
            raise RuntimeError("FFmpeg ran but output file is missing or empty")

        size_mb = os.path.getsize(output_mp3) / (1024 * 1024)
        print(f" Audio stitched: {size_mb:.1f} MB → {output_mp3}")
        return output_mp3

    finally:
        try:
            os.remove(concat_path)
        except OSError:
            pass