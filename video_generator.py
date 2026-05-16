import os
import subprocess
import tempfile

from PIL import Image

from config import USE_GPU



def _prepare_image(image_path: str) -> tuple[str, bool]:
    """
    Convert *image_path* to a JPEG with even pixel dimensions and no alpha
    channel.  Returns (processed_path, created_temp_file).
    """
    try:
        with Image.open(image_path) as img:
            # Flatten transparency to white.
            if img.mode in ('RGBA', 'LA', 'P'):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                else:
                    img = img.convert('RGB')
            else:
                img = img.convert('RGB')

            w, h = img.size
            new_w = w if w % 2 == 0 else w + 1
            new_h = h if h % 2 == 0 else h + 1

            # Skip re-save if no changes needed.
            if new_w == w and new_h == h and image_path.lower().endswith(('.jpg', '.jpeg')):
                return image_path, False

            if new_w != w or new_h != h:
                print(f"   Image: {w}×{h} → {new_w}×{new_h}")
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            temp_path = os.path.join(
                tempfile.gettempdir(),
                f'ab_cover_{os.path.basename(image_path)}.jpg',
            )
            img.save(temp_path, 'JPEG', quality=95)
            return temp_path, True

    except Exception as exc:
        print(f"   Image preprocessing failed ({exc}) — using original")
        return image_path, False


# ─── Chapter metadata ─────────────────────────────────────────────────────────

def _write_chapter_file(chapters: list[dict]) -> str | None:
    """
    Write an FFmpeg-compatible FFMETADATA chapter file.
    Returns the temp file path, or None if *chapters* is empty.
    """
    if not chapters:
        return None

    path = os.path.join(tempfile.gettempdir(), 'ab_chapters.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(';FFMETADATA1\n\n')
        for ch in chapters:
            start = int(ch.get('start', 0) * 1000)
            end   = int(ch.get('end',   0) * 1000)
            title = ch.get('title', 'Chapter').replace('=', r'\=')
            fh.write('[CHAPTER]\n')
            fh.write('TIMEBASE=1/1000\n')
            fh.write(f'START={start}\n')
            fh.write(f'END={end}\n')
            fh.write(f'title={title}\n\n')
    return path


# ─── Video filter ─────────────────────────────────────────────────────────────

def _build_vf(title: str, enable_zoom: bool) -> str:
    """
    Build the FFmpeg -vf filter string.

    Scale + pad to 1920×1080, optionally add Ken-Burns zoom, then overlay
    a title text box at the bottom of the frame.
    """
    # Escape FFmpeg drawtext special chars: \ : ' %
    safe_title = (
        title
        .replace('\\', '\\\\')
        .replace("'",  "'\\\\''")
        .replace(':',  '\\:')
        .replace('%',  '\\%')
    )

    scale_pad = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    zoom = (
        ",zoompan=z='min(zoom+0.0003,1.08)':d=125:s=1920x1080"
        if enable_zoom else ''
    )

    text_overlay = (
        ",drawtext="
        "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"text='{safe_title}':"
        "fontcolor=white:"
        "fontsize=56:"
        "box=1:"
        "boxcolor=black@0.65:"
        "boxborderw=18:"
        "x=(w-text_w)/2:"
        "y=h-120"
    )

    return scale_pad + zoom + text_overlay


# ─── Main entry point ─────────────────────────────────────────────────────────

def create_video(
    audio_path:  str,
    image_path:  str,
    output_path: str,
    # title:       str        = 'Audiobook',
    chapters:    list | None = None,
    # enable_zoom: bool       = True,
    title="Audiobook",
    enable_zoom=False
) -> str:
    """
    Combine *audio_path* and *image_path* into an MP4 audiobook video.

    Parameters
    ----------
    audio_path   : Path to the MP3 source audio.
    image_path   : Path to the cover / thumbnail image.
    output_path  : Destination MP4 path.
    title        : Text displayed at the bottom of the video.
    chapters     : Optional list of {'title', 'start', 'end'} dicts (seconds).
    enable_zoom  : Add a subtle Ken-Burns zoom animation (default: True).

    Returns
    -------
    Absolute path of the created MP4.
    """
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image file not found: {image_path}")

    print(f"🎬  Creating video")
    print(f"   Audio : {os.path.basename(audio_path)}")
    print(f"   Image : {os.path.basename(image_path)}")
    print(f"   Title : {title}")

    processed_image, is_temp_image = _prepare_image(image_path)
    chapter_file = _write_chapter_file(chapters or [])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        vf = _build_vf(title, enable_zoom)

        # ── Build FFmpeg command ───────────────────────────────────────────────
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i',    processed_image,
            '-i',    audio_path,
        ]

        # Optionally inject chapter metadata as a third input.
        if chapter_file:
            cmd += ['-i', chapter_file]

        # Choose GPU (h264_nvenc) or CPU (libx264) encoder.
        if USE_GPU:
            video_codec = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-b:v', '5M']
        else:
            video_codec = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                           '-tune', 'stillimage']

        cmd += [
            '-vf',      vf,
            *video_codec,
            '-pix_fmt', 'yuv420p',
            '-c:a',     'aac',
            '-b:a',     '192k',
            '-shortest',
            '-movflags', '+faststart',
        ]

        if chapter_file:
            cmd += ['-map_metadata', '2']

        cmd.append(output_path)

        print(f"   $ {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # If GPU encoding failed, try again with CPU fallback.
            if USE_GPU and 'nvenc' in result.stderr.lower():
                print("   GPU encoder failed — retrying with CPU (libx264)")
                cpu_cmd = [
                    c if c != 'h264_nvenc' else 'libx264'
                    for c in cmd
                ]
                # Remove GPU-specific flags and add CRF for CPU.
                result = subprocess.run(cpu_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed (exit {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"
                )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("FFmpeg ran but output video is missing or empty")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Video created: {size_mb:.1f} MB → {output_path}")
        return output_path

    finally:
        if is_temp_image:
            try:
                os.remove(processed_image)
            except OSError:
                pass
        if chapter_file:
            try:
                os.remove(chapter_file)
            except OSError:
                pass


# ─── Compatibility alias ──────────────────────────────────────────────────────

def create_video_cpu_fallback(
    audio_path:  str,
    image_path:  str,
    output_path: str,
) -> str:
    """Compatibility shim — delegates to :func:`create_video`."""
    return create_video(audio_path, image_path, output_path)