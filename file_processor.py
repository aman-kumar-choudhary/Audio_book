# file_processor

import os
import subprocess
import tempfile

from config import CALIBRE_CMD

# ─── Calibre-supported extensions ────────────────────────────────────────────
_CALIBRE_SUPPORTED = {
    '.epub', '.mobi', '.azw', '.azw3', '.fb2',
    '.lit',  '.lrf',  '.pdb', '.pdf',  '.rtf',
    '.html', '.htm',  '.docx', '.odt',
}

# Encoding preference order for plain text files.
_TXT_ENCODINGS = ('utf-8-sig', 'utf-8', 'utf-16', 'latin-1', 'cp1252')


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """
    Extract plain text from *file_path*.

    Dispatches to the appropriate extraction method based on file extension.

    Raises
    ------
    ValueError   — unsupported file extension
    RuntimeError — Calibre not found or conversion failed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.txt':
        return _extract_txt(file_path)

    if ext in _CALIBRE_SUPPORTED:
        return _extract_via_calibre(file_path)

    raise ValueError(
        f"Unsupported file type: '{ext}'.\n"
        f"Supported types: .txt, {', '.join(sorted(_CALIBRE_SUPPORTED))}"
    )


# ─── Plain text ───────────────────────────────────────────────────────────────

def _extract_txt(path: str) -> str:
    for enc in _TXT_ENCODINGS:
        try:
            with open(path, 'r', encoding=enc, errors='strict') as fh:
                text = fh.read()
            print(f"  TXT ({enc}): {len(text):,} chars")
            return _clean_spacing(text)
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort — replace undecodable bytes.
    with open(path, 'rb') as fh:
        text = fh.read().decode('utf-8', errors='replace')
    print(f"  TXT (utf-8/replace): {len(text):,} chars")
    return _clean_spacing(text)


# ─── Calibre ──────────────────────────────────────────────────────────────────

def _extract_via_calibre(source_path: str) -> str:
    """
    Convert *source_path* to plain text using Calibre's ebook-convert CLI.

    A temporary output file is used and deleted after reading.
    """
    _verify_calibre()

    ext = os.path.splitext(source_path)[1].lower()
    print(f" Calibre: {ext} → TXT  ({os.path.basename(source_path)})")

    fd, output_txt = tempfile.mkstemp(suffix='.txt', prefix='calibre_out_')
    os.close(fd)

    try:
        cmd = [
            CALIBRE_CMD,
            source_path,
            output_txt,
            '--enable-heuristics',
            '--no-chapters-in-toc',
            '--remove-paragraph-spacing',
            '--pretty-print'
            ]
        print(f"   $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            stderr_tail = (result.stderr or '').strip()[-3000:]
            raise RuntimeError(
                f"Calibre conversion failed (exit {result.returncode}):\n"
                f"{stderr_tail}"
            )

        if not os.path.exists(output_txt) or os.path.getsize(output_txt) == 0:
            raise RuntimeError(
                "Calibre completed but produced an empty output file.\n"
                "The input file may be encrypted or corrupt."
            )

        with open(output_txt, 'r', encoding='utf-8', errors='ignore') as fh:
            text = fh.read()

        print(f"  Calibre extracted {len(text):,} chars")
        return _clean_spacing(text)

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Calibre timed out after 5 minutes on "
            f"'{os.path.basename(source_path)}'."
        )
    finally:
        try:
            if os.path.exists(output_txt):
                os.remove(output_txt)
        except OSError:
            pass


def _verify_calibre() -> None:
    """
    Confirm that Calibre's ebook-convert command is available on PATH.
    Raises RuntimeError with installation instructions if not found.
    """
    try:
        result = subprocess.run(
            [CALIBRE_CMD, '--version'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise FileNotFoundError
        version = (result.stdout or result.stderr or '').strip().splitlines()[0]
        print(f"🔍  {version}")
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        raise RuntimeError(
            "Calibre (ebook-convert) not found on PATH.\n"
            "Install it with:\n"
            "  Ubuntu/Debian : sudo apt-get install calibre\n"
            "  macOS         : brew install --cask calibre\n"
            "  Windows       : https://calibre-ebook.com/download"
        )


# ─── Text normalisation ───────────────────────────────────────────────────────

def _clean_spacing(text: str) -> str:
    """
    Normalise whitespace:
    - Strip trailing spaces from each line.
    - Collapse runs of more than two blank lines into two.
    - Remove leading and trailing blank lines from the document.
    """
    lines   = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    blanks  = 0

    for line in lines:
        if line == '':
            blanks += 1
            if blanks <= 2:
                cleaned.append('')
        else:
            blanks = 0
            cleaned.append(line)

    # Trim leading / trailing blank lines.
    while cleaned and cleaned[0]  == '':
        cleaned.pop(0)
    while cleaned and cleaned[-1] == '':
        cleaned.pop()

    return '\n'.join(cleaned)