import re
import threading
import nltk


# ─── NLTK Bootstrap ───────────────────────────────────────────────────────────

def _ensure_nltk_data() -> None:
    """
    Download NLTK punkt tokeniser data only if not already present.
    Download runs in a background thread with a 15-second timeout so it
    never freezes the app on startup.
    """
    packages = ['punkt', 'punkt_tab']
    for pkg in packages:
        # ── Check if already on disk — skip download entirely ────────────────
        try:
            nltk.data.find(f'tokenizers/{pkg}')
            continue   # already present, nothing to do
        except (LookupError, OSError):
            pass

        # ── Not found — try a timed download ─────────────────────────────────
        print(f"⬇  Downloading NLTK '{pkg}' data…")
        success = threading.Event()

        def _download(p=pkg):
            try:
                nltk.download(p, quiet=True)
                success.set()
            except Exception:
                pass  # success stays unset → timeout path handles it

        t = threading.Thread(target=_download, daemon=True)
        t.start()
        t.join(timeout=15)   # wait max 15 s; carry on either way

        if success.is_set():
            print(f"  NLTK '{pkg}' ready")
        else:
            print(
                f"  ⚠️  NLTK '{pkg}' download timed-out or failed.\n"
                f"  Run once manually:  python -m nltk.downloader {pkg}\n"
                f"  Falling back to whitespace splitter."
            )


_ensure_nltk_data()

# Importing after bootstrap so sent_tokenize uses the freshly-downloaded data.
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


# ─── Chapter detection ────────────────────────────────────────────────────────

_CHAPTER_PATTERNS: list[re.Pattern] = [
    re.compile(r'^chapter\s+[\divxlcdm]+', re.IGNORECASE),
    re.compile(r'^\d+\.\s+\w'),
    re.compile(r'^part\s+[\divxlcdm]+',    re.IGNORECASE),
    re.compile(r'^book\s+[\divxlcdm]+',    re.IGNORECASE),
    re.compile(r'^section\s+[\divxlcdm]+', re.IGNORECASE),
    re.compile(r'^episode\s+[\divxlcdm]+', re.IGNORECASE),
    re.compile(r'^act\s+[\divxlcdm]+',     re.IGNORECASE),
    re.compile(r'^prologue$',              re.IGNORECASE),
    re.compile(r'^epilogue$',              re.IGNORECASE),
]


def detect_chapters(text: str) -> list[dict]:
    chapters: list[dict] = []
    lines    = text.split('\n')
    char_pos = 0

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            for pattern in _CHAPTER_PATTERNS:
                if pattern.match(stripped):
                    chapters.append({
                        'name':     stripped,
                        'position': char_pos,
                        'line':     line_idx,
                    })
                    break
        char_pos += len(line) + 1

    print(f"  Detected {len(chapters)} chapter(s)")
    return chapters


# ─── Sentence-level chunking ──────────────────────────────────────────────────

def _fallback_split(text: str, max_chars: int) -> list[str]:
    """Hard-split at max_chars boundaries, breaking at whitespace where possible."""
    chunks: list[str] = []
    while len(text) > max_chars:
        split_at = text.rfind(' ', 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def _sent_tokenize(text: str) -> list[str]:
    """Tokenise sentences, falling back to a simple regex split."""
    if _NLTK_AVAILABLE:
        try:
            return _nltk_sent_tokenize(text)
        except Exception:
            pass
    # Regex fallback: split on ". / ! / ?" followed by whitespace
    return re.split(r'(?<=[.!?])\s+', text)


def _sentence_chunks(text: str, max_chars: int) -> list[str]:
    sentences = _sent_tokenize(text)
    chunks: list[str] = []
    current = ''

    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ''
            chunks.extend(_fallback_split(sent, max_chars))
            continue

        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + ' ' + sent).strip()
        else:
            if current:
                chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


# ─── Chapter-aware chunking ───────────────────────────────────────────────────

def _chunk_by_chapters(text: str, chapters: list[dict], max_chars: int) -> list[dict]:
    chunks: list[dict] = []

    for i, chapter in enumerate(chapters):
        start = chapter['position']
        end   = chapters[i + 1]['position'] if i + 1 < len(chapters) else len(text)
        body  = text[start:end].strip()
        if not body:
            continue

        sub_chunks = _sentence_chunks(body, max_chars)
        for j, sub in enumerate(sub_chunks):
            chunks.append({
                'text':                    sub,
                'chapter':                 chapter['name'],
                'chapter_index':           i,
                'chunk_index':             j,
                'total_chunks_in_chapter': len(sub_chunks),
            })

    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────

def smart_chunk(
    text:          str,
    max_chars:     int  = 2000,
    with_chapters: bool = False,
    chapters:      list[dict] | None = None,
) -> list:
    if not text or not text.strip():
        return []

    if with_chapters and chapters:
        result = _chunk_by_chapters(text, chapters, max_chars)
        print(f"  Chapter-aware chunks: {len(result)}")
        return result

    result = _sentence_chunks(text, max_chars)
    print(f"  Plain chunks: {len(result)}")
    return result