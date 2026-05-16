import os
import time
import difflib
from typing import Callable

import requests

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_LLM_MODEL

LLM_MAX_CHARS   = int(os.getenv('LLM_MAX_CHARS', '8000'))
_RETRY_BACKOFF  = 2
_MAX_RETRIES    = 3
_NO_RETRY_CODES = {400, 401, 403, 404, 422}


# ─── Diff helper ──────────────────────────────────────────────────────────────

def compute_diff(original: str, cleaned: str) -> list[dict]:
    orig_lines    = original.splitlines(keepends=True)
    cleaned_lines = cleaned.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(None, orig_lines, cleaned_lines, autojunk=False)
    blocks: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            blocks.append({
                'type':           'equal',
                'lines':          orig_lines[i1:i2],
                'original_start': i1,
                'original_end':   i2,
            })
        elif tag in ('delete', 'replace'):
            blocks.append({
                'type':           'removed',
                'lines':          orig_lines[i1:i2],
                'original_start': i1,
                'original_end':   i2,
            })
            if tag == 'replace':
                blocks.append({
                    'type':           'added',
                    'lines':          cleaned_lines[j1:j2],
                    'original_start': i1,
                    'original_end':   i2,
                })
        elif tag == 'insert':
            blocks.append({
                'type':           'added',
                'lines':          cleaned_lines[j1:j2],
                'original_start': i1,
                'original_end':   i2,
            })

    return blocks


def diff_to_html(original: str, cleaned: str) -> str:
    blocks = compute_diff(original, cleaned)
    html_parts: list[str] = []

    for block in blocks:
        text = ''.join(block['lines'])
        escaped = (
            text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )
        if block['type'] == 'equal':
            html_parts.append(escaped)
        elif block['type'] == 'removed':
            html_parts.append(f'<mark class="diff-removed">{escaped}</mark>')
        elif block['type'] == 'added':
            html_parts.append(f'<mark class="diff-added">{escaped}</mark>')

    return ''.join(html_parts)


# ─── LLM Cleaner ──────────────────────────────────────────────────────────────

class LLMCleaner:
    """Cleans raw ebook text via an OpenRouter model."""

    def __init__(self, model_id: str = DEFAULT_LLM_MODEL) -> None:
        self.model_id = model_id
        print(f"  LLMCleaner: {self.model_id}")

    # ── Public API ────────────────────────────────────────────────────────────

    def clean_text(
        self,
        text: str,
        custom_prompt: str = '',
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        """Return cleaned text only."""
        if not text or not text.strip():
            return text
        return self._process(text, custom_prompt, progress_callback)

    def clean_with_diff(
        self,
        text: str,
        custom_prompt: str = '',
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """
        Clean text and compute a diff.

        Returns:
            {
                'cleaned':   str,
                'diff_html': str,
                'diff_data': list[dict],
            }
        """
        cleaned   = self.clean_text(text, custom_prompt, progress_callback)
        diff_html = diff_to_html(text, cleaned)
        diff_data = compute_diff(text, cleaned)
        return {
            'cleaned':   cleaned,
            'diff_html': diff_html,
            'diff_data': diff_data,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process(
        self,
        text: str,
        custom_prompt: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        chunks = [text[i:i + LLM_MAX_CHARS] for i in range(0, len(text), LLM_MAX_CHARS)]
        total  = len(chunks)
        print(f"  Cleaning {total} chunk(s) via OpenRouter")

        results: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{total}  ({len(chunk):,} chars)")
            prompt  = self._build_prompt(chunk, custom_prompt)
            cleaned = self._call_openrouter(prompt, chunk)
            results.append(cleaned)

            # ── Fire progress callback so callers can update the DB ──────────
            if progress_callback is not None:
                try:
                    progress_callback(i, total)
                except Exception as cb_exc:
                    print(f"  progress_callback error: {cb_exc}")

        joined = '\n'.join(results)
        print(f"  Cleaned text: {len(joined):,} chars")
        return joined

    # ── Prompt ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(text: str, custom_prompt: str) -> str:
        rules = (
            "You are a professional ebook editor. "
            "Clean the following raw ebook text extracted by Calibre.\n\n"
            "REMOVE:\n"
            "- Copyright / legal notices\n"
            "- Publisher info, ISBN numbers\n"
            "- Table of contents\n"
            "- Index / bibliography\n"
            "- Acknowledgements, dedications, forewords\n"
            "- Author bios and 'About the Author' sections\n"
            "- Advertisements or promotional content\n"
            "- Repetitive page headers / footers\n"
            "- Any front-matter or back-matter that is not the main story\n\n"
            "KEEP:\n"
            "- All chapter titles (preserve as clear headings)\n"
            "- The complete main story / body content\n"
            "- Dialogue and narration\n"
            "- Scene breaks (use '* * *' or a blank line)\n"
            "- Paragraph structure and line breaks\n\n"
            "RULES:\n"
            "- Return ONLY the cleaned text — no explanations, no preamble.\n"
            "- Do NOT summarise or paraphrase; keep original wording exactly.\n"
            "- Preserve the author's voice.\n"
        )
        if custom_prompt:
            rules += f"\nADDITIONAL INSTRUCTIONS:\n{custom_prompt}\n"

        return (
            f"{rules}\n\n"
            f"--- BEGIN BOOK TEXT ---\n{text}\n--- END BOOK TEXT ---\n\n"
            "CLEANED TEXT:"
        )

    # ── OpenRouter call ───────────────────────────────────────────────────────

    def _call_openrouter(self, prompt: str, raw_text: str) -> str:
        if not OPENROUTER_API_KEY:
            print("  OPENROUTER_API_KEY not set — rule-based fallback")
            return self._fallback_clean(raw_text)

        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type':  'application/json',
            'HTTP-Referer':  'https://audiobook-pipeline.local',
            'X-Title':       'Audiobook Pipeline',
        }
        payload = {
            'model':       self.model_id,
            'messages':    [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
            'max_tokens':  8000,
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                print(f"  OpenRouter attempt {attempt}/{_MAX_RETRIES}")
                resp = requests.post(
                    f'{OPENROUTER_BASE_URL}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    print(f"  OpenRouter OK ({len(content):,} chars)")
                    return content

                if resp.status_code in _NO_RETRY_CODES:
                    print(f"  OpenRouter {resp.status_code} (non-retryable): {resp.text[:200]}")
                    break

                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF * attempt
                    print(f"  Rate limited — wait {wait}s")
                    time.sleep(wait)
                    continue

                print(f"  OpenRouter {resp.status_code}: {resp.text[:200]}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF * attempt)

            except requests.exceptions.Timeout:
                print(f"  OpenRouter timeout (attempt {attempt})")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF)
            except requests.exceptions.ConnectionError as exc:
                print(f"  OpenRouter connection error: {exc}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF)
            except Exception as exc:
                print(f"  OpenRouter unexpected: {exc}")
                break

        print("  All attempts failed — rule-based fallback")
        return self._fallback_clean(raw_text)

    # ── Rule-based fallback ───────────────────────────────────────────────────

    @staticmethod
    def _fallback_clean(text: str) -> str:
        print("  Using rule-based fallback cleaner")
        UNWANTED = [
            'copyright', 'all rights reserved', 'published by',
            'acknowledgments', 'acknowledgements', 'bibliography',
            'about the author', 'isbn', 'library of congress',
            'first edition', 'printed in', 'dedication', 'foreword',
            'preface', 'table of contents', 'epub edition',
        ]
        lines          = text.splitlines()
        cleaned_lines: list[str] = []
        skip           = False
        skip_count     = 0

        for line in lines:
            lower = line.lower().strip()
            if not cleaned_lines and not line.strip():
                continue
            if any(kw in lower for kw in UNWANTED):
                skip = True
                skip_count = 0
            if not skip:
                cleaned_lines.append(line)
            else:
                skip_count += 1
                if (
                    skip_count > 60
                    or lower.startswith('chapter')
                    or (line.endswith('.') and len(line) > 40)
                ):
                    skip = False

        result = '\n'.join(cleaned_lines)
        print(f"  Rule-based result: {len(result):,} chars")
        return result


# ─── Module-level convenience ─────────────────────────────────────────────────

def clean_text(
    text: str,
    model_id: str = DEFAULT_LLM_MODEL,
    custom_prompt: str = '',
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    return LLMCleaner(model_id).clean_text(text, custom_prompt, progress_callback)


def clean_with_diff(
    text: str,
    model_id: str = DEFAULT_LLM_MODEL,
    custom_prompt: str = '',
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    return LLMCleaner(model_id).clean_with_diff(text, custom_prompt, progress_callback)