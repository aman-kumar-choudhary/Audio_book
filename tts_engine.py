"""
tts_engine.py
-------------
TTS interface — Kokoro-FastAPI only.
Connects to the local Kokoro server managed by kokoro_manager.py.
"""

import os

import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    KOKORO_API_URL,
    KOKORO_CPU_URL,
    KOKORO_DEFAULT_VOICES,
    KOKORO_HEALTH_URL,
    TEMP_FOLDER,
    USE_GPU,
)


def _make_session(retries: int = 3, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=['GET', 'POST'],
        ),
        pool_connections=10,
        pool_maxsize=20,
    )
    session.mount('http://',  adapter)
    session.mount('https://', adapter)
    return session


class TTSEngine:
    """
    Kokoro-FastAPI TTS wrapper.

    Tries the primary Kokoro URL first; falls back to CPU URL if unreachable.
    Raises RuntimeError if neither endpoint responds.
    """

    def __init__(self, use_gpu: bool = USE_GPU) -> None:
        self.use_gpu          = use_gpu and torch.cuda.is_available()
        self.kokoro_url       = KOKORO_API_URL
        self.available_voices: list = []
        self._session         = _make_session()

        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Try primary then CPU URL; raise if neither is reachable."""
        urls = [KOKORO_API_URL]
        if KOKORO_CPU_URL != KOKORO_API_URL:
            urls.append(KOKORO_CPU_URL)

        for url in urls:
            try:
                self._ping(url)
                self.kokoro_url = url
                label = 'GPU' if (self.use_gpu and url == KOKORO_API_URL) else 'CPU'
                print(f"  TTSEngine connected [{label}] → {url}")
                return
            except RuntimeError as exc:
                print(f"  Kokoro at {url} not available: {exc}")

        raise RuntimeError(
            f"Cannot connect to Kokoro-FastAPI.\n"
            f"Ensure Kokoro is running at {KOKORO_API_URL}\n"
            f"The app should have started it automatically — check console logs."
        )

    def _ping(self, url: str) -> None:
        """
        Verify a Kokoro endpoint is alive via /health then /v1/audio/voices.
        Raises RuntimeError on any failure.
        """
        # Health check
        try:
            h = self._session.get(f'{url}/health', timeout=5)
            if h.status_code != 200:
                raise RuntimeError(f"Health check returned {h.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Connection refused at {url}")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Health check timed out at {url}")

        # Fetch voice list
        try:
            v = self._session.get(f'{url}/v1/audio/voices', timeout=10)
            if v.status_code == 200:
                data = v.json()
                self.available_voices = data if isinstance(data, list) else data.get('voices', [])
                print(f"  Kokoro voices: {len(self.available_voices)} available")
        except Exception:
            pass  # voice list is informational; don't fail here

    # ── Public API ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text:  str,
        idx:   int,
        voice: str   = 'af_bella',
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> str:
        """
        Generate speech for one text chunk.
        Returns local path of the produced WAV file.
        Raises RuntimeError on failure.
        """
        if not text or not text.strip():
            raise ValueError(f"Chunk {idx} is empty")

        return self._synth(text, idx, voice, speed, pitch)

    def list_voices(self) -> list:
        self._ping(self.kokoro_url)
        return self.available_voices

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synth(
        self,
        text:      str,
        idx:       int,
        voice:     str,
        speed:     float,
        pitch:     float,
        _fallback: bool = False,
    ) -> str:
        actual_voice = self._map_voice(voice)
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        out_path = os.path.join(TEMP_FOLDER, f'chunk_{idx:04d}.wav')

        headers = {'X-Use-GPU': '1'} if self.use_gpu else {}
        payload = {
            'model':           'kokoro',
            'input':           text,
            'voice':           actual_voice,
            'response_format': 'wav',
            'speed':           speed,
        }

        print(f"  Chunk {idx:04d} | voice={actual_voice} | {len(text)} chars")

        try:
            resp = self._session.post(
                f'{self.kokoro_url}/v1/audio/speech',
                json=payload,
                headers=headers,
                timeout=180,
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Kokoro timeout on chunk {idx}")
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Kokoro connection lost on chunk {idx}: {exc}")

        if resp.status_code == 200:
            with open(out_path, 'wb') as fh:
                fh.write(resp.content)
            size_kb = os.path.getsize(out_path) / 1024
            if size_kb == 0:
                raise RuntimeError(f"Kokoro returned empty audio for chunk {idx}")
            print(f"  Chunk {idx:04d} done ({size_kb:.1f} KB)")
            return out_path

        # Voice rejected — retry once with safe default
        if resp.status_code == 400 and 'voice' in resp.text.lower() and not _fallback:
            print(f"  Voice '{actual_voice}' rejected — retrying with af_bella")
            return self._synth(text, idx, 'af_bella', speed, pitch, _fallback=True)

        raise RuntimeError(f"Kokoro HTTP {resp.status_code}: {resp.text[:300]}")

    def _map_voice(self, voice: str) -> str:
        if voice == 'default':
            return KOKORO_DEFAULT_VOICES.get('default', 'af_bella')
        if self.available_voices and voice in self.available_voices:
            return voice
        if voice in KOKORO_DEFAULT_VOICES:
            return KOKORO_DEFAULT_VOICES[voice]
        print(f"  Unknown voice '{voice}' — using af_bella")
        return 'af_bella'