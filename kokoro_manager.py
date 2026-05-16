"""
kokoro_manager.py
-----------------
Cross-platform Kokoro-FastAPI process manager.

Automatically:
  - Detects GPU availability (no manual config needed)
  - Locates the Kokoro-FastAPI directory (env var or auto-discovery)
  - Runs start-gpu.sh / start-cpu.sh on Linux/macOS
  - Runs start-gpu.ps1 / start-cpu.ps1 on Windows
  - Falls back to direct `python -m uvicorn` launch if scripts are missing
  - Health-checks and waits until Kokoro is ready

Works on Linux, macOS, Windows — GPU or CPU automatically.
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Read port/URL from config if available, otherwise use defaults ────────────
try:
    from config import KOKORO_API_URL, KOKORO_HEALTH_URL
except ImportError:
    KOKORO_API_URL    = "http://localhost:8880"
    KOKORO_HEALTH_URL = "http://localhost:8880/health"

_PORT            = int(KOKORO_API_URL.rstrip('/').split(':')[-1])
_HEALTH_TIMEOUT  = 300   # seconds to wait for Kokoro to become ready
_HEALTH_INTERVAL = 2     # seconds between health-check retries

# ── Kokoro-FastAPI directory discovery ───────────────────────────────────────
# Priority:
#   1. KOKORO_FASTAPI_DIR environment variable
#   2. Subdirectory named "Kokoro-FastAPI" next to this file
#   3. Common relative search paths from this file's location

def _find_kokoro_dir() -> Path | None:
    """Return the Kokoro-FastAPI root directory, or None if not found."""

    # 1. Explicit env override
    env_dir = os.getenv("KOKORO_FASTAPI_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
        print(f"  ⚠️  KOKORO_FASTAPI_DIR='{env_dir}' does not exist — ignoring")

    # 2. Search relative to this file (kokoro_manager.py)
    base = Path(__file__).resolve().parent
    candidates = [
        base / "Kokoro-FastAPI",
        base / "kokoro-fastapi",
        base / "kokoro_fastapi",
        base.parent / "Kokoro-FastAPI",
        base.parent / "kokoro-fastapi",
    ]
    for p in candidates:
        if p.is_dir() and (p / "api").is_dir():
            return p

    return None


# ── GPU auto-detection ────────────────────────────────────────────────────────

def _detect_gpu() -> bool:
    """
    Return True if a usable CUDA GPU is available.

    Tries (in order):
      1. torch.cuda.is_available()  — fast, reliable if torch is installed
      2. nvidia-smi                 — works even without torch
    """
    # Method 1: torch
    try:
        import torch
        available = torch.cuda.is_available()
        print(f"  GPU detection via torch: {'found' if available else 'not found'}")
        return available
    except ImportError:
        pass

    # Method 2: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"  GPU detection via nvidia-smi: found ({result.stdout.strip().splitlines()[0]})")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("  GPU detection: no CUDA GPU found — using CPU")
    return False


# ─────────────────────────────────────────────────────────────────────────────

class KokoroManager:
    """Manages a local Kokoro-FastAPI subprocess."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._use_gpu: bool = _detect_gpu()
        self._kokoro_dir: Path | None = _find_kokoro_dir()

        mode = "GPU" if self._use_gpu else "CPU"
        if self._kokoro_dir:
            print(f"  Kokoro-FastAPI dir  : {self._kokoro_dir}")
        else:
            print("  Kokoro-FastAPI dir  : not found — will try direct module launch")
        print(f"  Kokoro mode         : {mode}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start Kokoro if it is not already running on the configured port."""
        if self._is_already_running():
            print("  Kokoro already running — skipping start")
            return
        self._launch()

    def stop(self) -> None:
        """Gracefully stop the managed Kokoro subprocess."""
        if self._proc is None:
            return
        print("  Stopping Kokoro…")
        if platform.system() == "Windows":
            self._proc.terminate()
        else:
            try:
                self._proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None
        print("  Kokoro stopped")

    def wait_ready(self, timeout: int = _HEALTH_TIMEOUT) -> bool:
        """
        Block until /health returns HTTP 200.
        Returns True on success, False on timeout.
        """
        print(f"  Waiting for Kokoro at {KOKORO_HEALTH_URL} …")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(KOKORO_HEALTH_URL, timeout=5)
                if resp.status_code == 200:
                    try:
                        data   = resp.json()
                        gpu_ok = data.get("gpu_available", False)
                        print(f"  ✅ Kokoro ready  [{'GPU' if gpu_ok else 'CPU'}]")
                    except Exception:
                        print("  ✅ Kokoro ready")
                    return True
            except Exception:
                pass
            time.sleep(_HEALTH_INTERVAL)

        print(f"  ❌ Kokoro did not become ready within {timeout}s")
        return False

    def is_healthy(self) -> bool:
        try:
            resp = requests.get(KOKORO_HEALTH_URL, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_already_running(self) -> bool:
        try:
            resp = requests.get(KOKORO_HEALTH_URL, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _launch(self) -> None:
        """
        Try each launch candidate in order until one stays alive.
        Raises RuntimeError if all candidates fail.
        """
        mode = "GPU" if self._use_gpu else "CPU"
        print(f"  Launching Kokoro [{mode}] on port {_PORT}…")

        env = os.environ.copy()
        env["PORT"]           = str(_PORT)
        env["HOST"]           = "0.0.0.0"
        env["KOKORO_USE_GPU"] = "1" if self._use_gpu else "0"
        if self._use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        # Ensure the kokoro_fastapi package is importable when launching from
        # source (Kokoro-FastAPI/api/ is the package root).
        if self._kokoro_dir and (self._kokoro_dir / "api").is_dir():
            api_path = str(self._kokoro_dir / "api")
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{api_path}:{existing}" if existing else api_path

        for cmd, cwd in self._launch_candidates():
            label = " ".join(str(c) for c in cmd)
            print(f"    Trying: {label}")
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    env=env,
                    # Uncomment to silence Kokoro's own output:
                    # stdout=subprocess.DEVNULL,
                    # stderr=subprocess.STDOUT,
                )
                time.sleep(2.0)   # give it a moment to crash or stabilise
                if self._proc.poll() is not None:
                    print("    ↳ exited immediately — trying next")
                    self._proc = None
                    continue

                print(f"  Kokoro launched  (PID {self._proc.pid})")
                return

            except (FileNotFoundError, OSError, PermissionError) as exc:
                print(f"    ↳ failed ({exc})")
                self._proc = None
                continue

        raise RuntimeError(
            "❌ Could not launch Kokoro-FastAPI.\n"
            "   Please start it manually, e.g.:\n"
            "     cd Kokoro-FastAPI && bash start-gpu.sh\n"
            "   Then re-run app.py."
        )

    def _launch_candidates(self) -> list[tuple[list, Path | None]]:
        """
        Build an ordered list of (command, cwd) pairs to try.

        Priority
        --------
        1. Direct uvicorn — fast, no pip reinstall, works when kokoro is installed
        2. Shell / PowerShell startup scripts — fallback if uvicorn import fails
        3. Bare `kokoro` CLI — last resort

        The shell scripts (start-gpu.sh etc.) always run `pip install` on every
        invocation which wastes ~3s and spams the console. We only fall back to
        them when the direct launch fails.
        """
        candidates: list[tuple[list, Path | None]] = []

        kdir       = self._kokoro_dir   # may be None
        py         = sys.executable
        host       = "0.0.0.0"
        port       = str(_PORT)
        is_windows = platform.system() == "Windows"
        is_unix    = not is_windows

        # ── 1. Direct uvicorn — preferred, no pip reinstall overhead ─────────
        # The api/ subdirectory is the package root for Kokoro-FastAPI when
        # run from source; pass it as PYTHONPATH so the import resolves.
        if kdir and (kdir / "api").is_dir():
            api_dir = str(kdir / "api")
        else:
            api_dir = None

        candidates += [
            ([py, "-m", "uvicorn", "kokoro_fastapi.main:app",
              "--host", host, "--port", port], api_dir or kdir),
            ([py, "-m", "kokoro_fastapi",
              "--host", host, "--port", port], api_dir or kdir),
        ]

        # Windows py-launcher variant
        if is_windows:
            candidates += [
                (["py", "-m", "uvicorn", "kokoro_fastapi.main:app",
                  "--host", host, "--port", port], api_dir or kdir),
            ]

        # ── 2. Shell / PowerShell scripts (fallback) ─────────────────────────
        if kdir:
            if self._use_gpu:
                if is_unix:
                    script = kdir / "start-gpu.sh"
                    if script.exists():
                        candidates.append((["bash", str(script)], kdir))
                else:
                    script = kdir / "start-gpu.ps1"
                    if script.exists():
                        candidates.append((
                            ["powershell", "-ExecutionPolicy", "Bypass",
                             "-File", str(script)],
                            kdir
                        ))
            else:
                if is_unix:
                    script = kdir / "start-cpu.sh"
                    if script.exists():
                        candidates.append((["bash", str(script)], kdir))
                else:
                    script = kdir / "start-cpu.ps1"
                    if script.exists():
                        candidates.append((
                            ["powershell", "-ExecutionPolicy", "Bypass",
                             "-File", str(script)],
                            kdir
                        ))

        # ── 3. Bare `kokoro` CLI — last resort ───────────────────────────────
        kokoro_bin = shutil.which("kokoro")
        if kokoro_bin:
            candidates.append(([kokoro_bin, "--host", host, "--port", port], None))

        sibling_bin = Path(py).parent / ("kokoro.exe" if is_windows else "kokoro")
        if sibling_bin.exists():
            candidates.append(([str(sibling_bin), "--host", host, "--port", port], None))

        return candidates


# ─── Module-level singleton ───────────────────────────────────────────────────

_manager: KokoroManager | None = None


def get_manager() -> KokoroManager:
    global _manager
    if _manager is None:
        _manager = KokoroManager()
    return _manager


def start_kokoro() -> bool:
    """
    Start Kokoro and wait until healthy.
    Returns True if ready, False if timed out or failed to launch.
    """
    try:
        mgr = get_manager()
        mgr.start()
        return mgr.wait_ready()
    except RuntimeError as exc:
        print(exc)
        return False