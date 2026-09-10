"""Development launcher for the Circuit Tracer backend.

Runs `app.py` and restarts it whenever a tracked Python source file changes,
so you never have to re-run the server while editing Python. Frontend files
(React/CSS) are not watched here; iterate on those with `npm run dev` against
the same running backend.

Usage (from the repository root):

    python dev_server.py

Exit with Ctrl+C; the backend child is stopped with it.
"""

import subprocess
import sys
import threading
from pathlib import Path

from watchfiles import watch

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Directories that are either dependency trees or not part of the live
# Python sources. Skipped by the watcher so edits there never restart the
# backend (and the watch set stays small).
EXCLUDED_TOP = {".venv", ".git", "frontend", "tests", "checkpoints", "__pycache__"}


def interesting(change, path):
    """Keep watchfiles from descending into the excluded directories."""
    relative = Path(path).resolve().relative_to(ROOT)
    return (
        relative.parts[0] not in EXCLUDED_TOP
        and "__pycache__" not in relative.parts
    )


def start_backend() -> subprocess.Popen:
    command = [PYTHON, str(ROOT / "app.py")]
    print(f"starting backend: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=str(ROOT))


def stop_backend(child: subprocess.Popen) -> None:
    child.terminate()
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def main() -> None:
    stop_event = threading.Event()
    child = start_backend()
    print(f"watching Python sources under {ROOT}", flush=True)
    print("press Ctrl+C to stop", flush=True)
    try:
        for changes in watch(
            ROOT,
            watch_filter=interesting,
            yield_on_timeout=True,
            stop_event=stop_event,
        ):
            restart = any(
                Path(path).suffix == ".py" for _change, path in changes
            )
            if not restart:
                continue
            print("change detected; restarting backend", flush=True)
            stop_backend(child)
            child = start_backend()
    except KeyboardInterrupt:
        print("stopping backend", flush=True)
    finally:
        stop_backend(child)


if __name__ == "__main__":
    main()
