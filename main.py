import ctypes
import os
import subprocess
import time
import traceback
import requests


def _ensure_ollama_running() -> None:
    """
    Make sure Ollama server is running before the app starts.
    If it's already up, do nothing. If not, start it as a background process
    and wait until it responds on port 11434.
    """
    url = "http://localhost:11434/api/tags"

    # Check if already running
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            print("[Ollama] Server already running.")
            return
    except Exception:
        pass

    # Not running — start it
    print("[Ollama] Starting Ollama server...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,  # Windows: no console popup
        )
    except Exception as exc:
        print(f"[Ollama] Could not start Ollama: {exc}")
        return

    # Wait up to 30 seconds for it to come up
    for attempt in range(30):
        time.sleep(1)
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[Ollama] Server ready after {attempt + 1}s.")
                return
        except Exception:
            pass
        print(f"[Ollama] Waiting for server... ({attempt + 1}s)")

    print("[Ollama] Server did not start in time — continuing anyway.")


def _ensure_model_loaded(model: str = "gemma3:1b") -> None:
    """
    Force the model to fully load into GPU memory before the app starts.

    This is equivalent to running `ollama run gemma3:4b "hi"` in a terminal.
    It sends a real generate request and waits for a response, which forces
    Ollama to load the model weights into VRAM. Subsequent calls are instant.

    Without this, the first real LLM call during a BCI scan can fail with
    HTTP 500 because the model hasn't been loaded yet.
    """
    print(f"[Ollama] Loading {model} into memory...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "hi",
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,  # model load can take up to 60s on first run
        )
        if response.status_code == 200:
            print(f"[Ollama] {model} loaded and ready.")
        else:
            print(f"[Ollama] Model load got HTTP {response.status_code} — will retry on first use.")
    except Exception as exc:
        print(f"[Ollama] Model load failed ({exc}) — will retry on first use.")



def _enable_windows_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_enable_windows_dpi_awareness()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PADDLE_SUPPRESS_CCACHE_WARNING", "1")

from app_config import AppConfig
from bci_display import BCIDisplay
from bci_decoder import EEG2CodeBCI
from controller import BCIController


def main() -> None:
    _ensure_ollama_running()
    config = AppConfig.load()
    _ensure_model_loaded(config.llm_model)
    bci_screen = BCIDisplay()

    def show_phase(message: str) -> None:
        bci_screen.set_loading(True, message)

    def clear_phase() -> None:
        bci_screen.set_loading(False)

    # Show model ready status on the BCI display
    bci_screen.set_info(f"{config.llm_model} loaded and ready.")

    bci = EEG2CodeBCI(
        model_path=os.path.join(config.base_dir, "model", "EEG2Code_model.hdf5"),
        dataset_path=os.path.join(config.base_dir, "data", "VP1.mat"),
        status_callback=show_phase,
        clear_callback=clear_phase,
        correlation_callback=bci_screen.update_correlations,
        eeg_callback=bci_screen.update_eeg_wave,
    )
    bci_screen.set_loading(True, "Loading OmniParser...")
    app = BCIController(config, bci=bci, bci_screen=bci_screen)
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
    finally:
        print("Stopped")