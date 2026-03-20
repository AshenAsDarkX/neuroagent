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
    bci_screen = BCIDisplay()
    def show_phase(message: str) -> None:
        bci_screen.set_loading(True, message)

    def clear_phase() -> None:
        bci_screen.set_loading(False)

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