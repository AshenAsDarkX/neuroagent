import ctypes
import os
import traceback


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
from controller import BCIController


SUPPORTED_GOALS = [
    "Open Spotify",
    "Open Chrome",
    "Open File Explorer",
    "Search",
    "Close Window",
]


def main() -> None:
    config = AppConfig.load()
    app = BCIController(config)
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
