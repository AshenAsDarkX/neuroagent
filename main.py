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
from bci_display import BCIDisplay
from bci_decoder import EEG2CodeBCI
from controller import BCIController


def main() -> None:
    config = AppConfig.load()
    bci_screen = BCIDisplay()
    bci_screen.set_info("EEG2Code loading...")
    bci = EEG2CodeBCI(
        model_path=os.path.join(config.base_dir, "model", "EEG2Code_model.hdf5"),
        dataset_path=os.path.join(config.base_dir, "data", "VP1.mat"),
        status_callback=bci_screen.set_info,
    )
    bci_screen.set_info("Starting agent controller...")
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
