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
from bci_decoder import EEG2CodeBCI
from controller import BCIController


def main() -> None:
    config = AppConfig.load()
    bci = EEG2CodeBCI(
        model_path="EEG2Code_model.hdf5",
        dataset_path="VP1.mat"
    )
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
