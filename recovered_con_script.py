import os
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app_config import AppConfig
from controller import BCIController


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
