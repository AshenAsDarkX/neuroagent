import threading
import time
import tkinter as tk
import numpy as np
import scipy.io
import os
from screeninfo import get_monitors


REFRESH_RATE = 60

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "VP1.mat"
)

data = scipy.io.loadmat(DATASET_PATH)

stim_bits = data["test_data_y"]

SAMPLING_RATE = 600
STIM_REFRESH_RATE = 60
samples_per_bit = SAMPLING_RATE // STIM_REFRESH_RATE

TARGET_COUNT = 6

STIM_CODES = []

for i in range(TARGET_COUNT):
    raw = stim_bits[i].reshape(-1)
    bits = (raw > 0).astype(int)
    downsampled = bits[::samples_per_bit]
    STIM_CODES.append(downsampled)


class BCIDisplay:

    def __init__(self):

        monitors = get_monitors()

        # find the non-primary monitor (DISPLAY5)
        self.monitor = None
        for m in monitors:
            if not m.is_primary:
                self.monitor = m
                break

        if self.monitor is None:
            raise RuntimeError("Second monitor not detected")

        self.root = tk.Tk()
        self.root.title("NeuroAgent")

        # place window on second monitor
        self.root.geometry(
            f"{self.monitor.width}x{self.monitor.height}+{self.monitor.x}+{self.monitor.y}"
        )
        self.root.attributes("-topmost", True)
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.root.configure(bg="black")

        self.labels = []
        self._flicker_targets = []
        self._selected_index = None
        self._flicker_running = True
        self._loader_running = False
        self._loader_job = None
        self._loader_index = 0
        self._loader_text = "Loading"
        self._loader_width = 18
        self._header_bg_default = "#121212"
        self._info_fg_default = "#9be7ff"
        self._flicker_start_time = 0.0
        self._frame_index = 0
        self._frame_interval = max(1, int(1000 / REFRESH_RATE))
        self.grid = None

        self._build_ui()
        self._start_flicker()

    def _build_ui(self):
        self.header = tk.Frame(self.root, bg=self._header_bg_default, height=140)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)

        self.info_label = tk.Label(
            self.header,
            text="System starting...",
            font=("Consolas", 14),
            fg=self._info_fg_default,
            bg=self._header_bg_default,
            justify=tk.LEFT,
            anchor="w",
            padx=20,
            pady=10,
        )
        self.info_label.pack(fill=tk.X, padx=20, pady=(10, 10))

        grid = tk.Frame(self.root, bg="black")
        grid.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        self.grid = grid

        for r in range(2):
            grid.rowconfigure(r, weight=1, uniform="option_row")

        for c in range(3):
            grid.columnconfigure(c, weight=1, uniform="option_col")

        for i in range(6):
            label = tk.Label(
                grid,
                text="",
                font=("Arial", 20, "bold"),
                fg="white",
                bg="black",
                justify="center",
                wraplength=460,
                bd=3,
                relief="solid",
                highlightthickness=0,
            )

            row = i // 3
            col = i % 3
            label.grid(row=row, column=col, sticky="nsew", padx=12, pady=12)
            self.labels.append(label)

            self._flicker_targets.append(
                {
                    "label": label,
                    "index": i,
                }
            )

        grid.bind("<Configure>", self._on_grid_resize)
        self.root.update()

    def _on_grid_resize(self, event):
        if not self.labels:
            return

        cols = 3
        rows = 2
        cell_w = max(160, int(event.width / cols) - 24)
        cell_h = max(100, int(event.height / rows) - 24)

        wrap = max(120, cell_w - 24)
        font_size = max(20, min(30, int(cell_h * 0.17)))

        for label in self.labels:
            label.config(wraplength=wrap, font=("Arial", font_size, "bold"))

    def _start_flicker(self):
        self._flicker_start_time = time.perf_counter()
        self._frame_index = 0
        self._update_flicker_frame()

    def _update_flicker_frame(self):
        if not self._flicker_running:
            return

        try:
            elapsed = time.perf_counter() - self._flicker_start_time
            frame = int(elapsed * REFRESH_RATE)
            self._frame_index = frame

            for target in self._flicker_targets:
                idx = target["index"]
                label = target["label"]
                if idx >= len(STIM_CODES):
                    continue

                code = STIM_CODES[idx]
                bit = code[frame % len(code)]

                if self._selected_index == idx and label.cget("text").strip():
                    label.config(bg="green", fg="black", bd=4, relief="solid")
                    continue

                if not label.cget("text").strip():
                    label.config(bg="black", fg="white", bd=3, relief="solid")
                    continue

                if bit == 1:
                    label.config(bg="white", fg="black", bd=3, relief="solid")
                else:
                    label.config(bg="black", fg="white", bd=3, relief="solid")
        except Exception:
            pass
        finally:
            if self._flicker_running:
                self.root.after(self._frame_interval, self._update_flicker_frame)

    def set_info(self, text):
        if threading.current_thread() is threading.main_thread():
            self._set_info_now(text)
        else:
            self.root.after(0, lambda t=text: self._set_info_now(t))

    def _set_info_now(self, text):
        line = str(text).strip()
        if self._loader_running:
            self._loader_text = line or self._loader_text
            return
        self.info_label.config(text=line)
        self.root.update_idletasks()

    def clear_info(self):
        if threading.current_thread() is threading.main_thread():
            self.info_label.config(text="")
            self.root.update_idletasks()
        else:
            self.root.after(0, self.clear_info)

    def set_loading_text(self, text):
        if threading.current_thread() is threading.main_thread():
            line = str(text).strip()
            if line:
                self._loader_text = line
        else:
            self.root.after(0, lambda t=text: self.set_loading_text(t))

    def set_loading(self, enabled, text="Loading"):
        if threading.current_thread() is threading.main_thread():
            self._set_loading_now(enabled, text)
        else:
            self.root.after(0, lambda e=enabled, t=text: self._set_loading_now(e, t))

    def _set_loading_now(self, enabled, text):
        if enabled:
            self._loader_text = str(text).strip() or self._loader_text or "Loading"
            if not self._loader_running:
                self._loader_running = True
                self._loader_index = 0
                self._animate_loader()
            return

        self._loader_running = False
        if self._loader_job is not None:
            try:
                self.root.after_cancel(self._loader_job)
            except Exception:
                pass
            self._loader_job = None
        self.info_label.config(text="")
        self.root.update_idletasks()

    def _animate_loader(self):
        if not self._loader_running:
            return

        cycle = self._loader_width * 2 - 2
        pos = self._loader_index % cycle
        if pos >= self._loader_width:
            pos = cycle - pos

        filled = max(1, pos + 1)
        bar = "|" * filled + "_" * (self._loader_width - filled)
        self.info_label.config(text=f"{self._loader_text} {bar}")
        self._loader_index += 1
        self._loader_job = self.root.after(140, self._animate_loader)
        if self._loader_running:
            self.root.update_idletasks()

    def flash_signal_rejection(self):
        if threading.current_thread() is threading.main_thread():
            self._flash_signal_rejection_now()
        else:
            self.root.after(0, self._flash_signal_rejection_now)

    def _flash_signal_rejection_now(self):
        try:
            reject_bg = "#331111"
            reject_fg = "#ffb3b3"
            self.header.config(bg=reject_bg)
            self.info_label.config(bg=reject_bg, fg=reject_fg)
            self.root.after(220, self._reset_signal_rejection_flash)
        except Exception:
            pass

    def _reset_signal_rejection_flash(self):
        try:
            self.header.config(bg=self._header_bg_default)
            self.info_label.config(bg=self._header_bg_default, fg=self._info_fg_default)
        except Exception:
            pass

    def show_actions(self, actions):
        self._selected_index = None

        for i in range(6):
            if i < len(actions):
                text = f"{actions[i]}"
            else:
                text = ""
            self.labels[i].config(text=text, bg="black", fg="white")

        self.root.update_idletasks()

    def highlight(self, index):
        try:
            idx = int(index)
        except Exception:
            return

        if idx < 0:
            return

        self._selected_index = idx % len(self.labels)

        for i, lbl in enumerate(self.labels):
            if i == self._selected_index:
                lbl.config(bg="green", fg="black", bd=4, relief="solid")
            else:
                lbl.config(bd=3, relief="solid")

        self.root.update_idletasks()
