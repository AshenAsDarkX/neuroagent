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

        # Left side — status text
        self._header_left = tk.Frame(self.header, bg=self._header_bg_default)
        self._header_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(
            self._header_left,
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

        # Right side — EEG waveform display
        self._eeg_canvas = tk.Canvas(
            self.header,
            bg="#050d0d",
            width=380,
            height=130,
            highlightthickness=1,
            highlightbackground="#1a4a4a",
        )
        self._eeg_canvas.pack(side=tk.RIGHT, padx=(0, 16), pady=6)

        # EEG waveform state
        self._eeg_signal: list = []      # raw EEG samples (ch 0)
        self._eeg_code: list = []        # target stimulation code
        self._eeg_label_text: str = ""
        self._eeg_animating: bool = False
        self._eeg_scroll_offset: int = 0
        self._eeg_anim_job = None

        # Draw placeholder
        self._draw_eeg_placeholder()

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

        # CCA correlation bar chart — shown during EEG decode phase
        self._corr_canvas = tk.Canvas(
            self.root,
            bg="#0a0a0a",
            height=110,
            highlightthickness=0,
        )
        self._corr_canvas.pack(fill=tk.X, padx=20, pady=(0, 10))
        self._corr_values: list = []
        self._corr_selected: int = -1

        self.root.update()

    def _draw_eeg_placeholder(self):
        """Draw idle state — flat noise line with label."""
        try:
            c = self._eeg_canvas
            c.delete("all")
            w, h = 380, 130
            # Border label
            c.create_text(8, 6, text="EEG Signal Monitor", fill="#1a8a8a",
                          font=("Consolas", 8), anchor="nw")
            c.create_text(w - 8, 6, text="Ch.0 | Code",
                          fill="#1a5a5a", font=("Consolas", 8), anchor="ne")
            # Flat idle line
            mid = h // 2
            c.create_line(8, mid, w - 8, mid, fill="#0d3d3d", width=1)
            c.create_text(w // 2, mid + 18,
                          text="Waiting for EEG signal...",
                          fill="#1a5a5a", font=("Consolas", 9))
        except Exception:
            pass

    def update_eeg_wave(self, eeg_ch0: list, code: list, label: str = ""):
        """
        Update EEG waveform. If animation already running (live mode),
        just update the data buffer smoothly without restarting.
        """
        if threading.current_thread() is threading.main_thread():
            self._update_eeg_data(eeg_ch0, code, label)
        else:
            self.root.after(0, lambda e=eeg_ch0, cd=code, lb=label:
                            self._update_eeg_data(e, cd, lb))

    def _update_eeg_data(self, eeg_ch0: list, code: list, label: str):
        self._eeg_signal = list(eeg_ch0)
        self._eeg_code = list(code)
        self._eeg_label_text = label
        # Always show the latest end of the buffer
        self._eeg_scroll_offset = max(0, len(eeg_ch0) - 300)
        if not self._eeg_animating:
            self._start_eeg_animation(eeg_ch0, code, label)

    def _start_eeg_animation(self, eeg_ch0: list, code: list, label: str):
        # Stop any running animation
        if self._eeg_anim_job is not None:
            try:
                self.root.after_cancel(self._eeg_anim_job)
            except Exception:
                pass
        self._eeg_signal = list(eeg_ch0)
        self._eeg_code = list(code)
        self._eeg_label_text = label
        self._eeg_animating = True
        self._eeg_scroll_offset = 0
        self._animate_eeg()

    def _animate_eeg(self):
        if not self._eeg_animating:
            return
        try:
            self._render_eeg_frame()
            self._eeg_scroll_offset += 8
            # For live mode (large buffer), keep rendering;
            # for single-shot mode, stop after one pass
            max_offset = max(len(self._eeg_signal), 200)
            if self._eeg_scroll_offset > max_offset:
                # Reset to end of buffer for live scrolling
                self._eeg_scroll_offset = max(0, len(self._eeg_signal) - 200)
            self._eeg_anim_job = self.root.after(40, self._animate_eeg)
        except Exception:
            pass

    def _render_eeg_frame(self):
        try:
            c = self._eeg_canvas
            c.delete("all")
            W, H = 380, 130

            sig = self._eeg_signal
            code = self._eeg_code
            offset = self._eeg_scroll_offset
            n_sig = len(sig)
            if n_sig == 0:
                return

            # Layout
            top_pad = 18
            bot_pad = 18
            left_pad = 8
            right_pad = 8
            plot_w = W - left_pad - right_pad
            plot_h = H - top_pad - bot_pad
            eeg_h = int(plot_h * 0.62)   # top 62% for EEG trace
            code_h = plot_h - eeg_h - 4  # bottom strip for code

            # Background grid lines
            for gi in range(1, 4):
                y = top_pad + int(eeg_h * gi / 4)
                c.create_line(left_pad, y, W - right_pad, y,
                              fill="#0d2a2a", width=1)

            # --- EEG trace ---
            # Normalise amplitude to fit plot height
            sig_max = max(abs(v) for v in sig) or 1.0
            mid_y = top_pad + eeg_h // 2

            # Determine which samples to show (scroll window)
            samples_to_show = min(n_sig, plot_w)
            start = min(offset, n_sig - samples_to_show)
            start = max(0, start)
            window = sig[start: start + samples_to_show]

            eeg_points = []
            for xi, val in enumerate(window):
                px = left_pad + int(xi * plot_w / max(len(window) - 1, 1))
                py = mid_y - int((val / sig_max) * (eeg_h // 2) * 0.85)
                eeg_points.append((px, py))

            if len(eeg_points) >= 2:
                flat = [coord for pt in eeg_points for coord in pt]
                c.create_line(*flat, fill="#00cc88", width=1, smooth=False)

            # --- Stimulation code overlay ---
            if code:
                code_window = code[start: start + samples_to_show]
                code_y_top = top_pad + eeg_h + 6
                code_y_high = code_y_top
                code_y_low = code_y_top + code_h - 2
                prev_px = left_pad
                prev_bit = code_window[0] if code_window else 0

                for xi, bit in enumerate(code_window):
                    px = left_pad + int(xi * plot_w / max(len(code_window) - 1, 1))
                    py = code_y_high if bit else code_y_low
                    prev_py = code_y_high if prev_bit else code_y_low
                    # Horizontal segment
                    c.create_line(prev_px, prev_py, px, prev_py,
                                  fill="#0088bb", width=1)
                    # Vertical edge on transition
                    if bit != prev_bit:
                        c.create_line(px, prev_py, px, py,
                                      fill="#0088bb", width=1)
                    prev_px = px
                    prev_bit = bit

            # Labels
            c.create_text(left_pad, 4,
                          text="EEG Signal Monitor",
                          fill="#1a9a9a", font=("Consolas", 8), anchor="nw")
            c.create_text(W - right_pad, 4,
                          text=self._eeg_label_text,
                          fill="#00ffe0", font=("Consolas", 8, "bold"), anchor="ne")

            # Legend dots
            c.create_rectangle(left_pad, H - 14, left_pad + 14, H - 6,
                                fill="#00cc88", outline="")
            c.create_text(left_pad + 18, H - 10,
                          text="EEG ch.0", fill="#00cc88",
                          font=("Consolas", 7), anchor="w")
            c.create_rectangle(left_pad + 80, H - 14, left_pad + 94, H - 6,
                                fill="#0088bb", outline="")
            c.create_text(left_pad + 98, H - 10,
                          text="Stim code", fill="#0088bb",
                          font=("Consolas", 7), anchor="w")

            c.update_idletasks()
        except Exception as e:
            print(f"[EEG Display] render error: {e}")

    def clear_eeg_wave(self):
        """Reset to idle placeholder."""
        self._eeg_animating = False
        if self._eeg_anim_job is not None:
            try:
                self.root.after_cancel(self._eeg_anim_job)
            except Exception:
                pass
        if threading.current_thread() is threading.main_thread():
            self._draw_eeg_placeholder()
        else:
            self.root.after(0, self._draw_eeg_placeholder)

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

    def update_correlations(self, correlations: list, selected_idx: int = -1):
        """
        Update the CCA correlation bar chart.
        Called after each trial with the current mean correlations (list of 6 floats).
        selected_idx: index of the highest bar (-1 = none yet selected).
        """
        if threading.current_thread() is threading.main_thread():
            self._draw_correlations(correlations, selected_idx)
        else:
            self.root.after(0, lambda c=correlations, s=selected_idx:
                           self._draw_correlations(c, s))

    def _draw_correlations(self, correlations: list, selected_idx: int):
        try:
            self._corr_values = list(correlations)
            self._corr_selected = selected_idx
            canvas = self._corr_canvas

            # Force geometry update so winfo_width is accurate
            canvas.update_idletasks()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 50:
                w = self.monitor.width - 40
            if h < 50:
                h = 110

            canvas.delete("all")
            canvas.config(width=w, height=h)

            n = len(correlations)
            if n == 0:
                return

            # Title row
            canvas.create_text(
                w // 2, 6,
                text="CCA Correlation — EEG Signal Match per Target",
                fill="#9be7ff",
                font=("Consolas", 10),
                anchor="n",
            )

            padding = 20
            bar_area_w = w - padding * 2
            gap = 8
            bar_w = int((bar_area_w - gap * (n - 1)) / n)
            max_val = max(max(correlations), 0.70)
            top_margin = 22   # below title
            bot_margin = 22   # above bottom for labels
            bar_max_h = h - top_margin - bot_margin

            for i, val in enumerate(correlations):
                x = padding + i * (bar_w + gap)
                bar_h = max(2, int((val / max_val) * bar_max_h))
                y_bot = h - bot_margin
                y_top = y_bot - bar_h

                if i == selected_idx:
                    colour  = "#00ffe0"
                    outline = "#ffffff"
                    txt_col = "#ffffff"
                elif val == max(correlations):
                    colour  = "#0088aa"
                    outline = "#00ccdd"
                    txt_col = "#cccccc"
                else:
                    colour  = "#1a3a4a"
                    outline = "#2a5a6a"
                    txt_col = "#888888"

                canvas.create_rectangle(
                    x, y_top, x + bar_w, y_bot,
                    fill=colour, outline=outline, width=1,
                )

                # Value above bar
                canvas.create_text(
                    x + bar_w // 2, y_top - 2,
                    text=f"{val:.3f}",
                    fill=txt_col,
                    font=("Consolas", 8),
                    anchor="s",
                )

                # Target label below bar
                canvas.create_text(
                    x + bar_w // 2, y_bot + 3,
                    text=f"T{i+1}",
                    fill="#00ffe0" if i == selected_idx else "#555555",
                    font=("Consolas", 9, "bold"),
                    anchor="n",
                )

            # Match annotation on the right side
            if selected_idx >= 0 and selected_idx < len(correlations):
                conf = correlations[selected_idx]
                canvas.create_text(
                    w - 8, h // 2,
                    text=f"✓ T{selected_idx+1} matched  {conf:.3f}",
                    fill="#00ffe0",
                    font=("Consolas", 10, "bold"),
                    anchor="e",
                )

            canvas.update_idletasks()
        except Exception as e:
            print(f"[Display] Correlation chart error: {e}")

    def clear_correlations(self):
        """Clear the correlation chart."""
        if threading.current_thread() is threading.main_thread():
            self._corr_canvas.delete("all")
            self._corr_canvas.update_idletasks()
        else:
            self.root.after(0, lambda: self._corr_canvas.delete("all"))

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