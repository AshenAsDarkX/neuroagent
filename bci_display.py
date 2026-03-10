import threading
import tkinter as tk
from screeninfo import get_monitors


FLICKER_FREQUENCIES = [8, 9, 10, 11, 12, 13]


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
        self._status_lines = []
        self._selected_index = None
        self._flicker_running = True
        self.grid = None

        self._build_ui()
        self._start_flicker()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#121212", height=140)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        self.info_label = tk.Label(
            header,
            text="System starting...",
            font=("Consolas", 14),
            fg="#9be7ff",
            bg="#121212",
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

            period = int(1000 / (2 * FLICKER_FREQUENCIES[i]))
            self._flicker_targets.append(
                {
                    "label": label,
                    "period": period,
                    "state": False,
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
        for target in self._flicker_targets:
            self._flicker(target)

    def _flicker(self, target):
        if not self._flicker_running:
            return

        idx = target["index"]
        label = target["label"]
        has_text = bool(label.cget("text").strip())

        try:
            if self._selected_index == idx and has_text:
                label.config(bg="green", fg="black", bd=4, relief="solid")
            elif not has_text:
                label.config(bg="black", fg="white", bd=3, relief="solid")
            elif target["state"]:
                label.config(bg="white", fg="black", bd=3, relief="solid")
            else:
                label.config(bg="black", fg="white", bd=3, relief="solid")

            target["state"] = not target["state"]
            self.root.after(target["period"], lambda: self._flicker(target))
        except Exception:
            pass

    def set_info(self, text):
        if threading.current_thread() is threading.main_thread():
            self._set_info_now(text)
        else:
            self.root.after(0, lambda t=text: self._set_info_now(t))

    def _set_info_now(self, text):
        line = str(text).strip()
        if not line:
            return
        if self._status_lines and self._status_lines[-1] == line:
            return
        self._status_lines.append(line)
        self._status_lines = self._status_lines[-5:]
        self.info_label.config(text="\n".join(self._status_lines))
        self.root.update_idletasks()

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
