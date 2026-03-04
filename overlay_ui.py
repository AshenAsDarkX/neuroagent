from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List

from app_config import AppConfig
from models import DetectedElement
from utils import compute_page_slice

FLICKER_FREQUENCIES = [8, 9, 10, 11, 12, 13]


class FrequencyFlicker:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.targets: List[dict] = []
        self.running = False

    def clear(self) -> None:
        self.targets = []

    def add_target(self, widget: tk.Widget, frequency: int) -> None:
        period = int(1000 / (2 * frequency))
        self.targets.append(
            {
                "widget": widget,
                "period": period,
                "state": False,
            }
        )

    def start(self) -> None:
        self.running = True
        for target in self.targets:
            self._flicker(target)

    def stop(self) -> None:
        self.running = False

    def _flicker(self, target: dict) -> None:
        if not self.running:
            return

        widget = target["widget"]

        try:
            if target["state"]:
                widget.config(bg="white", fg="black")
            else:
                widget.config(bg="black", fg="white")

            target["state"] = not target["state"]
            self.root.after(target["period"], lambda: self._flicker(target))
        except Exception:
            pass


class OverlayUI:
    def __init__(
        self,
        config: AppConfig,
        on_scan: Callable[[], None],
        on_select: Callable[[int], None],
        on_quit: Callable[[], None],
    ) -> None:
        self.config = config
        self.on_scan = on_scan
        self.on_select = on_select
        self.on_quit = on_quit

        self.root = tk.Tk()
        self.root.title("BCI Manual")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        if self.config.overlay_position.lower() == "right":
            x = screen_w - self.config.overlay_width
        else:
            x = 0

        self.root.geometry(f"{self.config.overlay_width}x{screen_h}+{x}+0")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#2c3e50")
        self.flicker = FrequencyFlicker(self.root)

        self.status_label: tk.Label
        self.options_frame: tk.Frame

        self._build_ui()
        self._bind_keys()

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#1a252f", height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="BCI\nManual",
            font=("Arial", 18, "bold"),
            bg="#1a252f",
            fg="white",
            justify=tk.CENTER,
        )
        title.pack(pady=10)

        instructions = (
            "SPACE : Scan\n"
            "1-4   : Select action\n"
            "5     : More options\n"
            "6     : Go back\n"
            "Q/ESC : Quit\n"
        )

        instruction_frame = tk.Frame(self.root, bg="#34495e")
        instruction_frame.pack(fill=tk.X, padx=5, pady=5)

        instruction_label = tk.Label(
            instruction_frame,
            text=instructions,
            font=("Arial", 9),
            bg="#34495e",
            fg="white",
            justify=tk.LEFT,
        )
        instruction_label.pack(pady=5)

        ttk.Separator(self.root).pack(fill=tk.X, padx=5)

        self.status_label = tk.Label(
            self.root,
            text="Press SPACE to scan",
            font=("Arial", 10, "bold"),
            bg="#2c3e50",
            fg="#3498db",
            wraplength=self.config.overlay_width - 20,
            justify=tk.CENTER,
            pady=10,
        )
        self.status_label.pack(fill=tk.X)

        ttk.Separator(self.root).pack(fill=tk.X, padx=5)

        options_container = tk.Frame(self.root, bg="#2c3e50")
        options_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(options_container, bg="#2c3e50", highlightthickness=0)
        scrollbar = ttk.Scrollbar(options_container, orient="vertical", command=canvas.yview)

        self.options_frame = tk.Frame(canvas, bg="#2c3e50")

        canvas.create_window((0, 0), window=self.options_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.options_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _event: self.on_scan())
        for number in range(1, 7):
            self.root.bind(str(number), lambda _event, n=number: self.on_select(n))
        self.root.bind("q", lambda _event: self.on_quit())
        self.root.bind("<Escape>", lambda _event: self.on_quit())

    def clear_options(self) -> None:
        for widget in self.options_frame.winfo_children():
            widget.destroy()

    def set_status(self, text: str, color: str = "#f39c12") -> None:
        self.status_label.config(text=text, fg=color)

    def set_status_threadsafe(self, text: str, color: str = "#f39c12") -> None:
        self.root.after(0, lambda: self.set_status(text, color))

    def render_page(self, all_elements: List[DetectedElement], page: int, items_per_page: int) -> None:
        self.flicker.stop()
        self.flicker.clear()
        self.clear_options()

        total = len(all_elements)
        if total == 0:
            return

        start, end, has_prev, has_next = compute_page_slice(total, page, items_per_page)
        chunk = all_elements[start:end]

        for index, element in enumerate(chunk, 1):
            self._add_option_button(index, element.name)

        if has_next:
            self._add_option_button(5, "Other")

        if has_prev:
            self._add_option_button(6, "Go Back")

        hints = [f"1-{len(chunk)}"] if chunk else []
        if has_next:
            hints.append("5=More")
        if has_prev:
            hints.append("6=Back")

        self.set_status(
            text=f"OK {total} options | Page {page + 1}\nPress {', '.join(hints)}",
            color="#27ae60",
        )
        self.flicker.start()

    def _add_option_button(self, number: int, label: str) -> None:
        frame = tk.Frame(self.options_frame, bg="#2c3e50")
        frame.pack(fill=tk.X, pady=3)

        number_label = tk.Label(
            frame,
            text=f"[{number}]",
            font=("Arial", 12, "bold"),
            bg="#2c3e50",
            fg="#3498db",
            width=4,
        )
        number_label.pack(side=tk.LEFT)

        button = tk.Button(
            frame,
            text=label,
            font=("Arial", 12, "bold"),
            height=5,
            width=16,
            bg="#3498db",
            fg="white",
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            wraplength=self.config.overlay_width - 70,
            justify=tk.LEFT,
            command=lambda n=number: self.on_select(n),
        )
        button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        if number <= len(FLICKER_FREQUENCIES):
            self.flicker.add_target(button, FLICKER_FREQUENCIES[number - 1])

    def stop_flicker(self) -> None:
        self.flicker.stop()

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> None:
        self.root.after(delay_ms, callback)

    def run(self) -> None:
        self.root.mainloop()

    def stop(self) -> None:
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def get_hwnd(self) -> int:
        self.root.update()
        return int(self.root.winfo_id())

    def focus_overlay(self, overlay_hwnd: int | None) -> None:
        # disabled during scanning to prevent overlay being captured
        try:
            self.root.focus_force()
        except Exception:
            pass
