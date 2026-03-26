from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageGrab

from app_config import AppConfig
from models import DetectedElement, ParseResult
from utils import bbox_ratio_xyxy_to_pixels, safe_text


class OmniParserEngine:
    def __init__(self, config: AppConfig, status_callback=None) -> None:
        self.config = config
        self._status_callback = status_callback
        self.omni_available = False
        self._omni: Dict[str, Any] = {}
        self._device: str = "cpu"
        self._capture_offset: Tuple[int, int] = (0, 0)
        self._init_omniparser()

    def _status(self, message: str) -> None:
        if not self._status_callback:
            return
        try:
            self._status_callback(message)
        except Exception:
            pass

    def _init_omniparser(self) -> None:
        try:
            if not os.path.isdir(self.config.omniparser_dir):
                raise FileNotFoundError(f"OmniParser folder not found: {self.config.omniparser_dir}")

            if self.config.omniparser_dir not in sys.path:
                sys.path.insert(0, self.config.omniparser_dir)

            from util.utils import (
                check_ocr_box,
                get_caption_model_processor,
                get_som_labeled_img,
                get_yolo_model,
            )

            yolo_path = os.path.join(self.config.omniparser_dir, "weights", "icon_detect", "model.pt")
            caption_path = os.path.join(self.config.omniparser_dir, "weights", "icon_caption_florence")

            if not os.path.isfile(yolo_path):
                raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
            if not os.path.isdir(caption_path):
                raise FileNotFoundError(f"Caption model folder not found: {caption_path}")

            print("OmniParser detected")
            self._status("OmniParser detected")
            print("Loading models...")
            self._status("Loading models...")
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
            print(f"OmniParser device: {self._device}")

            start = time.time()
            self._status("YOLO loading...")
            yolo_model = get_yolo_model(model_path=yolo_path, device=self._device)
            print(f"YOLO loaded ({time.time() - start:.1f}s)")
            self._status("YOLO loaded")

            start = time.time()
            caption_model_processor = get_caption_model_processor(
                model_name="florence2",
                model_name_or_path=caption_path,
                device=self._device,
            )
            print(f"Florence2 loaded ({time.time() - start:.1f}s)")
            self._status("Florence2 loaded")

            self._omni = {
                "check_ocr_box": check_ocr_box,
                "get_som_labeled_img": get_som_labeled_img,
                "yolo_model": yolo_model,
                "caption_model_processor": caption_model_processor,
                "device": self._device,
            }
            self.omni_available = True
            print("OmniParser ready")
            self._status("OmniParser ready")
        except Exception as exc:
            print(f"OmniParser init failed: {exc}")
            self._status(f"OmniParser init failed: {exc}")
            traceback.print_exc()
            raise

    def capture_main_monitor(self) -> Optional[Image.Image]:
        """Capture only DISPLAY1 for parsing."""
        main_bbox = self._get_main_monitor_bbox()
        screenshot = self._capture_fullscreen_imagegrab(bbox=main_bbox)

        if screenshot is None:
            print("[Capture] All capture methods failed.")
            return None

        self._capture_offset = (main_bbox[0], main_bbox[1])
        return screenshot

    def capture_active_window_on_main_monitor(
        self,
        fallback_to_main: bool = True,
    ) -> Optional[Image.Image]:
        """Capture active window area clipped to DISPLAY1 only."""
        main_left, main_top, main_right, main_bottom = self._get_main_monitor_bbox()

        try:
            import win32gui

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                print("[Capture] No active window handle.")
                if fallback_to_main:
                    return self.capture_main_monitor()
                return None

            left, top, right, bottom = win32gui.GetWindowRect(hwnd)

            clip_left = max(left, main_left)
            clip_top = max(top, main_top)
            clip_right = min(right, main_right)
            clip_bottom = min(bottom, main_bottom)

            if clip_left >= clip_right or clip_top >= clip_bottom:
                print("[Capture] Active window is outside DISPLAY1.")
                if fallback_to_main:
                    return self.capture_main_monitor()
                return None

            bbox = (clip_left, clip_top, clip_right, clip_bottom)
            clip_w = clip_right - clip_left
            clip_h = clip_bottom - clip_top

            # Ignore tiny windows (system tray popups, tooltips, notification toasts).
            # Anything smaller than 300×200 is not a real app window.
            MIN_W, MIN_H = 300, 200
            if clip_w < MIN_W or clip_h < MIN_H:
                print(
                    f"[Capture] Active window too small ({clip_w}×{clip_h}) "
                    f"— likely a tray popup. Falling back to full desktop."
                )
                return self.capture_main_monitor()

            screenshot = self._capture_fullscreen_imagegrab(bbox=bbox)
            if screenshot is None:
                if fallback_to_main:
                    return self.capture_main_monitor()
                return None

            self._capture_offset = (bbox[0], bbox[1])
            print(f"[Capture] Active window clipped to DISPLAY1 bbox={bbox} size={screenshot.size}")
            return screenshot
        except Exception as exc:
            print(f"[Capture] Active-window capture failed: {exc}")
            if fallback_to_main:
                return self.capture_main_monitor()
            return None

    def _get_main_monitor_bbox(self) -> Tuple[int, int, int, int]:
        """Return DISPLAY1 bounds as (left, top, right, bottom)."""
        try:
            from screeninfo import get_monitors

            monitors = get_monitors()
            primary = next((monitor for monitor in monitors if monitor.is_primary), None)
            if primary is not None:
                return (
                    int(primary.x),
                    int(primary.y),
                    int(primary.x + primary.width),
                    int(primary.y + primary.height),
                )
        except Exception:
            pass

        return (0, 0, 1920, 1080)

    # ------------------------------------------------------------------
    # Capture backends
    # ------------------------------------------------------------------

    def _capture_fullscreen_win32(self) -> Optional[Image.Image]:
        """Capture the full virtual desktop using win32api."""
        try:
            import win32api
            import win32con
            import win32gui
            import win32ui

            # SM_XVIRTUALSCREEN / SM_YVIRTUALSCREEN / SM_CXVIRTUALSCREEN /
            # SM_CYVIRTUALSCREEN cover all monitors in a multi-monitor setup
            left   = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top    = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            width  = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

            hdesktop = win32gui.GetDesktopWindow()
            desktop_dc = win32gui.GetWindowDC(hdesktop)
            mfc_dc = win32ui.CreateDCFromHandle(desktop_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (left, top), win32con.SRCCOPY)

            bmp_info = bitmap.GetInfo()
            bmp_str  = bitmap.GetBitmapBits(True)

            img = Image.frombuffer(
                "RGB",
                (bmp_info["bmWidth"], bmp_info["bmHeight"]),
                bmp_str, "raw", "BGRX", 0, 1,
            )

            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hdesktop, desktop_dc)

            print(f"[Capture] Full desktop via win32 ({width}x{height})")
            return img

        except Exception as exc:
            print(f"[Capture] win32 fullscreen failed: {exc}")
            return None

    def _capture_fullscreen_imagegrab(
        self,
        bbox: Tuple[int, int, int, int] | None = None,
    ) -> Optional[Image.Image]:
        """Capture a screen region via PIL ImageGrab."""
        try:
            if bbox is None:
                bbox = self._get_main_monitor_bbox()
            img = ImageGrab.grab(bbox=bbox)
            img = img.convert("RGB")
            print(f"[Capture] ImageGrab bbox={bbox} size={img.size}")
            return img
        except Exception as exc:
            print(f"[Capture] ImageGrab fallback failed: {exc}")
            return None

    def capture_specific_window(self, hwnd: int) -> Optional[Image.Image]:
        """Capture only the window identified by hwnd.

        Used after a click opens a specific window — gives OmniParser a
        clean, focused image of just that app rather than the full desktop,
        which produces more relevant and precise element detection.

        Returns None if the window can't be found or captured.
        """
        try:
            import win32gui
            import win32ui
            import win32con

            # Make sure the window still exists
            if not win32gui.IsWindow(hwnd):
                print(f"[Capture] HWND {hwnd} is no longer valid.")
                return None

            # Bring it to front so we get a fully rendered frame
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass  # Non-fatal — still attempt capture

            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width  = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                print(f"[Capture] Window {hwnd} has zero size, skipping.")
                return None

            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            bmp_info = bitmap.GetInfo()
            bmp_str  = bitmap.GetBitmapBits(True)

            img = Image.frombuffer(
                "RGB",
                (bmp_info["bmWidth"], bmp_info["bmHeight"]),
                bmp_str, "raw", "BGRX", 0, 1,
            )

            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            print(f"[Capture] Specific window HWND={hwnd} ({width}x{height})")
            self._capture_offset = (left, top)
            return img

        except Exception as exc:
            print(f"[Capture] Specific window capture failed for HWND {hwnd}: {exc}")
            return None

    def parse(self, screenshot: Image.Image) -> ParseResult:
        if not self.omni_available or screenshot is None:
            return ParseResult(elements=[], labeled_img_b64=None, raw={})

        start_time = time.time()
        check_ocr_box = self._omni["check_ocr_box"]
        get_som_labeled_img = self._omni["get_som_labeled_img"]
        yolo_model = self._omni["yolo_model"]
        caption_model_processor = self._omni["caption_model_processor"]
        omni_device = self._omni.get("device")

        img_w, img_h = screenshot.size

        try:
            ocr_bbox_result, _ = check_ocr_box(
                screenshot,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=self.config.omni_use_paddleocr,
            )
            ocr_text = []
            ocr_bbox = []
            if isinstance(ocr_bbox_result, (list, tuple)) and len(ocr_bbox_result) >= 2:
                ocr_text = ocr_bbox_result[0] if isinstance(ocr_bbox_result[0], (list, tuple)) else []
                ocr_bbox = ocr_bbox_result[1] if isinstance(ocr_bbox_result[1], (list, tuple)) else []

            # get_som_labeled_img crashes when passed empty lists — pass None instead
            # so OmniParser's internals skip OCR processing cleanly.
            ocr_text_arg = ocr_text if ocr_text else None
            ocr_bbox_arg = ocr_bbox if ocr_bbox else None

            if not ocr_text:
                print("no ocr bbox!!!")

            box_ratio = img_w / 3200
            draw_bbox_config = {
                "text_scale": 0.8 * box_ratio,
                "text_thickness": max(int(2 * box_ratio), 1),
                "text_padding": max(int(3 * box_ratio), 1),
                "thickness": max(int(3 * box_ratio), 1),
            }

            labeled_b64, label_coordinates, parsed_content_list = get_som_labeled_img(
                screenshot,
                yolo_model,
                BOX_TRESHOLD=self.config.omni_box_thresh,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox_arg,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=ocr_text_arg,
                iou_threshold=self.config.omni_iou_thresh,
                imgsz=self.config.omni_img_sz,
                device=omni_device,
            )

            normalized = self._normalize_parsed_content(parsed_content_list)
            elements = self._build_elements(
                normalized,
                img_w,
                img_h,
                self._capture_offset,
            )
            elements.sort(key=lambda item: (1 if item.interactive else 0, item.score), reverse=True)

            total_time = time.time() - start_time
            print(f"Found {len(elements)} content-based elements in {total_time:.1f}s")

            raw_dump = {
                "label_coordinates": label_coordinates,
                "parsed_content_list": normalized,
            }

            return ParseResult(elements=elements, labeled_img_b64=labeled_b64, raw=raw_dump)
        except Exception as exc:
            print(f"OmniParser error: {exc}")
            traceback.print_exc()
            return ParseResult(elements=[], labeled_img_b64=None, raw={})

    @staticmethod
    def _normalize_parsed_content(parsed_content_list: Any) -> List[Dict[str, Any]]:
        if isinstance(parsed_content_list, list):
            return parsed_content_list
        if isinstance(parsed_content_list, dict):
            return [value for _, value in sorted(parsed_content_list.items(), key=lambda pair: str(pair[0]))]
        return []

    def _build_elements(
        self,
        parsed_content_list: List[Dict[str, Any]],
        img_w: int,
        img_h: int,
        capture_offset: Tuple[int, int],
    ) -> List[DetectedElement]:
        offset_x, offset_y = capture_offset
        seen_names = set()
        elements: List[DetectedElement] = []

        for item in parsed_content_list:
            try:
                if not isinstance(item, dict):
                    continue

                interactive = bool(item.get("interactivity", False))
                content = safe_text(item.get("content", ""))
                element_type = safe_text(item.get("type", ""))
                if not content:
                    continue

                name = " ".join(content.replace("PC ", "").split())
                key = name.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)

                bbox = item.get("bbox", None)
                xyxy = bbox_ratio_xyxy_to_pixels(bbox, img_w, img_h) if isinstance(bbox, list) else None
                if not xyxy:
                    continue

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                elements.append(
                    DetectedElement(
                        name=name,
                        bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                        center=(cx + offset_x, cy + offset_y),
                        interactive=interactive,
                        element_type=element_type,
                        source=item.get("source"),
                        score=self._score(name, interactive, element_type),
                    )
                )
            except Exception:
                continue

        return elements

    @staticmethod
    def _score(name: str, interactive: bool, element_type: str) -> int:
        lowered = name.lower()
        score = 0

        if interactive:
            score += 50

        # Prioritize likely user targets on desktop surfaces.
        if element_type.lower() == "folder":
            score += 40

        if element_type.lower() in ("file", "document", "video"):
            score += 30

        strong_keywords = [
            "open", "new", "delete", "rename", "copy", "paste", "cut", "share",
            "next", "close", "search", "sort", "view",
        ]
        if any(keyword in lowered for keyword in strong_keywords):
            score += 20

        toolbar_keywords = [
            "sort", "view", "copy", "paste", "cut",
            "delete", "rename", "share", "new",
        ]
        if any(keyword in lowered for keyword in toolbar_keywords):
            score -= 25

        if "m0,0" in lowered or "l9,0" in lowered:
            score -= 50

        name_len = len(lowered)
        if 2 <= name_len <= 18:
            score += 10
        elif name_len <= 30:
            score += 5
        else:
            score -= 5

        if element_type.lower() == "icon":
            score += 25

        return score