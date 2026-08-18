#!/usr/bin/env python3
"""
Live UI for the 6x6 tray ArUco detector.

Run:
    python3 tray_marker_ui.py

Suggested collaborator workflow:
    1. Put an empty tray under the overhead camera.
    2. Select the camera and click Start Camera.
    3. Click Save Layout to create that lab's slot ID map.
    4. Load that layout during real runs to detect covered/missing slot markers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from tray_marker_detector_v2 import (
    DEFAULT_COLS,
    DEFAULT_DICT,
    DEFAULT_EDGE_OCCLUSION_MARGIN_RATIO,
    DEFAULT_FLAKE_MIN_AREA,
    DEFAULT_OCCLUSION_BOTTOM_RATIO,
    DEFAULT_ROWS,
    analyze_image,
    camera_backend,
    draw_result,
    load_layout,
    make_layout,
    read_image,
    write_image,
    write_json,
)


RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

COMMON_DICTIONARIES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_6X6_250",
    "DICT_APRILTAG_36h11",
]

COMMON_BACKENDS = [
    "auto",
    "CAP_AVFOUNDATION",
    "CAP_DSHOW",
    "CAP_MSMF",
    "CAP_V4L2",
]

RESOLUTIONS = [
    "native",
    "640x480",
    "1280x720",
    "1920x1080",
]


class TrayMarkerApp(tk.Tk):
    def __init__(self, initial_layout: str | None = None, initial_camera: int | None = None, backend: str = "auto"):
        super().__init__()
        self.title("Tray Marker Detector V2")
        self.geometry("1340x880")
        self.minsize(1080, 720)
        self.configure_fonts()

        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.paused = False
        self.after_id: str | None = None
        self.last_frame: np.ndarray | None = None
        self.last_annotated: np.ndarray | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_error = ""
        self.layout: dict[str, Any] | None = None
        self.layout_path: Path | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self.camera_var = tk.StringVar()
        self.backend_var = tk.StringVar(value=backend)
        self.resolution_var = tk.StringVar(value="native")
        self.dictionary_var = tk.StringVar(value=DEFAULT_DICT)
        self.rows_var = tk.IntVar(value=DEFAULT_ROWS)
        self.cols_var = tk.IntVar(value=DEFAULT_COLS)
        self.occlusion_bottom_var = tk.DoubleVar(value=DEFAULT_OCCLUSION_BOTTOM_RATIO * 100.0)
        self.edge_occlusion_margin_var = tk.DoubleVar(value=DEFAULT_EDGE_OCCLUSION_MARGIN_RATIO)
        self.flake_detect_var = tk.BooleanVar(value=True)
        self.flake_min_area_var = tk.IntVar(value=int(DEFAULT_FLAKE_MIN_AREA))
        self.status_var = tk.StringVar(value="Ready")
        self.layout_var = tk.StringVar(value="No layout loaded")
        self.angle_var = tk.StringVar(value="Angle: --")
        self.marker_var = tk.StringVar(value="Markers: --")
        self.slot_var = tk.StringVar(value="Slots: --")
        self.flake_var = tk.StringVar(value="Chips/flakes: --")
        self.fps_var = tk.StringVar(value="FPS: --")

        self._build_ui()
        self._bind_events()
        self.refresh_cameras(prefer=initial_camera)
        if initial_layout:
            self.load_layout_file(Path(initial_layout))
        else:
            default_layout = Path(__file__).with_name("tray_marker_layout.json")
            if default_layout.exists():
                self.load_layout_file(default_layout)

    def configure_fonts(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=13)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=13)
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(size=13)
        style = ttk.Style(self)
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TCombobox", font=default_font)
        style.configure("TSpinbox", font=default_font)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(10, 8, 10, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        for col in range(12):
            toolbar.columnconfigure(col, weight=0)
        toolbar.columnconfigure(11, weight=1)

        ttk.Label(toolbar, text="Camera").grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.camera_combo = ttk.Combobox(toolbar, textvariable=self.camera_var, width=18, state="readonly")
        self.camera_combo.grid(row=0, column=1, padx=(0, 8), sticky="w")
        ttk.Button(toolbar, text="Refresh", command=self.refresh_cameras).grid(row=0, column=2, padx=(0, 12))

        self.start_button = ttk.Button(toolbar, text="Start Camera", command=self.toggle_camera)
        self.start_button.grid(row=0, column=3, padx=(0, 6), sticky="w")
        self.pause_button = ttk.Button(toolbar, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=4, padx=(0, 12), sticky="w")

        ttk.Label(toolbar, text="Backend").grid(row=1, column=0, padx=(0, 4), pady=(6, 0), sticky="w")
        self.backend_combo = ttk.Combobox(toolbar, textvariable=self.backend_var, width=17, values=COMMON_BACKENDS, state="readonly")
        self.backend_combo.grid(row=1, column=1, padx=(0, 12), pady=(6, 0), sticky="w")

        ttk.Label(toolbar, text="Resolution").grid(row=1, column=2, padx=(0, 4), pady=(6, 0), sticky="w")
        self.resolution_combo = ttk.Combobox(toolbar, textvariable=self.resolution_var, width=12, values=RESOLUTIONS, state="readonly")
        self.resolution_combo.grid(row=1, column=3, padx=(0, 12), pady=(6, 0), sticky="w")

        ttk.Label(toolbar, text="Dictionary").grid(row=1, column=4, padx=(0, 4), pady=(6, 0), sticky="w")
        self.dictionary_combo = ttk.Combobox(toolbar, textvariable=self.dictionary_var, width=18, values=COMMON_DICTIONARIES, state="readonly")
        self.dictionary_combo.grid(row=1, column=5, padx=(0, 12), pady=(6, 0), sticky="w")

        ttk.Label(toolbar, text="Rows").grid(row=1, column=6, padx=(0, 4), pady=(6, 0), sticky="w")
        ttk.Spinbox(toolbar, from_=1, to=12, textvariable=self.rows_var, width=4).grid(row=1, column=7, padx=(0, 8), pady=(6, 0))
        ttk.Label(toolbar, text="Cols").grid(row=1, column=8, padx=(0, 4), pady=(6, 0), sticky="w")
        ttk.Spinbox(toolbar, from_=1, to=12, textvariable=self.cols_var, width=4).grid(row=1, column=9, padx=(0, 12), pady=(6, 0))

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 8))

        left = ttk.Frame(body)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        body.add(left, weight=4)

        self.image_label = tk.Label(
            left,
            anchor="center",
            bg="#202124",
            fg="#d9dde3",
            text="Click Start Camera to open a live camera\nor Open Image to test a still frame.",
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        body.add(right, weight=1)

        metrics = ttk.Frame(right)
        metrics.grid(row=0, column=0, sticky="ew")
        metrics.columnconfigure(0, weight=1)
        ttk.Label(metrics, textvariable=self.angle_var, font=("TkDefaultFont", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(metrics, textvariable=self.marker_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(metrics, textvariable=self.slot_var).grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(metrics, textvariable=self.flake_var).grid(row=3, column=0, sticky="w", pady=(2, 0))
        ttk.Label(metrics, textvariable=self.fps_var).grid(row=4, column=0, sticky="w", pady=(2, 0))

        vision_options = ttk.Frame(metrics)
        vision_options.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        vision_options.columnconfigure(5, weight=1)
        ttk.Checkbutton(vision_options, text="Detect chips/flakes", variable=self.flake_detect_var).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(vision_options, text="Fixed arm mask: on").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(vision_options, text="Min chip area").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Spinbox(vision_options, from_=50, to=100000, increment=50, textvariable=self.flake_min_area_var, width=8).grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(vision_options, text="Edge margin").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(vision_options, from_=0, to=1.5, increment=0.05, textvariable=self.edge_occlusion_margin_var, width=5).grid(row=2, column=1, sticky="w", padx=(6, 12), pady=(6, 0))

        layout_row = ttk.Frame(right)
        layout_row.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        layout_row.columnconfigure(0, weight=1)
        ttk.Label(layout_row, textvariable=self.layout_var, wraplength=300).grid(row=0, column=0, sticky="w")

        action_grid = ttk.Frame(right)
        action_grid.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for col in range(2):
            action_grid.columnconfigure(col, weight=1)
        ttk.Button(action_grid, text="Load Layout", command=self.pick_layout).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(action_grid, text="Clear Layout", command=self.clear_layout).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=3)
        ttk.Button(action_grid, text="Save Layout", command=self.save_layout).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(action_grid, text="Open Image", command=self.open_image).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=3)
        ttk.Button(action_grid, text="Save JSON", command=self.save_json).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(action_grid, text="Save PNG", command=self.save_png).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=3)

        ttk.Label(right, text="Slot Map").grid(row=3, column=0, sticky="w", pady=(14, 4))
        self.report = tk.Text(right, width=42, height=26, wrap="word", state="disabled", font=("Menlo", 13))
        self.report.grid(row=4, column=0, sticky="nsew")

        status = ttk.Frame(self, padding=(10, 0, 10, 8))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

    def _bind_events(self) -> None:
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_cameras())

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def selected_camera_index(self) -> int:
        value = self.camera_var.get()
        if not value:
            return 0
        try:
            return int(value.split()[1])
        except Exception:
            try:
                return int(value)
            except ValueError:
                return 0

    def open_capture(self, index: int) -> cv2.VideoCapture:
        backend = camera_backend(self.backend_var.get())
        if backend:
            cap = cv2.VideoCapture(index, backend)
        else:
            cap = cv2.VideoCapture(index)
        self.apply_resolution(cap)
        return cap

    def apply_resolution(self, cap: cv2.VideoCapture) -> None:
        value = self.resolution_var.get()
        if value == "native" or "x" not in value:
            return
        width_s, height_s = value.split("x", 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width_s))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height_s))

    def refresh_cameras(self, prefer: int | None = None) -> None:
        was_running = self.running
        if was_running:
            self.stop_camera()

        available: list[str] = []
        for index in range(10):
            cap = None
            try:
                cap = self.open_capture(index)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        available.append(f"Index {index}")
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()

        if not available:
            available = ["Index 0"]
        self.camera_combo["values"] = available
        target = f"Index {prefer}" if prefer is not None else self.camera_var.get()
        self.camera_var.set(target if target in available else available[0])
        self.set_status(f"Found {len(available)} camera option(s)")

        if was_running:
            self.start_camera()

    def toggle_camera(self) -> None:
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        index = self.selected_camera_index()
        cap = self.open_capture(index)
        if not cap.isOpened():
            messagebox.showerror("Camera", f"Could not open camera index {index}")
            cap.release()
            return
        self.cap = cap
        self.running = True
        self.paused = False
        self.start_button.configure(text="Stop Camera")
        self.pause_button.configure(text="Pause", state="normal")
        self.set_status(f"Camera {index} started")
        self.schedule_next_frame(1)

    def stop_camera(self) -> None:
        self.running = False
        self.paused = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_button.configure(text="Start Camera")
        self.pause_button.configure(text="Pause", state="disabled")
        self.image_label.configure(
            image="",
            text="Click Start Camera to open a live camera\nor Open Image to test a still frame.",
        )
        self.set_status("Camera stopped")

    def toggle_pause(self) -> None:
        if not self.running:
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.set_status("Paused" if self.paused else "Running")
        if not self.paused:
            self.schedule_next_frame(1)

    def schedule_next_frame(self, delay_ms: int = 30) -> None:
        if self.running and not self.paused:
            self.after_id = self.after(delay_ms, self.update_frame)

    def update_frame(self) -> None:
        self.after_id = None
        if self.cap is None or not self.running or self.paused:
            return
        start = time.monotonic()
        ok, frame = self.cap.read()
        if ok:
            self.process_frame(frame, image_name=f"camera:{self.selected_camera_index()}")
            elapsed = max(time.monotonic() - start, 1e-6)
            self.fps_var.set(f"FPS: {1.0 / elapsed:.1f}")
        else:
            self.set_status("Camera frame read failed")
        self.schedule_next_frame()

    def process_frame(self, frame: np.ndarray, image_name: str | None = None) -> None:
        self.last_frame = frame.copy()
        try:
            dictionary = self.layout.get("dictionary", self.dictionary_var.get()) if self.layout else self.dictionary_var.get()
            result = analyze_image(
                frame,
                image_path=image_name,
                dictionary_name=dictionary,
                rows=int(self.rows_var.get()),
                cols=int(self.cols_var.get()),
                layout=self.layout,
                occlusion_bottom_ratio=float(self.occlusion_bottom_var.get()) / 100.0,
                detect_flakes=bool(self.flake_detect_var.get()),
                flake_min_area=float(self.flake_min_area_var.get()),
                edge_occlusion_margin_ratio=float(self.edge_occlusion_margin_var.get()),
                use_fixed_arm_mask=True,
            )
            annotated = draw_result(frame, result)
            self.last_result = result
            self.last_annotated = annotated
            self.last_error = ""
            self.update_metrics(result)
            self.update_report(result)
            self.set_status("Detection OK")
            self.show_frame(annotated)
        except Exception as exc:
            self.last_result = None
            self.last_error = str(exc)
            self.angle_var.set("Angle: --")
            self.marker_var.set("Markers: --")
            self.slot_var.set("Slots: --")
            self.flake_var.set("Chips/flakes: --")
            annotated = frame.copy()
            cv2.putText(
                annotated,
                str(exc),
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            self.last_annotated = annotated
            self.update_report(None)
            self.set_status(str(exc))
            self.show_frame(annotated)

    def update_metrics(self, result: dict[str, Any]) -> None:
        visible = result["grid"]["visible_slot_count"]
        total = result["grid"]["rows"] * result["grid"]["cols"]
        missing = result["grid"]["missing_slot_count"]
        occluded = result["grid"].get("occluded_slot_count", 0)
        occupied = result["grid"].get("occupied_slot_count", 0)
        unread = result["grid"].get("visible_unread_slot_count", 0)
        warning = result["grid"].get("warning_slot_count", 0)
        abnormal = result["grid"].get("abnormal_slot_count", 0)
        off_grid = result["grid"].get("off_grid_chip_count", 0)
        flakes = result.get("flake_detection", {}).get("count", 0)
        self.angle_var.set(f"Angle: {result['tray_angle_deg']:.3f} deg")
        self.marker_var.set(f"Markers: {result['detected_marker_count']}")
        self.slot_var.set(
            f"Slots: {visible}/{total} visible, {occupied} occupied, "
            f"{unread} unread QR, {warning} warn, {abnormal} bad, {missing} missing, {occluded} occluded"
        )
        self.flake_var.set(f"Chips/flakes: {flakes}, off-grid {off_grid}")

    def update_report(self, result: dict[str, Any] | None) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        if result is None:
            self.report.insert("end", self.last_error or "No result")
            self.report.configure(state="disabled")
            return

        lines = ["Slot marker ID grid:"]
        for row in result["grid"]["slot_id_grid"]:
            lines.append("  " + " ".join(f"{int(x):>3}" if x is not None else "  ." for x in row))
        lines.append("")
        if self.layout is None and any(slot.get("id") is None for slot in result["slots"] if slot["state"] != "visible"):
            lines.append("Note: load a saved empty-tray layout to know covered marker IDs.")
            lines.append("")
        if result.get("occupied_slots"):
            lines.append("Occupied by chip/flake:")
            for slot in result["occupied_slots"]:
                flags = ",".join(slot.get("chip_flags", [])) or "-"
                lines.append(
                    f"  R{slot['row']}C{slot['col']} id={slot['id']} "
                    f"flake={slot['matched_flake_idx']} status={slot.get('chip_status', 'ok')} flags={flags}"
                )
        else:
            lines.append("Occupied by chip/flake: none")
        lines.append("")
        if result.get("visible_unread_slots"):
            lines.append("Visible QR/marker pattern, ID not decoded:")
            for slot in result["visible_unread_slots"]:
                lines.append(f"  R{slot['row']}C{slot['col']} id={slot['id']}")
        else:
            lines.append("Visible QR/marker pattern, ID not decoded: none")
        lines.append("")
        if result.get("warning_slots"):
            lines.append("Warning slots:")
            for slot in result["warning_slots"]:
                flags = ",".join(slot.get("chip_flags", [])) or "-"
                lines.append(f"  R{slot['row']}C{slot['col']} id={slot['id']} {flags}")
        else:
            lines.append("Warning slots: none")
        lines.append("")
        if result.get("abnormal_slots"):
            lines.append("Abnormal/error slots:")
            for slot in result["abnormal_slots"]:
                flags = ",".join(slot.get("chip_flags", [])) or "-"
                lines.append(f"  R{slot['row']}C{slot['col']} id={slot['id']} {flags}")
        else:
            lines.append("Abnormal/error slots: none")
        lines.append("")
        if result.get("off_grid_chips"):
            lines.append("Off-grid chip candidates:")
            for flake in result["off_grid_chips"]:
                nearest = flake.get("nearest_slot")
                flags = ",".join(flake.get("chip_flags", [])) or "-"
                lines.append(f"  #{flake['idx']} nearest={nearest} flags={flags}")
        else:
            lines.append("Off-grid chip candidates: none")
        lines.append("")
        if result.get("extra_chips"):
            lines.append("Extra chips near a slot:")
            for flake in result["extra_chips"]:
                nearest = flake.get("nearest_slot")
                flags = ",".join(flake.get("chip_flags", [])) or "-"
                lines.append(f"  #{flake['idx']} nearest={nearest} flags={flags}")
        else:
            lines.append("Extra chips near a slot: none")
        lines.append("")
        if result["missing_slots"]:
            lines.append("Missing/unexplained slots:")
            for slot in result["missing_slots"]:
                lines.append(f"  R{slot['row']}C{slot['col']} id={slot['id']}")
        else:
            lines.append("Missing/unexplained slots: none")
        lines.append("")
        if result.get("occluded_slots"):
            lines.append("Occluded/unknown slots:")
            for slot in result["occluded_slots"]:
                lines.append(f"  R{slot['row']}C{slot['col']} id={slot['id']} {slot['occlusion_reason']}")
        else:
            lines.append("Occluded/unknown slots: none")
        lines.append("")
        if result.get("flakes"):
            lines.append("Chip/flake candidates:")
            for flake in result["flakes"]:
                cx, cy = flake["center_px"]
                side = flake.get("square_side_px")
                side_text = f" side={side:.0f}" if side is not None else ""
                rel = flake.get("angle_relative_to_tray_deg")
                rel_text = f" rel={rel:+.1f}" if rel is not None else ""
                flags = ",".join(flake.get("chip_flags", [])) or "-"
                lines.append(
                    f"  #{flake['idx']} center=({cx:.0f},{cy:.0f}) "
                    f"area={flake['area']:.0f}{side_text} angle={flake['final_angle_deg']:.1f}{rel_text} "
                    f"status={flake.get('chip_status', 'unassigned')} flags={flags} "
                    f"{flake.get('kind', 'chip')} {flake['angle_source']}"
                )
        else:
            lines.append("Chip/flake candidates: none")
        lines.append("")
        roi = result.get("tray_roi", {})
        if roi.get("enabled"):
            lines.append(f"Tray ROI: {roi.get('source', 'unknown')}")
            lines.append("")
        lines.append("Locator IDs: " + ", ".join(str(x) for x in result["locator_ids"]))
        if result["unknown_markers"]:
            lines.append("")
            lines.append("Unknown markers:")
            for marker in result["unknown_markers"]:
                lines.append(f"  id={marker['id']}")
        self.report.insert("end", "\n".join(lines))
        self.report.configure(state="disabled")

    def show_frame(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        panel_width = max(self.image_label.winfo_width(), 640)
        panel_height = max(self.image_label.winfo_height(), 480)
        image = Image.fromarray(rgb)
        image.thumbnail((panel_width, panel_height), RESAMPLE_LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="")

    def pick_layout(self) -> None:
        path = filedialog.askopenfilename(
            title="Load tray layout",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.load_layout_file(Path(path))

    def load_layout_file(self, path: Path) -> None:
        try:
            self.layout = load_layout(path)
            self.layout_path = path
            self.dictionary_var.set(str(self.layout.get("dictionary", self.dictionary_var.get())))
            self.rows_var.set(int(self.layout.get("rows", self.rows_var.get())))
            self.cols_var.set(int(self.layout.get("cols", self.cols_var.get())))
            self.layout_var.set(f"Layout: {path.name}")
            self.set_status(f"Loaded layout: {path}")
        except Exception as exc:
            messagebox.showerror("Layout", str(exc))

    def clear_layout(self) -> None:
        self.layout = None
        self.layout_path = None
        self.layout_var.set("No layout loaded")
        self.set_status("Layout cleared")

    def save_layout(self) -> None:
        if self.last_result is None:
            messagebox.showwarning("Save Layout", "No detection result is available.")
            return
        missing = int(self.last_result["grid"]["missing_slot_count"])
        occluded = int(self.last_result["grid"].get("occluded_slot_count", 0))
        occupied = int(self.last_result["grid"].get("occupied_slot_count", 0))
        if missing or occluded or occupied:
            ok = messagebox.askyesno(
                "Save Layout",
                f"The current frame has {occupied} occupied, {missing} missing, and {occluded} occluded slot marker(s). Save layout anyway?",
            )
            if not ok:
                return
        default = self.layout_path.name if self.layout_path else "tray_marker_layout.json"
        path = filedialog.asksaveasfilename(
            title="Save tray layout",
            defaultextension=".json",
            initialfile=default,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = make_layout(self.last_result)
        write_json(Path(path), payload)
        self.load_layout_file(Path(path))

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            frame = read_image(Path(path))
            self.process_frame(frame, image_name=path)
        except Exception as exc:
            messagebox.showerror("Open Image", str(exc))

    def save_json(self) -> None:
        if self.last_result is None:
            messagebox.showwarning("Save JSON", "No detection result is available.")
            return
        path = filedialog.asksaveasfilename(
            title="Save analysis JSON",
            defaultextension=".json",
            initialfile="tray_marker_analysis.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        write_json(Path(path), self.last_result)
        self.set_status(f"Saved JSON: {path}")

    def save_png(self) -> None:
        if self.last_annotated is None:
            messagebox.showwarning("Save PNG", "No annotated frame is available.")
            return
        path = filedialog.asksaveasfilename(
            title="Save annotated PNG",
            defaultextension=".png",
            initialfile="tray_marker_annotated.png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        write_image(Path(path), self.last_annotated)
        self.set_status(f"Saved PNG: {path}")

    def on_close(self) -> None:
        self.stop_camera()
        self.destroy()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", help="Load an existing tray layout JSON on startup.")
    parser.add_argument("--camera", type=int, help="Preferred camera index on startup.")
    parser.add_argument("--backend", default="auto", choices=COMMON_BACKENDS, help="OpenCV camera backend.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    app = TrayMarkerApp(initial_layout=args.layout, initial_camera=args.camera, backend=args.backend)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
