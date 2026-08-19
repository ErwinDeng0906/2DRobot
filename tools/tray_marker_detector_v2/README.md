# Tray Marker Detector V2

> 分层合并说明：本目录保留为旧版回归基线。新的毫米坐标、槽位投影、
> 硅片质量和 fail-closed 融合入口见
> `src/scara/vision/tray_vision_fusion.py`、`tools/analyze_layered_tray.py` 和
> `docs/tray_marker_layered_integration.md`。`tray_marker_layout.json` 中新增的
> `metric_slot_transform=rot270` 固定了旧图像行列与当前 P00...P55 的关系。

This folder keeps the tray marker tool developed for the 6x6 wafer tray.
It is intentionally kept as a standalone tool first, so it can be added to the
main 2DRobot repository without changing the SCARA control flow.

## Files

- `tray_marker_detector_v2.py`: core vision logic. It detects ArUco markers,
  estimates the tray/grid angle from marker centers, maps marker IDs to tray
  slots, checks slot states, handles occlusion logic, and detects purple wafers.
- `tray_marker_ui_v2.py`: Tk UI for live camera preview and offline image
  analysis.
- `tray_marker_layout.json`: saved 6x6 tray slot layout. This records which
  marker ID belongs to each slot.

## Run The UI

From the repository root:

```bash
cd "/Users/chenge/Desktop/二维机器人/2DRobot"
python3 tools/tray_marker_detector_v2/tray_marker_ui_v2.py
```

Useful startup options:

```bash
python3 tools/tray_marker_detector_v2/tray_marker_ui_v2.py \
  --layout tools/tray_marker_detector_v2/tray_marker_layout.json \
  --camera 0
```

In the UI:

1. Select the camera index.
2. Click `Start Camera` for live detection, or `Open Image` for a saved image.
3. Use `Load Layout` to load `tray_marker_layout.json`.
4. Use `Save JSON` or `Save PNG` to export the current analysis result.

## Run One Image From Command Line

```bash
python3 tools/tray_marker_detector_v2/tray_marker_detector_v2.py \
  --image "/path/to/image.png" \
  --layout tools/tray_marker_detector_v2/tray_marker_layout.json \
  --save-json tray_marker_analysis.json \
  --annotate tray_marker_annotated.png
```

To create or refresh a layout from an empty tray image:

```bash
python3 tools/tray_marker_detector_v2/tray_marker_detector_v2.py \
  --image "/path/to/empty_tray.png" \
  --save-layout tools/tray_marker_detector_v2/tray_marker_layout.json \
  --annotate empty_tray_annotated.png
```

## Run Live Detection From Command Line

```bash
python3 tools/tray_marker_detector_v2/tray_marker_detector_v2.py \
  --camera 0 \
  --layout tools/tray_marker_detector_v2/tray_marker_layout.json
```

Press `q` or `Esc` in the OpenCV preview window to stop.

## Dependencies

The tool needs:

- `opencv-contrib-python` or another OpenCV build with `cv2.aruco`
- `numpy`
- `Pillow`

If ArUco detection fails with `cv2.aruco` missing, install the contrib OpenCV
package instead of plain `opencv-python`.

## Integration Notes

For future integration into `src/scara/vision`, reuse these functions from
`tray_marker_detector_v2.py`:

- `detect_markers_multiscale`
- `analyze_image`
- `make_layout`
- `load_layout`
- `draw_result`

The detector does not move the SCARA arm. It only returns visual observations:
marker IDs, marker centers/corners, tray angle, slot occupancy, occlusion state,
wafer candidates, and annotated images. Robot motion should continue to go
through the existing SCARA controller/action worker.
