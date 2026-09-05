"""Replay lossless camera sequences through the real, controller-free runtime.

Input is capture.json + PNGs, or a directory of ordered raw_task14 images.
No camera/robot connection, motor command, or calibration write is possible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--module-root', type=Path, help='optional backed-up scara package parent')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--simulate-stationary-robot', action='store_true',
        help='offline readiness test ONLY; inject synthetic stationary robot timestamps, never connect hardware')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    if args.module_root:
        sys.path.insert(0, str(args.module_root))
    from scara.vision.wafer_transfer_runtime import LiveWaferTransferRuntime
    runtime = LiveWaferTransferRuntime(root)
    if args.simulate_stationary_robot:
        runtime.session.select_source('P22')
    manifest = args.input / 'capture.json'
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding='utf-8'))['frames']
    else:
        entries = [{'file': p.name, 'elapsed_s': i * .1} for i, p in enumerate(sorted(args.input.glob('raw_task14*.jpg')))]
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError('no input frames')
    duration = entries[-1]['elapsed_s'] - entries[0]['elapsed_s']
    replay_fps = (len(entries) - 1) / duration if duration > 0 else 10.
    args.output.mkdir(parents=True, exist_ok=False)
    records, previous, writer = [], {}, None
    try:
        for index, entry in enumerate(entries):
            image = cv2.imread(str(args.input / entry['file']))
            started = time.perf_counter()
            timestamp = 1000 + entry['elapsed_s']
            with patch('time.monotonic', return_value=timestamp):
                robot_state = ({'captured_monotonic_s': timestamp,
                    'joints': [32.92, 63.84, -27.01, -7.90],
                    'pose': [168.24, 296.08, -27.01, 180., 0., -1.13]}
                    if args.simulate_stationary_robot else None)
                frame = runtime.process_camera1(image, frame_sequence=index + 1,
                    captured_monotonic_s=timestamp, robot_state=robot_state)
                # Selection is intent only; exercise the displayed polygon on
                # every distinct frame, including metric WAIT, without arming.
                clicked = None
                target = next((s for s in frame.result.slots if s.projection.slot_key == 'P22'), None)
                if target is not None:
                    center = np.mean(target.projection.polygon_px, axis=0)
                    try:
                        runtime.select_pixel(center, role='source', displayed_frame=frame)
                        clicked = runtime.session.source_slot
                    except ValueError as exc:
                        clicked = str(exc)
            elapsed = time.perf_counter() - started
            result = frame.result
            slots = {}
            for slot in result.slots:
                key = slot.projection.slot_key
                wafer = slot.wafer
                polygon = np.asarray(slot.projection.polygon_px)
                box = np.asarray(slot.wafer_box_image_px)
                prior = previous.get(key)
                slots[key] = {'state': slot.decision.state.value,
                    'yaw': None if wafer is None else wafer.yaw_relative_to_tray_deg,
                    'flags': [] if wafer is None else wafer.flags,
                    'metric_check': result.projection_diagnostics.get('metric_slot_checks', {}).get(key),
                    'grid_step_px': None if prior is None else float(np.max(np.linalg.norm(polygon - prior[0], axis=1))),
                    'box_step_px': None if prior is None or box.shape != (4, 2) or prior[1].shape != (4, 2) else min(
                        float(np.max(np.linalg.norm(box - np.roll(prior[1], shift, axis=0), axis=1))) for shift in range(4))}
                previous[key] = (polygon, box)
            records.append({'frame': index + 1, 'input': entry['file'], 'elapsed_s': entry['elapsed_s'],
                'processing_s': elapsed, 'metric_passed': result.quality_passed and result.coordinate_mapping_allowed,
                'reason': result.failure_reason, 'pose': result.pose.to_json(),
                'geometry': result.projection_diagnostics, 'slots': slots, 'selected_by_click': clicked,
                'motion_authorized': frame.session_snapshot['tracking_ready'],
                'selection_gates': frame.session_snapshot.get('selection_gates'),
                'registration_error': frame.session_snapshot.get('registration_error')})
            if args.video:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output / 'replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), replay_fps, (image.shape[1], image.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('video writer unavailable')
                writer.write(frame.annotated_bgr)
            if index % 20 == 0:
                cv2.imwrite(str(args.output / f'frame_{index:04d}.png'), frame.annotated_bgr)
                print(json.dumps({'processed': index + 1, 'total': len(entries)}, ensure_ascii=True), flush=True)
    finally:
        if writer is not None:
            writer.release()
    summary = {'synthetic_robot_state': args.simulate_stationary_robot,
        'frames': len(records), 'metric_pass_frames': sum(r['metric_passed'] for r in records),
        'metric_switches': sum(a['metric_passed'] != b['metric_passed'] for a, b in zip(records, records[1:])),
        'clicked_P22_frames': sum(r['selected_by_click'] == 'P22' for r in records),
        'motion_authorized_frames': sum(r['motion_authorized'] for r in records),
        'processing_ms_p50_p95': np.percentile([r['processing_s'] * 1000 for r in records], [50, 95]).tolist(),
        'failures': dict(Counter(r['reason'] for r in records if not r['metric_passed'])), 'slots': {}}
    for key in sorted({key for r in records for key in r['slots']}):
        rows = [r['slots'][key] for r in records if key in r['slots']]
        summary['slots'][key] = {'states': dict(Counter(r['state'] for r in rows)),
            'switches': sum(a['state'] != b['state'] for a, b in zip(rows, rows[1:])),
            'metric_pass_frames': sum(bool((r['metric_check'] or {}).get('passed')) for r in rows),
            'grid_step_p95_px': np.percentile([r['grid_step_px'] for r in rows if r['grid_step_px'] is not None] or [0], 95),
            'box_step_p95_px': np.percentile([r['box_step_px'] for r in rows if r['box_step_px'] is not None] or [0], 95)}
    (args.output / 'frames.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'slots'}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
