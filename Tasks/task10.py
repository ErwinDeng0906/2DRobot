"""Task 10 / Stage 7A: one supervised camera-guided XY correction at P22.

This imported action intentionally contains only the basic route operations:
wait, record state, capture camera 1, request one
runtime-approved joint move, then record/capture the result.  Stage-3 vision,
Jacobian control equations, fixed-Rz IK planning, safety-gate evaluation, the
operator dialog, and JSON analysis live in reusable ``scara`` modules.

The runtime move is default-deny.  Closing or declining the Stage7A dialog
issues no movement and the five ``after`` images are still saved for audit.
There is no Z command, DO command, vacuum command, or automatic repetition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scara.pipeline.xy_correction_planner import load_stage7a_motion_contract


ACTION_API_VERSION = 1

CAMERA_SOURCE = 1
TARGET_NAME = "P22"
FRAMES_PER_PHASE = 5
INITIAL_SETTLE_SECONDS = 0.8
POST_MOVE_SETTLE_SECONDS = 1.0
FRAME_INTERVAL_SECONDS = 0.15

# Engine-enforced Stage7A limits are repeated explicitly in the task record so
# an operator can audit the imported action before it runs.  The ActionWorker
# refuses any imported value that is less restrictive than its hard ceiling.
DOMAIN_MARGIN_MM = 0.20
J3_TOLERANCE_MM = 0.15
# The current pose may be slightly off after an ordinary World-XY jog.  One
# explicit J4-only precompensation brings it back to the Task9 absolute Rz
# before the XY correction.  Endpoint and in-motion limits remain separate.
CURRENT_RZ_TOLERANCE_DEG = 0.20
TARGET_RZ_TOLERANCE_DEG = 0.15
MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG = 0.30
PRECOMPENSATE_RZ = True
MAXIMUM_STEP_NORM_MM = 0.25
MAXIMUM_STEP_AXIS_MM = 0.25
MAXIMUM_STATE_DRIFT_XY_MM = 0.05
MAXIMUM_STATE_DRIFT_JOINT = 0.05
MAXIMUM_SEQUENTIAL_TRANSIENT_MM = 0.50
# ``goto_joints_sync`` also uses this value as its command deadband.  Stage7A
# joint changes can be smaller than 0.05 deg, so 0.01 is required to avoid a
# false "arrived" result in which one or more small axes were never commanded.
MOVE_TOLERANCE_DEG_OR_MM = 0.01
PROPOSAL_MAX_AGE_SECONDS = 20.0
FK_POSE_XY_TOLERANCE_MM = 0.20


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _record_name(phase: str, frame_index: int) -> str:
    return (
        f"TASK10|target={TARGET_NAME}|phase={phase}|"
        f"frame={frame_index:02d}/{FRAMES_PER_PHASE:02d}"
    )


def _append_capture_burst(actions: list[dict[str, Any]], phase: str) -> None:
    for frame_index in range(1, FRAMES_PER_PHASE + 1):
        actions.append(
            {
                "type": "record_point",
                "name": _record_name(phase, frame_index),
            }
        )
        actions.append({"type": "capture", "source": CAMERA_SOURCE})
        if frame_index < FRAMES_PER_PHASE:
            actions.append({"type": "wait", "seconds": FRAME_INTERVAL_SECONDS})


def build_action() -> dict[str, Any]:
    project_root = _project_root()
    contract = load_stage7a_motion_contract(project_root)

    actions: list[dict[str, Any]] = [
        {"type": "wait", "seconds": INITIAL_SETTLE_SECONDS},
    ]
    _append_capture_burst(actions, "before")
    actions.append(
        {
            "type": "runtime_move_joints",
            "name": "Stage7A人工确认的单步XY视觉修正",
            # This key is part of the runtime/worker safety contract.  Keep it
            # identical to Stage7ASingleStepRuntime and the audit record.
            "request_key": "stage7a_p22_single_step",
            "target_name": TARGET_NAME,
            "calibration_sha256": contract["stage5_sha256"],
            "anchor_robot_xy_mm": contract["anchor_robot_xy_mm"],
            "local_extent_mm": contract["local_extent_mm"],
            "domain_margin_mm": DOMAIN_MARGIN_MM,
            "required_j3_mm": contract["required_j3_mm"],
            "required_rz_deg": contract["required_rz_deg"],
            "max_xy_step_norm_mm": MAXIMUM_STEP_NORM_MM,
            "max_xy_axis_mm": MAXIMUM_STEP_AXIS_MM,
            "j3_tolerance_mm": J3_TOLERANCE_MM,
            "rz_tolerance_deg": CURRENT_RZ_TOLERANCE_DEG,
            "target_rz_tolerance_deg": TARGET_RZ_TOLERANCE_DEG,
            "max_sequential_transient_rz_deg": (
                MAXIMUM_SEQUENTIAL_TRANSIENT_RZ_DEG
            ),
            "precompensate_rz": PRECOMPENSATE_RZ,
            "max_state_drift_xy_mm": MAXIMUM_STATE_DRIFT_XY_MM,
            "max_state_drift_joint": MAXIMUM_STATE_DRIFT_JOINT,
            "max_sequential_transient_xy_mm": MAXIMUM_SEQUENTIAL_TRANSIENT_MM,
            "move_tolerance": MOVE_TOLERANCE_DEG_OR_MM,
            "proposal_max_age_s": PROPOSAL_MAX_AGE_SECONDS,
            "fk_pose_xy_tolerance_mm": FK_POSE_XY_TOLERANCE_MM,
        }
    )
    actions.append({"type": "wait", "seconds": POST_MOVE_SETTLE_SECONDS})
    _append_capture_burst(actions, "after")

    return {
        "name": "task10 Stage7A P22受监督单步XY视觉修正",
        "description": (
            "从P22附近当前位置先拍5张相机1照片，仅在已标定的局部域、"
            "固定观察高度和Rz安全门通过后，先做一次明确记录的J4-only Rz预补偿，"
            "再计算并执行一次最大0.25mm的"
            "XY修正并显示人工确认弹窗；只有全部安全门通过且人员明确确认才运动。"
            "随后再拍5张验证照片。关闭或拒绝弹窗不会运动；不改变Z、不控制DO/真空。"
        ),
        "camera_model": {
            "offset_mm": 0.0,
            "angle_reference": "world_negative_y",
            "positive_rotation": "counter_clockwise_from_above",
        },
        "actions": actions,
    }


def create_task_runtime(output_dir: Path, parent=None):
    from scara.vision.stage7a_runtime import create_stage7a_runtime

    return create_stage7a_runtime(
        output_dir=output_dir,
        project_root=_project_root(),
        parent=parent,
    )


if __name__ == "__main__":
    preview = build_action()
    action_types = [item["type"] for item in preview["actions"]]
    print(preview["name"])
    print(
        f"points={action_types.count('record_point')}; "
        f"photos={action_types.count('capture')}; "
        f"supervised_moves={action_types.count('runtime_move_joints')}"
    )
    print("camera=1; Z motion=none; DO/vacuum=none; automatic loop=none")
