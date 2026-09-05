# Camera 1: continuous slot geometry and selection (2026-09-04)

> Historical implementation note: the 2 px crop-anchor strategy below was
> superseded on 2026-09-05. See camera1_stationary_validation_0905.md for the
> live-sequence evidence, current implementation and verification limits.

## Scope and evidence

The supplied 18.4-second screen recording switches its displayed metric mode
between PASS and READ-ONLY PLANAR 27 times. This is screen-time evidence, not
552 independent camera measurements. It does not contain a running XY task.

This change isolates that mode switch from slot-image analysis and operator
selection. It does **not** relax the marker count, RANSAC, reprojection, robot
timestamp, fixed-J3, vacuum/DO, or final-J4 safety contracts. It does not connect
the currently unavailable camera-2 near-field observer.

## Implementation

- LiveWaferTransferRuntime enables consistent_slot_geometry. Both metric PASS
  and metric WAIT analyze the current image using the same checked marker-plane
  registration. The original analyzer mode is retained for other callers.
- Every image still needs a newly accepted plane. Small crop-coordinate changes
  (maximum corner displacement across all slots <= 2 image pixels) can retain
  the observation anchor. The comparison is against the fixed anchor, so slow
  accumulated motion cannot evade the bound. A gap over 0.8 seconds, failed
  plane, stream invalidation, or registration reset discards/replaces the
  anchor. Wafer pixels and classifications are never cached by this mechanism.
- Metric PnP remains current and independent. Normal-looking live slots also
  require a current calibrated-crop boundary check before navigation can arm.
  A retained image anchor is not a cached robot pose or motion authorization.
- Clicking hit-tests the exact displayed frame's slot polygons and records its
  stable slot ID. The returned image-plane coordinates are selection references
  only. The UI also provides P00-P55 number selection when image selection is
  unavailable. Empty/uncertain slots can be selected, not automatically used.
- A metric-invalid frame is not_evaluated, not wafer dropout. It neither adds
  positive evidence nor clears history by a dropout count. Existing evidence
  expires after 2 seconds; two normal observations, current metric validity and
  other motion gates are still required. Real unknown/occluded detections retain
  their existing dropout allowance; explicit warning/empty still fail readiness.
- Duplicate/decreasing sequence IDs and decreasing timestamps are discarded
  before replacing state. Distinct IDs may share Windows' coarse clock tick.
- The image and panel display frame IDs. The panel shows the actual pose failure,
  marker IDs, per-marker RMS and RANSAC count. Save Navigation Report includes
  up to 180 recent overview diagnostic records, including rejected frames.

## Verification and remaining acceptance

Run the targeted tests without hardware:

```powershell
conda run --no-capture-output -n scara_cvdev python -B -m unittest tests.test_transfer_selection_ui tests.test_transfer_observation_continuity tests.test_wafer_transfer_tracking tests.test_wafer_pick_xy_positioning tests.test_wafer_pick_xy_action_integration tests.test_planar_tray_registration tests.test_tray_stage2_stage3 tests.test_tray_marker_layered_integration tests.test_task14_silicon_detection tests.test_silicon_detection_config -q
```

124 tests passed in the installed scara_cvdev environment. The added sequence
tests exercise alternating metric PASS/WAIT, adjacent original frames 013/014,
anchor limits, real movement, stale/old frames, selection and forbidden motion.
These are regression tests, not live acceptance or a replay of raw camera data
from the screen recording.

Restart the application to load the edited source. First verify stationary
selection without arming motion: a click should retain its slot ID through
metric WAIT, while Start XY remains disabled until the current gates pass.
Check that the banner reads SLOT GRID, and use Save Navigation Report if WAIT
persists. Only then perform supervised XY validation with the existing safety
checks and emergency stop available. Genuine metric failures still pause motion;
this patch makes their causes visible rather than ignoring them.
