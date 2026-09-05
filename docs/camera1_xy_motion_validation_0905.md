# Camera 1 XY motion interruption fix — 2026-09-05

## Observed failure mode

The image/slot overlay could remain visually stable while the XY workflow still
paused for repeated four-second observation windows.  The older successful run
`Trajectory Photos/260901225515/points.json` completed, but four of its fourteen
runtime windows produced no motion because two accepted camera-1 frames were not
available in time.

The movement path reused the source-selection gates after arming.  That made a
moving forearm camera repeatedly prove that the same wafer was still visible,
even though the arm can legitimately occlude it.  At the same time the live
runtime rebuilt `W<-T` from a moving camera and an asynchronously sampled robot
state, so small timing differences could appear as tray motion and block the
locked session.

## Implemented state split

- Acquisition remains strict: before ARM, the selected source must be
  `occupied`, the two-frame source consensus must pass, the current metric pose
  and slot geometry must pass, `W<-T` must exist, and camera/robot timestamps
  must be synchronized.
- After ARM, the selected slot and `W<-T` are immutable.  New frames cannot
  overwrite the world target.
- Every XY observation window still requires two fresh, distinct, ordered
  camera frames, a current hard-passed marker pose, a valid locked registration,
  and fresh synchronized robot state.
- `unknown`, `out_of_view`, and `occluded` are treated as temporary visibility
  loss for the already locked source.  Explicit `empty`, `warning`, `stacked`,
  or outside-slot results still reject the window.
- Each current `C<-T` pose is recombined with the ActionWorker request state to
  check the tray against the locked registration.  This current estimate is
  monitoring evidence only and never replaces the target.
- ActionWorker remains the only hardware owner and still re-reads controller
  state before each command.  J3, Rz, T1 mode, speed, alarms, E-stop, workspace,
  per-step, cumulative-path, kinematic, and final-arrival checks are unchanged.

## Verification

- 61 directly related tests passed, covering source selection, frame ordering,
  target locking, current tray-pose monitoring, repeated source occlusion,
  complete XY convergence, final J4 alignment, and ActionWorker's independent
  safety audit.
- A complete controller/kinematics replay deliberately marked the source
  `occluded` in every movement window.  It converged monotonically to the locked
  slot, then completed the J4-only tray orientation step.
- The opposite case is covered: an explicit source `empty` result fails the XY
  motion gate.
- Full repository discovery ran 368 tests: 356 passed and 11 skipped.  The one
  failure is the pre-existing `test_full_tray_positioning` string-search false
  positive caused by the metadata key `camera_capture_settings`; it is outside
  this change.
- The desktop application was closed normally and restarted from the modified
  checkout with `scara_cvdev`; the main window loaded without a startup error.

No physical robot movement was issued during automated verification.
