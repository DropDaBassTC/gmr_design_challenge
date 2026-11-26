# Sensor YAML configuration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a root YAML file that controls simulation route speed and the existing simulated scanner settings, and bind it into the Compose simulation service.

**Architecture:** `MujocoBridge` loads a safe YAML mapping and passes shared scanner settings to every `ScannerNode`, with route speed and crane lane spacing passed to `ScanSequence`. Direct bridge callers default to root `sensors.yml`; explicit paths support focused tests. Compose mounts the same file read-only at `/ros_ws/sensors.yml`.

**Tech Stack:** Python, PyYAML, ROS 2 Jazzy, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-25-sensor-configuration-design.md`

## Global Constraints
- Preserve current defaults: route speed `0.5`, crane lane step `8.0`, scanner samples `180 × 8`, range `10.0 m`, simulated distance-noise standard deviation `0.01 m`, publication rate `10.0 Hz`.
- Do not add or imply an exposure parameter; exposure is not modeled and no verified hardware driver interface is in this repository.
- Mount the root configuration read-only into the simulation service only; do not change hardware launch behavior.

## Review Focus
- Missing/malformed config: fail clearly with path/cause.
- Empty/non-mapping YAML, missing keys and wrong types: reject instead of partially defaulting.
- Unsafe YAML tags and duplicate keys: reject.
- Speeds outside supported actuator bounds: existing actuator clamping remains authoritative.
- Compose mount resolves `./sensors.yml` to `/ros_ws/sensors.yml:ro` for `sim_bridge`, not the hardware services.

---

### Task 1: Load YAML config and bound raw captures to scan passes

**Files:** Create `sensors.yml`; modify `Dockerfile`, `compose.yml`, `src/sim/mujoco_bridge.py`, `src/sim/scan_sequence.py`, `src/scanning/scanning/scanner_node.py`, `src/sim/README.md`, `src/scanning/README.md`; test `src/sim/test/test_bridge_nodes.py`, `src/sim/test/test_scan_sequence.py`, `src/scanning/test/test_pointcloud_stitching.py`.

**Interfaces:** `MujocoBridge(..., config_path=Path("sensors.yml"))` loads YAML and passes scanner settings (`horizontal_samples`, `vertical_channels`, `max_range`, `noise_std`, `scan_hz`) to every scanner and route settings (`speed`, `crane_y_step`) to `ScanSequence`. Add recorder capture-boundary methods used by `ScanSequence` and manual service activation/deactivation; keep direct constructor defaults.

- [ ] Add `test_yaml_config_controls_scan_rate_density_range_noise_and_route_speed`; verify YAML overrides reach all five scanners and route settings.
- [ ] Run the focused test and confirm it fails because the config interface is absent.
- [ ] Create root `sensors.yml` with current defaults and a comment that exposure is unsupported/not modeled.
- [ ] Install Ubuntu `python3-yaml` in the container; load with `yaml.safe_load`; validate required mapping and value types, reject duplicate keys and report malformed/missing-file errors clearly.
- [ ] Pass YAML values to scanner and sequence constructors; retain direct-call defaults.
- [ ] Add a failing recorder/bridge test showing that multiple scanner frames during one active rail traversal produce exactly one raw PLY containing the combined point count, and that deactivation/next rail closes one capture and starts a separate one.
- [ ] Implement a capture lifecycle so each route scan segment accumulates only its own points; close the floor-rail capture after its outbound scan and before return/reposition, then open the next rail's capture. Keep crane movement/reposition frames out of captures; if crane passes are captured, write one file per complete X pass. Preserve manual service behavior by mapping one activation/deactivation interval to one PLY.
- [ ] Verify PLY header vertex count and payload equal the points from the included frames, exclude frames outside the capture boundaries, and a full four-rail route creates four floor capture files rather than one per published frame.
- [ ] Bind mount `sensors.yml` read-only into the simulation service only.
- [ ] Re-run focused tests; render Compose config and verify mount target/read-only mode.
- [ ] Run relevant bridge, recorder, and simulation tests; report outcomes.

### Task 2: Document supported controls

**Files:** Modify `src/sim/README.md`, `src/scanning/README.md`.

- [ ] Document config path, all keys/units, and that route speed is limited by MuJoCo actuator bounds.
- [ ] State optical exposure is not simulated/configurable here and requires a confirmed hardware driver/SKU interface.
- [ ] Check field names and units against the implementation.