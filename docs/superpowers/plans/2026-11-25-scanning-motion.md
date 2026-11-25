# MuJoCo Scanning and Motion Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ROS 2 Jazzy scanning and motion packages that raycast the MuJoCo hangar, gate FARO-style scan publication, and move the simulated sensor carriage.

**Architecture:** `mujoco_bridge.py` remains the sole owner of MuJoCo model/data and physics. It instantiates scanning and motion ROS nodes with that shared state in one process and one executor, avoiding duplicate worlds and concurrent state mutation. Scanning owns raycasts and `PointCloud2`; motion owns the track-speed command interface and carriage actuator.

**Tech Stack:** ROS 2 Jazzy, Python `rclpy`, MuJoCo, NumPy, `sensor_msgs`, `std_srvs`, `std_msgs`, colcon.

**Spec:** `docs/superpowers/specs/2026-11-25-scanning-motion-design.md`

## Global Constraints

- Keep physics/model/data in the existing bridge and share the live state with both packages.
- Model the provided `src/sim/models/Hangar_model.stl`; hangar floor is 80 m × 72 m, centered at origin, x length, y width, z up.
- The scanner uses 360° horizontal × 300° vertical FARO Focus family dome; exact Core SKU/settings remain unconfirmed.
- Do not introduce aircraft ground truth or use an aircraft mesh in the scan pipeline.
- Keep ROS 2 Jazzy compatibility and existing `points`, `joint_states`, and `motor_cmd` conventions where applicable.
- Preserve pre-existing uncommitted Docker, Compose, and simulation work.

## Review Focus

- Invalid or un-loadable hangar STL units/orientation: test mesh loading and bounds; fail clearly rather than silently mis-scale.
- Concave hollow hangar collision behavior: test that rays see interior surfaces and do not rely on a convex hull as the interior shell.
- Scan inactive/active transitions: test no cloud while inactive and cloud publication while active.
- Motion command limits and stop: test track speed clamps to configured range and zero command stops carriage.
- Shared state/callback ordering: test motion changes the pose used by the scanner without creating a second model/data instance.

---

### Task 1: Create ROS 2 package scaffolds and MuJoCo hangar scene

**Files:**
- Create: `src/scanning/package.xml`, `setup.py`, `setup.cfg`, `resource/scanning`, `scanning/__init__.py`
- Create: `src/motion/package.xml`, `setup.py`, `setup.cfg`, `resource/motion`, `motion/__init__.py`
- Modify: `src/sim/models/scene.xml`
- Modify: `Dockerfile` only if the current build does not include the STL at its referenced runtime path
- Test: add focused scene/package tests under each package as required by colcon

**Interfaces:**
- Produces two discoverable `ament_python` packages named `scanning` and `motion`.
- MuJoCo model contains a limited x-axis sensor carriage/slide joint, a velocity actuator for that joint, and the supplied hangar mesh.

- [ ] Test MuJoCo scene load and assert the hangar asset resolves, carriage joint/actuator exist, and hangar mesh bounds correspond to the brief's 80 m × 72 m floor/origin/axes after any explicit mesh transform.
- [ ] Verify the hollow hangar is represented as interior ray/collision surfaces, not an invalid convex-only shell. If MuJoCo's native mesh collision cannot preserve the interior, use the minimum practical collision representation and keep the STL as the visible/map asset.
- [ ] Add minimal package metadata declaring actual ROS runtime dependencies and console entry points; no extra dependencies beyond existing ROS/MuJoCo/NumPy stack.
- [ ] Run `colcon list` and the scene test; require both packages discovered and valid MuJoCo model load.

### Task 2: Implement scanner node in `src/scanning`

**Files:**
- Create: `src/scanning/scanning/scanner_node.py`
- Modify: package entry point and tests

**Interfaces:**
- `ScannerNode(model, data)` accepts the bridge-owned MuJoCo model/data.
- `std_srvs/srv/SetBool` service `set_scanning` activates/deactivates raycasting.
- Publish `sensor_msgs/msg/PointCloud2` on `points`; use the existing `lidar` frame/site and preserve XYZ float32 layout.

- [ ] Test activation state transitions and ensure inactive scans do not publish.
- [ ] Implement shared-state raycasting and cloud publication, using 360° × 300° dome and the current bridge's remaining temporary sampling/range/noise defaults only where the design has no verified SKU values; label these simulation assumptions in node parameters/docs.
- [ ] Validate generated ray directions and cloud layout, including empty returns and points within maximum range.
- [ ] Run scanner tests and Python compilation.

### Task 3: Implement motion node in `src/motion`

**Files:**
- Create: `src/motion/motion/motion_node.py`
- Modify: package entry point and tests

**Interfaces:**
- `MotionNode(model, data)` accepts the same bridge-owned MuJoCo model/data.
- Subscribe to `std_msgs/msg/Float64` `track_speed_cmd`; command velocity actuator control, clamp to its configured limits, and accept zero as stop.

- [ ] Test speed commands at negative/positive limits, over-limit inputs, and zero; assert bounded actuator control and stop behavior.
- [ ] Implement minimal one-axis simulated carriage speed control; document as a simulation shortcut, not the four-rail hardware arrangement.
- [ ] Run motion tests and Python compilation.

### Task 4: Integrate packages with the bridge and verify end-to-end

**Files:**
- Modify: `src/sim/mujoco_bridge.py`
- Modify: `Dockerfile` or `compose.yml` only when required by runtime package/asset paths
- Modify: `src/scanning/README.md`, `src/motion/README.md`, and `src/sim/README.md`

**Interfaces:**
- Bridge creates exactly one `MjModel`/`MjData`, constructs both package nodes with them, and adds bridge/scanner/motion nodes to one single-threaded executor.
- Bridge retains physics stepping, joint-state publication, and existing `motor_cmd` pan behavior; scanner replaces bridge raycasting and motion controls the track actuator.

- [ ] Remove the temporary scan implementation from the bridge and construct scanner/motion nodes against its model/data.
- [ ] Verify the scanner service turns `points` publication off/on, motion commands change carriage position, and subsequent clouds use the changed sensor pose.
- [ ] Run package tests, `colcon build --symlink-install`, and a bounded simulation smoke test with ROS topics/service calls in the Jazzy container when available.
- [ ] Inspect the final diff and working-tree status to ensure unrelated pre-existing changes remain untouched; do not commit.
