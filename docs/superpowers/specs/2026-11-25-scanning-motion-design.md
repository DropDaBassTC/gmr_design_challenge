# Scanning and motion packages design

## Intent

Add small ROS 2 Python packages under `src/scanning` and `src/motion` for the MuJoCo simulation. Scanning should activate/deactivate point-cloud gathering, raycast the shared simulated world, and publish `PointCloud2`. Motion should command the sensor carriage speed along a simulated track. This gives downstream nodes scanner and motion interfaces without introducing a physical Faro dependency.

## Goals

- Build and run both packages in the existing ROS 2 Jazzy/colcon workspace.
- Keep MuJoCo physics, model, and data owned by the existing simulation bridge, with scanning and motion operating against the same live state rather than independent simulations.
- Include the provided `src/sim/models/Hangar_model.stl` as the MuJoCo hangar environment, honoring the project design's hangar dimensions, origin, axes, and collision constraints.
- Move raycasting and point-cloud publication out of the temporary bridge scan implementation and into scanning.
- Provide a service to activate/deactivate scanning and a command interface for carriage speed.
- Preserve existing topic conventions where applicable (`points`, `joint_states`, `motor_cmd`).

## Non-goals

- Faro hardware SDK/API support, calibration workflow, or production sensor equivalence.
- Point-cloud merging, filtering, or file capture.
- A separate scanner simulator or duplicate MuJoCo world.

## Approaches considered

1. Keep all behavior in the bridge: smallest immediate edit, but does not create the requested ROS packages or separable interfaces.
2. Make packages own independent MuJoCo state: creates ROS boundaries but duplicates world state and risks inconsistent poses.
3. Keep the bridge as simulation owner and load package-provided node components in-process against its live model/data: preserves one simulation state while putting scan and motion logic in their requested packages. Selected.

## Design

The existing bridge remains responsible for model/data creation and physics stepping. It loads scanning and motion node implementations from the ROS packages in the same process and supplies the shared MuJoCo model/data. The scene includes the supplied `src/sim/models/Hangar_model.stl` as the hangar map and collision geometry, positioned/scaled so the 80 m × 72 m floor is centered at the origin, x is the length axis, y is the width axis, and z is up. Preserve the hangar mesh as allowed environment data; do not add aircraft geometry to the scanner-facing scene or expose aircraft ground truth. Validate the STL's units, orientation, watertightness, and MuJoCo mesh/collision behavior before relying on it for clearance. Scanning owns ray generation, raycasts, activation/deactivation, and `PointCloud2` publishing. The simulation's existing scan defaults are retained as the initial simulation configuration, not represented as real Faro specifications. Motion owns a minimal carriage slide axis in the scene and the command interface to set its track speed; the bridge applies that command to the simulated state. The exact ROS interface names and message/service choices are builder calls, keeping existing conventions where practical.

## Integration and validation

Add package manifests, Python package metadata/entry points, and package tests only as needed for ROS 2 Jazzy and current dependencies. Update the Docker build/install path only if required for colcon discovery. Ensure the hangar STL is available at runtime in the container and validate its MuJoCo load, scale, orientation, and collision/raycast behavior. Check that both packages build in the project container/workspace, scanner activation gates clouds, raycast geometry is valid, and motion commands move the sensor pose observed by scanning. Existing uncommitted Docker/Compose/simulation files must be preserved.
