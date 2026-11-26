# Sensor configuration design

## Intent
Provide one root-level YAML file for simulator rail scan speed and scanner parameters, mounted into the simulation container so operators can tune it without editing code.

## Goals / non-goals
- Configure route speed and crane lane spacing, plus scanner angular sample counts, range, simulated range-noise standard deviation, and publication frequency.
- Preserve current defaults as the initial YAML values.
- Save one raw point-cloud file per complete floor-rail pass, aggregating all scan frames from that rail movement before moving to the next rail; do not write one raw file per published frame.
- Continue saving the overhead gantry pass data as one capture per complete X traversal (one file per gantry pass); finalize the merged cloud from the bounded set of raw capture files.
- Do not claim to configure optical exposure: the simulator has no exposure model, and no hardware driver parameter API is present in this repository.
- Keep configuration scoped to the simulator; hardware launch behavior remains unchanged.

## Approaches
1. Add YAML and load it in the simulation bridge (selected): directly wires the requested values through existing constructors, minimal code and one source of truth.
2. Add YAML as an unused mounted template: lower code impact, but values would not control behavior.
3. Put values in Compose environment variables: supported by containers but not the requested YAML configuration file.

## Design
Create `sensors.yml` beside `compose.yml`, bind it read-only to `/ros_ws/sensors.yml` for `sim_bridge`, and have `MujocoBridge` load it with `yaml.safe_load`. Configure `scan_sequence.speed_mps` (0.5), `scan_sequence.crane_y_step_m` (8.0), and shared `scanner` settings: `horizontal_samples` (180), `vertical_channels` (8), `max_range_m` (10.0), `range_noise_std_m` (0.01), and `scan_hz` (10.0). Retain the existing constructor defaults for direct callers; validate malformed/missing config rather than silently misconfiguring when the bridge explicitly loads the file. Verify by a focused configuration test and Compose config rendering.

## Constraints / risks
The YAML reader must be available in the ROS runtime image; add the appropriate existing-platform dependency if absent. These are simplified simulation controls, not confirmed Focus Core optical or exposure settings. Rail speed in an automatic route is subject to MuJoCo actuator limits.
