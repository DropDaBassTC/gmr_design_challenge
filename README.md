# GMR Design Challenge

ROS 2 Jazzy and MuJoCo simulation for the hangar scanning system.

## Start the simulation with the MuJoCo window

Prerequisites: Docker Compose and a local X11-compatible display. The viewer
container needs `DISPLAY`, the host X11 socket at `/tmp/.X11-unix`, and a valid
`XAUTHORITY` file available to the shell running Compose. Wayland desktops
usually provide XWayland. Headless or remote shells need X11 forwarding or a
virtual display.

From the repository root, allow the local container to use the display and
start the automatic scan route:

```sh
xhost +si:localuser:root
docker compose --profile viewer run --build --rm sim_viewer
```

The route homes the carriages, scans four floor rails, then raster-scans the
overhead crane across X while shifting the trolley along Y. When the route
finishes or you stop the viewer with Ctrl-C, revoke display access:

```sh
xhost -si:localuser:root
```

The simulator can also run without a window:

```sh
docker compose --profile sim up --build
```

Do not run the `sim` and `viewer` profiles at the same time; each starts an
independent simulation world.

## Run the test suite (headless)

Tests need no display and do not open a UI. From the repository root:

```sh
docker compose --profile test run --build --rm test_headless
```

The test service sets `MUJOCO_GL=egl`, unsets display variables, and does not
mount the host display socket or Xauthority file. It runs the simulation,
scanning, and motion Python unittest suites.

## Scan outputs

After the automatic route completes, raw captures are written to
`output/raw_scans/` and the combined cloud to `output/mergedPointcloud.ply`.
The default route produces 12 raw captures: four floor-rail outbound passes
and eight overhead X traversals across the Y lanes. Files are binary PLY in
world coordinates. A completed new run replaces the merged cloud; timestamped
raw captures are retained.

Simulator controls are in `sensors.yml`. See [`src/sim/README.md`](src/sim/README.md)
for display setup, route behavior, ROS topics, configuration options, and
capture details.
