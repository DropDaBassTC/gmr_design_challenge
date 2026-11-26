"""Automatic simulated continuous-scan route for the five MuJoCo scanners."""

import mujoco
from std_srvs.srv import SetBool


AXES = {
    "floor_rail_1": ("track_speed_cmd", "track"),
    "floor_rail_2": ("floor_rail_2_speed_cmd", "floor_rail_2"),
    "floor_rail_3": ("floor_rail_3_speed_cmd", "floor_rail_3"),
    "floor_rail_4": ("floor_rail_4_speed_cmd", "floor_rail_4"),
    "crane_x": ("crane_bridge_speed_cmd", "crane_bridge_x"),
    "crane_y": ("crane_trolley_speed_cmd", "crane_trolley_y"),
}


class ScanSequence:
    """Home axes, scan each floor rail out-and-back, then raster the crane in X/Y."""

    def __init__(
        self, model, data, scanners, motion_node=None, speed=0.5,
        crane_y_step=8.0, enabled=True, on_complete=None,
        on_capture_start=None, on_capture_finish=None,
    ):
        if speed <= 0 or crane_y_step <= 0:
            raise ValueError("scan speed and crane Y step must be positive")
        self.model = model
        self.data = data
        self.scanners = scanners
        self.motion_node = motion_node
        self.on_complete = on_complete
        self.on_capture_start = on_capture_start
        self.on_capture_finish = on_capture_finish
        self.speed = speed
        self.crane_y_step = crane_y_step
        self.running = enabled
        self.phase = "home" if enabled else "disabled"
        self.floor_index = 0
        self.authorized_scanner = None
        self._stopping = {axis: False for axis in AXES}
        self._goals = {axis: None for axis in AXES}
        self.axes = {}
        self.actuator_ids = {}
        for axis, (topic, joint_name) in AXES.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint < 0:
                raise ValueError(f"MuJoCo scene is missing joint '{joint_name}'")
            joint_range = model.jnt_range[joint].copy()
            home = float(joint_range[0])
            self.axes[axis] = (
                model.jnt_qposadr[joint], model.jnt_dofadr[joint],
                joint_range, home,
            )
            actuator_name = {
                "floor_rail_1": "track_velocity",
                "floor_rail_2": "floor_rail_2_velocity",
                "floor_rail_3": "floor_rail_3_velocity",
                "floor_rail_4": "floor_rail_4_velocity",
                "crane_x": "crane_bridge_velocity",
                "crane_y": "crane_trolley_velocity",
            }[axis]
            actuator = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator < 0:
                raise ValueError(f"MuJoCo scene is missing actuator '{actuator_name}'")
            if motion_node is not None:
                if motion_node.actuator_ids[topic] != actuator:
                    raise ValueError(f"motion topic '{topic}' maps to the wrong actuator")
            self.actuator_ids[axis] = actuator

    def _set_control(self, axis, speed):
        actuator = self.actuator_ids[axis]
        low, high = self.model.actuator_ctrlrange[actuator]
        self.data.ctrl[actuator] = min(high, max(low, speed))

    def _move_to(self, axis, bound):
        return self._move_to_target(axis, self.axes[axis][2][bound])

    def _home_axis(self, axis):
        return self._move_to_target(axis, self.axes[axis][3])

    def _move_to_target(self, axis, target):
        qpos, dof, _limits, _home = self.axes[axis]
        if self._goals[axis] != target:
            self._goals[axis] = target
            self._stopping[axis] = False
        position = self.data.qpos[qpos]
        velocity = self.data.qvel[dof]
        direction = 1.0 if target > position else -1.0
        if self._stopping[axis] or abs(target - position) < 0.03:
            self._stopping[axis] = True
            self._set_control(axis, 0.0)
            return abs(velocity) < 1e-3
        self._set_control(axis, direction * self.speed)
        return False

    def _activate(self, scanner_name):
        self.authorized_scanner = scanner_name
        request, response = SetBool.Request(), SetBool.Response()
        request.data = True
        self.scanners[scanner_name]._set_active(request, response)
        self.authorized_scanner = None
        if not response.success:
            raise RuntimeError(f"sequence could not activate {scanner_name}: {response.message}")

    def _begin_floor_scan(self):
        scanner_name = f"floor_rail_{self.floor_index + 1}"
        self.phase = "floor_scan"
        self._activate(scanner_name)
        if self.on_capture_start is not None:
            self.on_capture_start(scanner_name)

    def _begin_crane_scan(self):
        self._begin_crane_pass("crane_scan_forward")

    def _begin_crane_pass(self, phase):
        self.phase = phase
        self._activate("crane")
        if self.on_capture_start is not None:
            self.on_capture_start("crane")

    def _deactivate(self, scanner_name):
        request, response = SetBool.Request(), SetBool.Response()
        request.data = False
        self.scanners[scanner_name]._set_active(request, response)
        if self.on_capture_finish is not None:
            self.on_capture_finish(scanner_name)

    def _advance_crane_y(self, next_phase, finish_at_limit=False):
        y_position = self.data.qpos[self.axes["crane_y"][0]]
        y_max = self.axes["crane_y"][2][1]
        if finish_at_limit and y_position >= y_max - 0.03:
            self._complete()
            return
        self.crane_y_target = min(y_position + self.crane_y_step, y_max)
        self.next_crane_phase = next_phase
        self.phase = "crane_shift_y"

    def _complete(self):
        self.phase = "complete"
        self.running = False
        for axis in AXES:
            self._set_control(axis, 0.0)
        if self.on_complete is not None:
            self.on_complete()

    def step(self):
        if not self.running:
            return

        if self.phase == "home":
            homed = [self._home_axis(axis) for axis in AXES]
            if all(homed):
                self._begin_floor_scan()
            return

        if self.phase == "floor_scan":
            axis = f"floor_rail_{self.floor_index + 1}"
            if self._move_to(axis, 1):
                self._deactivate(axis)
                self.phase = "floor_return"
            return

        if self.phase == "floor_return":
            axis = f"floor_rail_{self.floor_index + 1}"
            if self._move_to(axis, 0):
                self.floor_index += 1
                if self.floor_index < 4:
                    self._begin_floor_scan()
                else:
                    self._begin_crane_scan()
            return

        if self.phase == "crane_scan_forward":
            if self._move_to("crane_x", 1):
                self._deactivate("crane")
                self._advance_crane_y("crane_scan_reverse", finish_at_limit=True)
            return

        if self.phase == "crane_shift_y":
            if self._move_to_target("crane_y", self.crane_y_target):
                self._begin_crane_pass(self.next_crane_phase)
            return

        if self.phase == "crane_scan_reverse":
            if self._move_to("crane_x", 0):
                self._deactivate("crane")
                self._advance_crane_y("crane_scan_forward", finish_at_limit=True)
