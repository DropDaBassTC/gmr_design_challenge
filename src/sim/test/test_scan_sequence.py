import unittest
from pathlib import Path
import mujoco

from scan_sequence import ScanSequence

SCENE_PATH = Path(__file__).parents[1] / "models" / "scene.xml"
AXES = {
    "floor_rail_1": "track",
    "floor_rail_2": "floor_rail_2",
    "floor_rail_3": "floor_rail_3",
    "crane_x": "crane_bridge_x",
    "crane_y": "crane_trolley_y",
}


class FakeScanner:
    def __init__(self):
        self.active = False

    def _set_active(self, request, response):
        self.active = request.data
        response.success = True
        response.message = "active" if request.data else "inactive"
        return response


class ScanSequenceTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.scanners = {
            name: FakeScanner()
            for name in ("floor_rail_1", "floor_rail_2", "floor_rail_3", "crane")
        }
        self.sequence = ScanSequence(self.model, self.data, self.scanners, None)

    def set_axis_position(self, name, bound):
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, AXES[name])
        qpos = self.model.jnt_qposadr[joint]
        dof = self.model.jnt_dofadr[joint]
        self.data.qpos[qpos] = self.model.jnt_range[joint][bound]
        self.data.qvel[dof] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_axis_target(self, name, target):
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, AXES[name])
        self.data.qpos[self.model.jnt_qposadr[joint]] = target
        self.data.qvel[self.model.jnt_dofadr[joint]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def test_floor_scanners_scan_outbound_and_return_before_next_rail(self):
        self.sequence.phase = "floor_scan"
        self.scanners["floor_rail_1"].active = True
        self.assertEqual(
            [name for name, scanner in self.scanners.items() if scanner.active],
            ["floor_rail_1"],
        )
        self.sequence.step()
        rail_1 = self.sequence.actuator_ids["floor_rail_1"]
        self.assertEqual(self.data.ctrl[rail_1], 0.5)

        for index in range(1, 4):
            name = f"floor_rail_{index}"
            actuator = self.sequence.actuator_ids[name]
            self.set_axis_position(name, 1)
            self.sequence.step()
            self.assertFalse(self.scanners[name].active)
            self.assertEqual(self.sequence.phase, "floor_return")
            self.sequence.step()
            self.assertEqual(self.data.ctrl[actuator], -0.5)
            self.set_axis_position(name, 0)
            self.sequence.step()
            if index < 3:
                next_name = f"floor_rail_{index + 1}"
                self.assertEqual(self.sequence.phase, "floor_scan")
                self.assertEqual(
                    [key for key, scanner in self.scanners.items() if scanner.active],
                    [next_name],
                )
            else:
                self.assertEqual(self.sequence.phase, "crane_scan_forward")
                self.assertTrue(self.scanners["crane"].active)

    def test_crane_scans_x_forward_shifts_y_then_scans_x_backward(self):
        self.sequence.crane_y_step = 56.0
        self.sequence.phase = "crane_scan_forward"
        self.sequence.floor_index = 3
        self.scanners["floor_rail_1"].active = False
        self.scanners["crane"].active = True

        self.sequence.step()
        crane_x = self.sequence.actuator_ids["crane_x"]
        crane_y = self.sequence.actuator_ids["crane_y"]
        self.assertEqual(self.data.ctrl[crane_x], 0.5)
        self.set_axis_position("crane_x", 1)
        self.sequence.step()
        self.assertFalse(self.scanners["crane"].active)
        self.assertEqual(self.sequence.phase, "crane_shift_y")
        self.sequence.step()
        self.assertEqual(self.data.ctrl[crane_y], 0.5)
        self.set_axis_position("crane_y", 1)
        self.sequence.step()
        self.assertEqual(self.sequence.phase, "crane_scan_reverse")
        self.assertTrue(self.scanners["crane"].active)
        self.sequence.step()
        self.assertEqual(self.data.ctrl[crane_x], -0.5)
        self.set_axis_position("crane_x", 0)
        self.sequence.step()
        self.assertEqual(self.sequence.phase, "complete")
        self.assertFalse(self.sequence.running)
        self.assertFalse(any(scanner.active for scanner in self.scanners.values()))
        self.assertTrue(all(self.data.ctrl[actuator] == 0.0 for actuator in self.sequence.actuator_ids.values()))

    def test_crane_repeats_alternating_x_passes_across_y_travel(self):
        crane_y = self.sequence.axes["crane_y"][2]
        self.set_axis_position("crane_y", 0)
        self.set_axis_position("crane_x", 0)
        self.sequence.phase = "crane_scan_forward"
        self.sequence.floor_index = 3
        self.scanners["crane"].active = True
        directions = []
        while self.sequence.running:
            if self.sequence.phase in ("crane_scan_forward", "crane_scan_reverse"):
                direction = 1 if self.sequence.phase == "crane_scan_forward" else -1
                directions.append((direction, self.data.qpos[self.sequence.axes["crane_y"][0]]))
                self.scanners["crane"].active = False
                self.set_axis_position("crane_x", 1 if direction > 0 else 0)
                self.sequence.step()
            elif self.sequence.phase == "crane_shift_y":
                self.set_axis_target("crane_y", self.sequence.crane_y_target)
                self.sequence.step()
            else:
                self.fail(f"unexpected phase: {self.sequence.phase}")

        self.assertEqual([direction for direction, _ in directions], [1, -1] * 4)
        self.assertEqual(directions[0][1], crane_y[0])
        self.assertEqual(directions[-1][1], crane_y[1])

    def test_sequence_homes_all_axes_before_starting_first_scan(self):
        self.assertEqual(self.sequence.phase, "home")
        self.assertFalse(any(scanner.active for scanner in self.scanners.values()))
        self.sequence.step()
        for name, joint_name in AXES.items():
            actuator = self.sequence.actuator_ids[name]
            home = self.sequence.axes[name][3]
            position = self.data.qpos[self.sequence.axes[name][0]]
            if position == home:
                self.assertEqual(self.data.ctrl[actuator], 0.0)
            else:
                self.assertEqual(self.data.ctrl[actuator], -self.sequence.speed)

    def test_move_to_waits_for_axis_to_settle_before_continuing(self):
        qpos, dof, limits, _home = self.sequence.axes["floor_rail_1"]
        self.data.qpos[qpos] = limits[1] - 0.01
        self.data.qvel[dof] = 0.2
        self.assertFalse(self.sequence._move_to("floor_rail_1", 1))
        self.assertEqual(
            self.data.ctrl[self.sequence.actuator_ids["floor_rail_1"]], 0.0
        )

    def test_rejects_nonpositive_route_speed_and_y_spacing(self):
        with self.assertRaises(ValueError):
            ScanSequence(self.model, self.data, self.scanners, speed=0.0)
        with self.assertRaises(ValueError):
            ScanSequence(self.model, self.data, self.scanners, crane_y_step=-1.0)


if __name__ == "__main__":
    unittest.main()
