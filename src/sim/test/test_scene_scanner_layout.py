import unittest
from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = Path(__file__).parents[1] / "models" / "scene.xml"


class ScannerLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def test_three_independent_floor_rails_carry_upward_facing_scanners(self):
        for index, expected_y in enumerate((-24.0, 0.0, 24.0), start=1):
            joint_name = "track" if index == 1 else f"floor_rail_{index}"
            actuator_name = "track_velocity" if index == 1 else f"floor_rail_{index}_velocity"
            site_name = f"floor_lidar_{index}"
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"floor_rail_{index}_beam")
            self.assertGreaterEqual(joint, 0, joint_name)
            self.assertGreaterEqual(actuator, 0, actuator_name)
            self.assertGreaterEqual(site, 0, site_name)
            self.assertGreaterEqual(geom, 0, f"floor_rail_{index}_beam")
            self.assertEqual(self.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_SLIDE)
            self.assertTrue(self.model.actuator_ctrllimited[actuator])
            self.assertAlmostEqual(self.data.site_xpos[site][1], expected_y, places=5)
            scan_axis = self.data.site_xmat[site].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(scan_axis, [0.0, 0.0, 1.0], atol=1e-6)

    def test_floor_scanners_start_above_rails_and_face_upward(self):
        for index in range(1, 4):
            rail = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"floor_rail_{index}_beam"
            )
            site = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"floor_lidar_{index}"
            )
            self.assertGreaterEqual(rail, 0)
            self.assertGreaterEqual(site, 0)
            rail_top = self.model.geom_pos[rail][2] + self.model.geom_size[rail][2]
            self.assertGreater(self.data.site_xpos[site][2], rail_top)
            scan_axis = self.data.site_xmat[site].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(scan_axis, [0.0, 0.0, 1.0], atol=1e-6)

    def test_crane_bridge_has_x_y_travel_and_fourth_downward_scanner(self):
        for joint_name, actuator_name in (
            ("crane_bridge_x", "crane_bridge_velocity"),
            ("crane_trolley_y", "crane_trolley_velocity"),
        ):
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.assertGreaterEqual(joint, 0, joint_name)
            self.assertGreaterEqual(actuator, 0, actuator_name)
            self.assertEqual(self.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_SLIDE)

        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.assertGreaterEqual(site, 0)
        self.assertAlmostEqual(self.data.site_xpos[site][0], -11.0, places=5)
        self.assertAlmostEqual(self.data.site_xpos[site][1], 6.0, places=5)
        self.assertAlmostEqual(self.data.site_xpos[site][2], 22.05, places=5)
        scan_axis = self.data.site_xmat[site].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(scan_axis, [0.0, 0.0, -1.0], atol=1e-6)

    def test_all_four_scanner_sites_are_unique(self):
        names = ["floor_lidar_1", "floor_lidar_2", "floor_lidar_3", "lidar"]
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
        self.assertTrue(all(site_id >= 0 for site_id in ids))
        self.assertEqual(len(set(ids)), 4)

    def test_crane_scanner_mount_clears_the_hook_cable(self):
        scanner = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "overhead_scanner_body"
        )
        cable = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "crane_hook_cable"
        )
        self.assertGreaterEqual(scanner, 0)
        self.assertGreaterEqual(cable, 0)
        x_clearance = (
            abs(self.data.geom_xpos[scanner][0] - self.data.geom_xpos[cable][0])
            - self.model.geom_size[scanner][0]
            - self.model.geom_size[cable][0]
        )
        self.assertGreater(x_clearance, 0.0, "overhead scanner overlaps hook cable")


if __name__ == "__main__":
    unittest.main()
