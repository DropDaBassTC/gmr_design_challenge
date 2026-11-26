import unittest
from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = Path(__file__).parents[1] / "models" / "scene.xml"
RAIL_Y = (-27.375, -9.125, 9.125, 27.375)


class ScannerLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def mesh_world_bounds(self, geom_id):
        mesh_id = self.model.geom_dataid[geom_id]
        start = self.model.mesh_vertadr[mesh_id]
        count = self.model.mesh_vertnum[mesh_id]
        vertices = self.model.mesh_vert[start:start + count]
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        world_vertices = vertices @ rotation.T + self.data.geom_xpos[geom_id]
        return world_vertices.min(axis=0), world_vertices.max(axis=0)

    def test_four_floor_rails_use_real_meshes_centered_and_evenly_spaced(self):
        for index, expected_y in enumerate(RAIL_Y, start=1):
            rail_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"floor_rail_{index}"
            )
            rail_geom = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"floor_rail_{index}_profile"
            )
            self.assertGreaterEqual(rail_body, 0)
            self.assertGreaterEqual(rail_geom, 0)
            np.testing.assert_allclose(
                self.model.body_pos[rail_body], [39.725, expected_y, 0.018], atol=1e-5
            )
            self.assertEqual(self.model.geom_type[rail_geom], mujoco.mjtGeom.mjGEOM_MESH)
            low, high = self.mesh_world_bounds(rail_geom)
            np.testing.assert_allclose([low[0], high[0]], [-39.875, 39.875], atol=0.03)
            self.assertAlmostEqual(low[1], expected_y - 0.0521, places=3)
            self.assertAlmostEqual(high[1], expected_y + 0.0521, places=3)

    def test_carriages_start_at_each_rail_home_and_travel_along_negative_x(self):
        for index, expected_y in enumerate(RAIL_Y, start=1):
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"floor_carriage_{index}"
            )
            joint_name = "track" if index == 1 else f"floor_rail_{index}"
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            carriage = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"floor_carriage_{index}_mesh"
            )
            site = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"floor_lidar_{index}"
            )
            self.assertGreaterEqual(body, 0)
            self.assertGreaterEqual(joint, 0)
            self.assertGreaterEqual(carriage, 0)
            self.assertGreaterEqual(site, 0)
            self.assertEqual(self.model.geom_type[carriage], mujoco.mjtGeom.mjGEOM_MESH)
            np.testing.assert_allclose(
                self.model.body_pos[body], [39.725, expected_y, 0.0444], atol=1e-5
            )
            np.testing.assert_allclose(self.model.jnt_axis[joint], [-1.0, 0.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(self.model.jnt_range[joint], [0.0, 79.45], atol=1e-5)
            self.assertAlmostEqual(self.data.xpos[body][0], 39.725, places=3)
            self.assertAlmostEqual(self.data.xpos[body][1], expected_y, places=5)
            self.assertGreater(self.data.site_xpos[site][2], 0.018)
            scan_axis = self.data.site_xmat[site].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            np.testing.assert_allclose(scan_axis, [0.0, 0.0, 1.0], atol=1e-6)

    def test_crane_bridge_has_x_y_travel_and_fifth_downward_scanner(self):
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
        np.testing.assert_allclose(self.data.site_xpos[site], [-11.0, 6.0, 22.05], atol=1e-5)
        scan_axis = self.data.site_xmat[site].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(scan_axis, [0.0, 0.0, -1.0], atol=1e-6)

    def test_all_five_scanner_sites_are_unique(self):
        names = [f"floor_lidar_{index}" for index in range(1, 5)] + ["lidar"]
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) for name in names]
        self.assertTrue(all(site_id >= 0 for site_id in ids))
        self.assertEqual(len(set(ids)), 5)

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

    def test_crane_scanner_can_reach_both_x_travel_limits_without_wall_contact(self):
        bridge_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "crane_bridge_x"
        )
        scanner = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "overhead_scanner_body"
        )
        wall = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "hangar_wall_x_pos"
        )
        qpos = self.model.jnt_qposadr[bridge_joint]
        for limit in self.model.jnt_range[bridge_joint]:
            self.data.qpos[qpos] = limit
            mujoco.mj_forward(self.model, self.data)
            fromto = np.empty(6)
            distance = mujoco.mj_geomDistance(
                self.model, self.data, scanner, wall, 10.0, fromto
            )
            self.assertGreaterEqual(distance, 0.0, "crane scanner intersects the hangar wall")


if __name__ == "__main__":
    unittest.main()
