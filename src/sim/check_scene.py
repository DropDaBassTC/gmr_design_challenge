#!/usr/bin/env python3
"""Smoke-test the hangar mesh and simulated carriage in the MuJoCo scene."""
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "hangar")
    assert mesh_id >= 0, "Hangar_model.stl is not loaded as the hangar mesh"
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start : start + count]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.mesh_quat[mesh_id])
    rotation = rotation.reshape(3, 3)
    world_vertices = vertices @ rotation.T + model.mesh_pos[mesh_id]
    bounds = np.vstack((world_vertices.min(axis=0), world_vertices.max(axis=0)))
    np.testing.assert_allclose(bounds[:, 0], [-40.5, 40.5], atol=0.02)
    np.testing.assert_allclose(bounds[:, 1], [-36.5, 36.5], atol=0.02)
    np.testing.assert_allclose(bounds[:, 2], [-0.5, 31.51], atol=0.02)

    rail_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "track")
    assert rail_id >= 0, "sensor track joint is missing"
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "track_velocity")
    assert actuator_id >= 0, "track velocity actuator is missing"
    assert model.jnt_limited[rail_id] and np.allclose(model.jnt_range[rail_id], [-39.0, 39.0])

    mujoco.mj_forward(model, data)
    ray_groups = np.zeros(6, dtype=np.uint8)
    ray_groups[2] = 1
    geomid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([0.0, 0.0, 12.0]), np.array([0.0, 1.0, 0.0]),
        ray_groups, -1, -1, geomid,
    )
    assert 33.0 < distance < 37.0, f"interior hangar wall ray returned {distance} m"
    assert model.geom_group[geomid[0]] == 2, "interior ray did not hit the hangar mesh"

    print("hangar bounds, interior ray, track joint, and velocity actuator verified")


if __name__ == "__main__":
    main()
