import numpy as np

def get_link_by_name(robot, name):
    for link in robot.get_links():
        if link.get_name() == name:
            return link
    raise ValueError(f"Link {name} not found")

def joint_to_eef_aloha_agilex(robot, action14, left_ee_name, right_ee_name):
    """
    robot: sapien.Articulation for aloha-agilex
    action14: shape (14,), [l_arm6, l_gripper1, r_arm6, r_gripper1]
    left_ee_name/right_ee_name: end-effector link names in URDF
    """
    action14 = np.asarray(action14).reshape(-1)
    assert action14.shape[0] == 14

    left_arm_q = action14[:6]
    right_arm_q = action14[7:13]

    # make sure articulation qpos sequence
    # [left_arm6, left_gripper, right_arm6, right_gripper]
    full_qpos = action14.copy()
    robot.set_qpos(full_qpos)

    left_link = get_link_by_name(robot, left_ee_name)
    right_link = get_link_by_name(robot, right_ee_name)

    left_pose = left_link.get_pose()
    right_pose = right_link.get_pose()

    left_eef = np.concatenate([left_pose.p, left_pose.q])   # [x,y,z,qx,qy,qz,qw]
    right_eef = np.concatenate([right_pose.p, right_pose.q])

    return {
        "left_arm_q": left_arm_q,
        "right_arm_q": right_arm_q,
        "left_eef": left_eef,
        "right_eef": right_eef,
    }