import sys
sys.path.append("/home/student/meam520_ws/src/meam520_labs")
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from lib.calculateFK import FK
from lib.IK_position_null import IK
import time, copy

def place_red(offset, T_w_b, target_r, q_pseudo):

    target_r[2,3] = offset
    seed  = q_pseudo
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_r, seed, method='J_pseudo', alpha=.5)
    arm.safe_move_to_position(q_pseudo)
    arm.exec_gripper_cmd(0.5, force=50)

    return q_pseudo

def pick_pose(T0e, detector, q_pseudo_orig):
    H_ee_camera = detector.get_H_ee_camera()
    global list_static_blocks
    start_time = int(time.time())
    T_obj_to_end=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    while True:
        if len(list_static_blocks)>0:
            break
        duration = int(time.time()) - start_time
        if duration == 10:
            return q_pseudo_orig
        list_static_blocks = detector.get_detections()
    
    (name_block, pose_block) = list_static_blocks.pop()
    if (np.abs(pose_block[2,1])>0.93):
        pose_block[:3,:3]=np.concatenate((pose_block[:3,2].reshape(3,1),pose_block[:3,0].reshape(3,1),pose_block[:3,1].reshape(3,1)),axis=1)
    if (np.abs(pose_block[2,0])>0.93):
        pose_block[:3,:3]=np.concatenate((pose_block[:3,1].reshape(3,1),pose_block[:3,2].reshape(3,1),pose_block[:3,0].reshape(3,1)),axis=1)
    seed = q_pseudo_orig
    target_pose = H_ee_camera @ pose_block
    if (target_pose[2,2]<0):
        target_pose = target_pose @ T_obj_to_end
    print('Block:', name_block, 'Pose:', target_pose, "Transform:", H_ee_camera)
    a = copy.deepcopy(target_pose)
    # # Calculate pitch (theta_y)
    # theta_y = np.arcsin(-R[2, 0])

    # # Check for gimbal lock (singularities)
    # if np.abs(np.cos(theta_y)) > 1e-6:
    #     theta = np.arctan(R[1, 0]/R[0, 0])
    # else:
    #     # Gimbal lock: handle accordingly (e.g., set one angle to zero)
    #     theta = np.arctan(-R[0, 1]/R[1, 1])
    #     # theta = np.arctan(R[1, 0]/R[0, 0])
    # print('Theta_my:', theta/np.pi*180)
    x,y = np.argwhere(np.abs(a[:3,:3]) == np.max(np.abs(a[:3,:3])))[0]
    a = np.delete(a, x, axis=0)
    a = np.delete(a, y, axis=1)
    # theta_ori = np.abs(np.arccos(np.abs(a[0][0])))
    theta_ori = np.arccos(a[0][0])
    if np.abs(np.sin(theta_ori) - a[1][0]) > 1e-3:
        theta_ori = -theta_ori
    if theta_ori > np.pi/2:
        theta_ori = theta_ori - np.pi/2
    elif theta_ori < -np.pi/2:
        theta_ori = theta_ori + np.pi/2
    if theta_ori > np.pi/4:
        theta_ori = theta_ori - np.pi/2
    elif theta_ori < -np.pi/4:
        theta_ori = theta_ori + np.pi/2
    # if np.abs(a[0][0]) - np.abs(a[1][1]) < 1e-5 and a[0][0]*a[1][1] < 0:
    #     if a[0][1] * a[0][0] > 0:
    #         theta = np.arccos(a[0][0]) - np.pi/2
    #     else:
    #         theta = np.arccos(a[1][1]) - np.pi/2
    # elif np.abs(a[0][1]) - np.abs(a[1][0]) < 1e-5 and a[0][1]*a[1][0] < 0:
    #     if a[0][0] * a[0][1] > 0:
    #         theta = np.arccos(a[0][1]) - np.pi/2
    #     else:
    #         theta = np.arccos(a[1][0]) - np.pi/2
    # print('Theta_my:', theta/np.pi*180)
    # theta_ori = np.arccos(a[0,0]) - np.pi/2
    print('Theta_original:', theta_ori/np.pi*180)
    theta = theta_ori

    T = np.array([[np.cos(theta), -np.sin(theta), 0,0],[np.sin(theta), np.cos(theta),0,0],[0,0,1,0],[0,0,0,1]])
    T[:,3] = target_pose[:,3]
    T_trans = np.array([[1,0,0,0],[0,1,0,-0.01],[0,0,1,0],[0,0,0,1]])
    T = T0e @ T @ T_trans
    T_step1 = copy.deepcopy(T)

    # Step 1 (Change orientation)
    T_step1[2,3] = 0.3
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T_step1, seed, method='J_pseudo', alpha=.5)
    arm.safe_move_to_position(q_pseudo)
    seed = q_pseudo

    # Step 2 (Grab the block)
    T[2,3] = 0.235
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T, seed, method='J_pseudo', alpha=.5)
    arm.safe_move_to_position(q_pseudo)
    return q_pseudo

def inverse_homogeneous(H):
    R = H[:3,:3]
    p = H[:3,3]
    H_inv = np.zeros((4,4))
    H_inv[:3,:3] = R.T
    H_inv[:3,3] = -R.T @ p
    H_inv[-1,-1] = 1
    return H_inv

def pick_pose_dynamic(T0e, detector, q_pseudo_orig):
    H_ee_camera = detector.get_H_ee_camera()
    print('H_ee_camera:', H_ee_camera)
    T_obj_to_end=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    while True:
        list_dynamic_blocks = detector.get_detections()
        select = None
        if len(list_dynamic_blocks) > 0:
            for (i, block) in enumerate(list_dynamic_blocks):
                (name_block, pose_block) = block
                T_w_b = np.array([[1,0,0,0],\
                            [0,1,0,0.990],\
                            [0,0,1,0],\
                            [0,0,0,1]])
                # T_w_b = inverse_homogeneous(T_b_w)
                T_w_c = inverse_homogeneous(H_ee_camera) @ inverse_homogeneous(T0e) @ T_w_b
                T_c_w = inverse_homogeneous(T_w_c)
                pose_block_world = T_c_w @ pose_block
                x = pose_block_world[0,3]
                if x < -0.04 and x > -0.10:
                    select = i
        if len(list_dynamic_blocks) > 0 and select is not None:
            pose_block = list_dynamic_blocks[select][1]
            T_w_b = np.array([[1,0,0,0],\
                            [0,1,0,0.990],\
                            [0,0,1,0],\
                            [0,0,0,1]])
            # T_w_b = inverse_homogeneous(T_b_w)
            T_w_c = inverse_homogeneous(H_ee_camera) @ inverse_homogeneous(T0e) @ T_w_b
            T_c_w = inverse_homogeneous(T_w_c)
            pose_block_world = T_c_w @ pose_block
            angle = 8
            T_pred = np.array([[np.cos(0.017 * angle), -np.sin(0.017 * angle), 0,0],[np.sin(0.017 * angle), np.cos(0.017 * angle),0,0],[0,0,1,0],[0,0,0,1]])
            pose_pred = T_pred @ pose_block_world
            print('Pose original in world frame:', '\n', pose_block_world, '\n', 'Pose predicted in world frame:', '\n', pose_pred)
            # pose_pred = T_w_b @ pose_pred
            pose_pred = T_w_c @ pose_pred
            print('Pose original in camera frame:', '\n', pose_block)
            # print('Pose original in camera frame:', '\n', pose_block, '\n', 'Pose predicted in base frame:', '\n', pose_pred)


            seed = q_pseudo_orig
            if (np.abs(pose_block[2,1])>0.93):
                pose_pred[:3,:3]=np.concatenate((pose_pred[:3,2].reshape(3,1),pose_pred[:3,0].reshape(3,1),pose_pred[:3,1].reshape(3,1)),axis=1)
            if (np.abs(pose_block[2,0])>0.93):
                pose_pred[:3,:3]=np.concatenate((pose_pred[:3,1].reshape(3,1),pose_pred[:3,2].reshape(3,1),pose_pred[:3,0].reshape(3,1)),axis=1)

            target_pose = H_ee_camera @ pose_pred
            if (target_pose[2,2]<0):
                target_pose = target_pose @ T_obj_to_end

            a = copy.deepcopy(target_pose)
            x,y = np.argwhere(np.abs(a[:3,:3]) == np.max(np.abs(a[:3,:3])))[0]
            a = np.delete(a, x, axis=0)
            a = np.delete(a, y, axis=1)
            # theta = np.abs(np.arccos(np.abs(a[0][0])))
            theta = np.arccos(a[0][0])

            if np.abs(np.sin(theta) - a[1][0]) > 1e-3:
                theta = -theta
            if theta > np.pi/2:
                theta = theta - np.pi/2
            elif theta < -np.pi/2:
                theta = theta + np.pi/2
            if theta > np.pi/4:
                theta = theta - np.pi/2
            elif theta < -np.pi/4:
                theta = theta + np.pi/2

            T = np.array([[np.cos(theta), -np.sin(theta), 0,0],[np.sin(theta), np.cos(theta),0,0],[0,0,1,0],[0,0,0,1]])
            T[:,3] = target_pose[:,3]
            T_trans = np.array([[1,0,0,0.02],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            # T_trans = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            T = T0e @ T @ T_trans
            T_step1 = copy.deepcopy(T)
            # T_step1[2,3] = 0.225
            T_step1[2,3] = 0.225 - 0.05
            q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T_step1, seed, method='J_pseudo', alpha=.5)
            if success_pseudo:
                break
    arm.safe_move_to_position(q_pseudo)
    arm.exec_gripper_cmd(0, force=100)
    print('Grab the block')
    return q_pseudo_orig


def check_grip(arm, threshold):
    """
    This function checks if the gripper has caught the block.
    :param arm: the arm class
    :param threshold: threshold to differentiate if the block is gripped
    :return: true for block gripped, false for not gripped
    """
    gripper_state = arm.get_gripper_state()
    # Compute the distance between 2 gripper fingers
    actual_gripper_width = sum(gripper_state['position'])
    print(gripper_state)
    # Check the finger distance to determine if the block is gripped successfully
    if actual_gripper_width < threshold:
        return False
    else:
        return True


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    if team == 'red':
        # STUDENT CODE HERE
        gripper_threshold = 0.025
        fk = FK()
        ik = IK()
        list_static_blocks = []
        list_dynamic_blocks = []
        i_offset = 0
        
        print('STATIC BLOCKS')

        ## static block code
        offset = [0.232, 0.292, 0.342, 0.392, 0.442, 0.492, 0.542, 0.592]#[0.227, 0.277, 0.327, 0.377]
        T_w_b = np.array([[1,0,0,0],\
                            [0,1,0,0.990],\
                            [0,0,1,0],\
                            [0,0,0,1]])
        target_black = np.array([
                [1,0,0,0.532],
                [0,-1,0,-1.159],
                [0,0,-1,0.6],
                [0,0,0, 1],
        ])
        target_black = T_w_b @ target_black
        initial_guess = np.array([-0.1680977, 0.24084676, -0.16153996, -1.04057937, 0.04007242, 1.27862972, 0.47189346])
        q_pseudo_bc, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_black, initial_guess, method='J_pseudo', alpha=.5)
        # target_red = np.array([
        #             [1,0,0,0.532],
        #             [0,-1,0,-0.9],
        #             [0,0,-1,0.6],
        #             [0,0,0, 1],
        #         ])
        target_red = np.array([
                    [1,0,0,0.562],
                    [0,-1,0,-0.81],
                    [0,0,-1,0.6],
                    [0,0,0, 1],
                ])
        target_r = T_w_b @ target_red
        initial_guess = np.array([0.26322347, 0.29056701, 0.02899833, -1.24011962, -0.00831355, 1.53057261, 1.07607072])
        q_pseudo_rc, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_r, initial_guess, method='J_pseudo', alpha=.5)
        arm.exec_gripper_cmd(0.5, force=50)
        arm.safe_move_to_position(q_pseudo_bc)

        for i in range(4):
            _, T0e = fk.forward(q_pseudo_bc)
            q_pseudo = pick_pose(T0e, detector, q_pseudo_bc)
            arm.exec_gripper_cmd(0, force=50)

            arm.safe_move_to_position(q_pseudo_bc)
            if check_grip(arm, gripper_threshold):
                q_pseudo = place_red(offset[i-i_offset], T_w_b, target_r, q_pseudo_rc)
                arm.safe_move_to_position(q_pseudo_rc)
            else:
                i_offset += 1
                arm.exec_gripper_cmd(0.5, force=50)


        print('DYNAMIC BLOCKS')

        target_detect = np.array([
            [0,1,0,0],
            [1,0,0,-0.265],
            [0,0,-1,0.45],
            [0,0,0,1],])
        # target_detect = np.array([
        #         [1,0,0,0],
        #         [0,-1,0,-0.295],
        #         [0,0,-1,0.6],
        #         [0,0,0,1],])
        T_w_b = np.array([[1,0,0,0],\
                        [0,1,0,0.990],\
                        [0,0,1,0],\
                        [0,0,0,1]])
        target_detect = T_w_b @ target_detect
        initial_guess = np.array([1.4843889, 0.74249556, 0.20540477, -0.65803304, -0.14061001, 1.39170198, 0.82609649])
        q_pseudo_db, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_detect, initial_guess, method='J_pseudo', alpha=.5)
        q_pseudo_db_up = np.array([1.4843889, 0.74249556, 0.20540477, -0.15, -0.14061001, 1.39170198, 0.82609649])
        # target_red = np.array([
        #             [1,0,0,0.532],
        #             [0,-1,0,-0.9],
        #             [0,0,-1,0.6],
        #             [0,0,0, 1],
        #         ])
        # target_r_1 = T_w_b @ target_red_stage_1
        # initial_guess = np.array([0.26322347, 0.29056701, 0.02899833, -1.24011962, -0.00831355, 1.53057261, 1.07607072])
        # q_pseudo_temp_rc, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_r, initial_guess, method='J_pseudo', alpha=.5)
        for i in range(4, 12):
            arm.safe_move_to_position(q_pseudo_db)
            _, T0e = fk.forward(q_pseudo_db)
            print('Move to dynamic detect position')
            arm.exec_gripper_cmd(0.5, force=50)
            pick_pose_dynamic(T0e, detector, q_pseudo_db)
            arm.safe_move_to_position(q_pseudo_db_up)
            if check_grip(arm, gripper_threshold):
                q_pseudo = place_red(offset[i-i_offset], T_w_b, target_r, q_pseudo_rc)
                arm.safe_move_to_position(q_pseudo_rc)
            else:
                i_offset += 1

        # print('DYNAMIC BLOCKS SECOND STAGE')
        # target_red_stage_1 = np.array([
        #             [1,0,0,0.562],
        #             [0,-1,0,-0.75],
        #             [0,0,-1,0.6],
        #             [0,0,0, 1],
        #         ])
        # target_r_1 = T_w_b @ target_red_stage_1
        # initial_guess = np.array([0.26322347, 0.29056701, 0.02899833, -1.24011962, -0.00831355, 1.53057261, 1.07607072])
        # q_pseudo_temp_rc, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_r_1, initial_guess, method='J_pseudo', alpha=.5)
        # for i in range(4, 8):
        #     arm.safe_move_to_position(q_pseudo_temp_rc)
        #     _, T0e = fk.forward(q_pseudo_temp_rc)
        #     q_pseudo = pick_pose_dynamic_second_stage(T0e, detector, q_pseudo_temp_rc)
        #     arm.exec_gripper_cmd(0, force=50)

        #     arm.safe_move_to_position(q_pseudo_temp_rc)
        #     q_pseudo = place_red(offset[i], T_w_b, target_r_2, q_pseudo_rc)
        #     arm.safe_move_to_position(q_pseudo_rc)

    # get the transform from camera to panda_end_effector
    # H_ee_camera = detector.get_H_ee_camera()

    # # Detect some blocks...
    # for (name, pose) in detector.get_detections():
    #      print(name,'\n',pose)

    # Uncomment to get middle camera depth/rgb images
    # mid_depth = detector.get_mid_depth()
    # mid_rgb = detector.get_mid_rgb()

    # Move around...

    # END STUDENT CODE