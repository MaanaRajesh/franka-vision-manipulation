import numpy as np
import sys
sys.path.append("/home/student/meam520_ws/src/meam520_labs")
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE

    # Initialize the FK object to access forward kinematics and transformation matrices
    fk = FK()
    
    # Perform forward kinematics to get joint positions and the end-effector transformation matrix
    joint_positions, T0e = fk.forward(q_in)  # Joint positions is a 8x3 matrix

    # Extract the position of the end-effector
    p_e = T0e[:3, 3]  # End-effector position from the transformation matrix

    # Initialize transformation matrix from the base frame
    T = np.identity(4)  # Start from the base frame
    z = np.array([0, 0, 1])  # Initial z-axis direction (aligned with the base frame)

    # Compute Jacobian columns for each joint
    j = 0
    for i in range(7):
        # Update the transformation matrix for the next joint
        a = fk.DH_params[j][0]
        alpha = fk.DH_params[j][1]
        d = fk.DH_params[j][2]
        if j == 0 or j == 5:
            theta = fk.DH_params[j][3]  # Add joint angle to DH theta
        else:
            theta = q_in[i-1] + fk.DH_params[j][3]  # Add joint angle to DH theta
        T_joint = fk.dh_transform(theta, d, a, alpha)
        T = np.dot(T, T_joint)

        if j == 5:
            j = j + 1
            a = fk.DH_params[j][0]
            alpha = fk.DH_params[j][1]
            d = fk.DH_params[j][2]
            theta = q_in[i-1] + fk.DH_params[j][3]  # Add joint angle to DH theta
            T_joint = fk.dh_transform(theta, d, a, alpha)
            T = np.dot(T, T_joint)

        # Extract the current joint position
        p_i = joint_positions[i, :]  # Position of the ith joint
        
        # Extract the rotation axis (z-axis of the current joint's frame)
        z_i = T[0:3, 2]  # z-axis of the current joint (rotation axis)

        # Linear velocity component
        J_v = np.cross(z_i, (p_e - p_i))

        # Angular velocity component for revolute joints
        J_w = z_i

        # Set the Jacobian columns (first 3 rows for linear, last 3 rows for angular velocity)
        J[0:3, i] = J_v  # Linear velocity part
        J[3:6, i] = J_w  # Angular velocity part

        j = j + 1

    return J

if __name__ == '__main__':
    # q= np.array([0, 0, 0, 0, 0, 0, 0])
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
