import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))
    J = calcJacobian(q_in)

    # Extract the linear part of the Jacobian (first 3 rows)
    J_linear = J[:3, :]  # 3x7 matrix for the linear velocity part
    J_angular = J[3:, :]  # 3x7 matrix for the angular velocity part

    # Handle NaN values in the target velocities (linear velocity v_target)
    v_mask = ~np.isnan(v_in).flatten()  # Boolean mask for valid (non-NaN) components of v_target
    J_linear_reduced = J_linear[v_mask, :]  # Use only the rows where v_target is valid
    v_target_reduced = v_in[v_mask]  # Use only the non-NaN components of v_target

    # Handle NaN values in the target velocities (angular velocity omega_target)
    omega_mask = ~np.isnan(omega_in).flatten()  # Boolean mask for valid (non-NaN) components of omega_target
    J_angular_reduced = J_angular[omega_mask, :]  # Use only the rows where omega_target is valid
    omega_target_reduced = omega_in[omega_mask]  # Use only the non-NaN components of omega_target

    J_reduced = np.vstack((J_linear_reduced, J_angular_reduced))  # Combine the linear and angular Jacobians
    v_target_reduced = np.vstack((v_target_reduced, omega_target_reduced))  # Combine the linear and angular target velocities

    dq_solution, _, _, _ = np.linalg.lstsq(J_reduced, v_target_reduced, rcond=None)
    dq_solution = dq_solution.flatten()

    dq[0, :] = dq_solution

    J_pseduo_inverse = np.linalg.pinv(J_reduced)
    null_proj_matrix = np.eye(7) - J_pseduo_inverse @ J_reduced

    null[0, :] = (null_proj_matrix @ b).flatten()


    return dq + null