import numpy as np
from lib.calcJacobian import calcJacobian


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    ## STUDENT CODE STARTS HERE
    # Step 1: Compute the relative rotation matrix
    R_rel = np.dot(R_curr.T, R_des)

    # Step 2: Compute the skew-symmetric part of the relative rotation matrix
    S = 0.5 * (R_rel - R_rel.T)

    # Step 3: Extract the vector components from the skew-symmetric matrix
    omega = np.array([S[2, 1], S[0, 2], S[1, 0]])
    omega = np.dot(R_curr, omega)

    return omega